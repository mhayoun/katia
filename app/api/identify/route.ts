import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { generateText, type LanguageModel } from "ai";
import { google } from "@ai-sdk/google";
import { auth } from "@/auth";

export const maxDuration = 60;

/**
 * Model selection:
 *  - "claude" → Anthropic Claude via Vercel AI Gateway (paid, needs a card).
 *  - otherwise → Google Gemini (free). `model` may pick a specific Gemini model;
 *    only gemini-* names are accepted (fallback to the default otherwise).
 */
const GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash"];

/**
 * Ordered list of models to try: the requested one first, then the other
 * Gemini models as fallbacks (e.g. when one returns "high demand" / 503).
 */
function pickModels(provider: string, model?: string): { id: string; model: LanguageModel }[] {
  if (provider === "claude") {
    const id = process.env.AI_GATEWAY_MODEL || "anthropic/claude-sonnet-4.5";
    return [{ id, model: id as LanguageModel }];
  }
  const safe = model && /^gemini-[a-z0-9.\-]+$/i.test(model) ? model : GEMINI_MODELS[0];
  const ids = [safe, ...GEMINI_MODELS.filter((m) => m !== safe)];
  return ids.map((id) => ({ id, model: google(id) }));
}

function buildPrompt(known: string[]): string {
  let p =
    "Identifie l'espèce d'oiseau sur cette photo. " +
    "Réponds UNIQUEMENT par le nom courant de l'espèce en hébreu, " +
    "sans phrase, sans ponctuation, sans explication.";
  if (known.length) {
    p +=
      " Voici une liste d'espèces DÉJÀ connues (référence). " +
      "Si l'oiseau de la photo correspond à l'une d'elles, réponds EXACTEMENT " +
      "avec ce nom de la liste (orthographe identique). Sinon, donne ton meilleur " +
      "nom en hébreu. Liste : " +
      known.slice(0, 300).join(", ") +
      ".";
  }
  p += " Si ce n'est pas un oiseau, réponds exactement: ?";
  return p;
}

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  }

  let image: string | undefined;
  let provider = "gemini";
  let model: string | undefined;
  let known: string[] = [];
  try {
    const body = await req.json();
    image = body?.image;
    if (body?.provider) provider = String(body.provider);
    if (body?.model) model = String(body.model);
    if (Array.isArray(body?.known)) known = body.known.filter((x: unknown) => typeof x === "string");
  } catch {
    /* ignore */
  }
  if (!image || typeof image !== "string") {
    return NextResponse.json({ error: "no_image" }, { status: 400 });
  }

  let lastError: unknown;
  for (const m of pickModels(provider, model)) {
    try {
      const { text } = await generateText({
        model: m.model,
        // No retries: a 429 "retry in 56s" would otherwise block the request
        // until the 60s limit. Fail fast and try the next model.
        maxRetries: 0,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: buildPrompt(known) },
              { type: "image", image },
            ],
          },
        ],
      });
      const name = (text || "").trim().replace(/^["'.\s]+|["'.\s]+$/g, "");
      return NextResponse.json({ name, model: m.id });
    } catch (e) {
      console.error(`[identify] ${m.id}`, e);
      lastError = e;
    }
  }
  return NextResponse.json({ error: String(lastError) }, { status: 502 });
}
