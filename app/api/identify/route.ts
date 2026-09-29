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
function pickModel(provider: string, model?: string): LanguageModel {
  if (provider === "claude") {
    return (process.env.AI_GATEWAY_MODEL || "anthropic/claude-sonnet-4.5") as LanguageModel;
  }
  const safe = model && /^gemini-[a-z0-9.\-]+$/i.test(model) ? model : "gemini-3.8-flash";
  return google(safe);
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

  try {
    const { text } = await generateText({
      model: pickModel(provider, model),
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
    return NextResponse.json({ name });
  } catch (e) {
    console.error("[identify]", e);
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
