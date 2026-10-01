"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";
import exifr from "exifr";

interface Meta {
  file?: string;
  album?: string | null;
  group: string | null;
  date: string | null;
  description: string | null;
  species?: string | null;
  context?: string | null;
}

interface Photo {
  id: string; // stable: `${album}/${name}`
  name: string;
  album: string; // Drive folder this photo belongs to
  url: string;
  blob?: Blob;
  driveId?: string; // Drive file id (when loaded from Drive)
  group: string | null;
  date: string | null;
  description: string | null;
  ts: string | null;
}

const HEB_STOP = new Set([
  "עם", "של", "על", "את", "לא", "כן", "זה", "או", "גם", "יש", "אני",
  "הוא", "היא", "אבל", "כי", "אם", "מה", "מן", "עד", "בין", "אל", "הם", "הן",
]);
const IMG_EXT = /\.(jpe?g|png|gif|webp)$/i;

// Image recognition (SigLIP embeddings + k-NN).
const EMB_NAME = "embeddings.json";
const SIGLIP_MODEL = "Xenova/siglip-base-patch16-224";
const KNN_K = 5; // neighbours to vote
const SIM_OK = 0.72; // cosine threshold to accept an image match (else Gemini)

/** Cosine similarity of two L2-normalized vectors = dot product. */
function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

/**
 * Text input with its own local state so typing is never reset by parent
 * re-renders (which happen on every keystroke because filters/sorts recompute).
 * It syncs from the prop only when NOT focused (so AI/chip updates still show).
 */
function EditableInput({
  value,
  onChange,
  placeholder,
  listId,
  className,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  listId: string;
  className: string;
}) {
  const [v, setV] = useState(value);
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setV(value);
  }, [value]);
  const dirty = v !== value;
  return (
    <span className="edit-row">
      <input
        className={className}
        list={listId}
        dir="auto"
        placeholder={placeholder}
        value={v}
        onFocus={() => {
          focused.current = true;
        }}
        onBlur={() => {
          focused.current = false;
          if (v !== value) onChange(v);
        }}
        onChange={(e) => setV(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            onChange(v);
            (e.target as HTMLInputElement).blur();
          }
        }}
      />
      {dirty && (
        <button
          type="button"
          className="ok-btn"
          title="Valider le nom"
          // mousedown (not click) so we commit before the input loses focus
          onMouseDown={(e) => {
            e.preventDefault();
            onChange(v);
          }}
        >
          ✓
        </button>
      )}
    </span>
  );
}

/** Read a Blob as a data URL (for sending an image to the AI route). */
function blobToDataURL(b: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(b);
  });
}

/** Downscale an image (max dimension + JPEG quality) to save Drive space. */
async function downscaleImage(
  file: Blob,
  maxDim = 1600,
  quality = 0.72,
): Promise<Blob> {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const h = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.drawImage(bitmap, 0, 0, w, h);
    const out = await new Promise<Blob | null>((res) =>
      canvas.toBlob((b) => res(b), "image/jpeg", quality),
    );
    return out && out.size < file.size ? out : file;
  } catch {
    return file;
  }
}

// Persist edits locally (per browser), keyed by the photo's filename — so a
// refresh + re-selecting the same ZIP restores the last names/contexts.
const EDITS_KEY = "katia_edits_v1";
function loadStoredEdits(): Record<string, { species?: string; context?: string }> {
  try {
    const raw = localStorage.getItem(EDITS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}
function persistEdit(name: string, species: string, context: string) {
  try {
    const obj = loadStoredEdits();
    obj[name] = { species, context };
    localStorage.setItem(EDITS_KEY, JSON.stringify(obj));
  } catch {
    /* storage unavailable — ignore */
  }
}

// Grouping operations not yet written to Drive (per project), replayed after a
// refresh and committed by « 💾 Save ».
const PENDING_KEY = "katia_pending_v1";
type PendingOp =
  | { type: "name"; ids: string[]; species: string }
  | {
      type: "move";
      items: { id: string; driveId?: string }[];
      targetId: string;
      targetName: string;
    };
function loadPending(project: string): PendingOp[] {
  try {
    const raw = localStorage.getItem(PENDING_KEY);
    const all = raw ? JSON.parse(raw) : {};
    return Array.isArray(all[project]) ? all[project] : [];
  } catch {
    return [];
  }
}
function storePending(project: string, ops: PendingOp[]) {
  try {
    const raw = localStorage.getItem(PENDING_KEY);
    const all = raw ? JSON.parse(raw) : {};
    if (ops.length) all[project] = ops;
    else delete all[project];
    localStorage.setItem(PENDING_KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable — ignore */
  }
}

/**
 * Replay pending grouping ops onto freshly loaded photos (mutates in place):
 * names are re-applied and moved photos get their new album/id.
 */
function replayPending(
  ops: PendingOp[],
  list: Photo[],
  sp: Map<string, string>,
  ctx: Map<string, string>,
) {
  for (const op of ops) {
    if (op.type === "name") {
      const have = new Set(list.map((p) => p.id));
      for (const id of op.ids) if (have.has(id)) sp.set(id, op.species);
      continue;
    }
    for (const it of op.items) {
      const i = list.findIndex((p) => (it.driveId ? p.driveId === it.driveId : p.id === it.id));
      if (i < 0) continue;
      const p = list[i];
      const nid = `${op.targetName}/${p.name}`;
      if (nid === p.id) continue;
      list[i] = { ...p, album: op.targetName, id: nid };
      if (sp.has(p.id)) { sp.set(nid, sp.get(p.id)!); sp.delete(p.id); }
      if (ctx.has(p.id)) { ctx.set(nid, ctx.get(p.id)!); ctx.delete(p.id); }
    }
  }
  // Individual edits made after a move are stored under the new id.
  const stored = loadStoredEdits();
  for (const p of list) {
    const s = stored[p.id];
    if (!s) continue;
    if (typeof s.species === "string") sp.set(p.id, s.species);
    if (typeof s.context === "string") ctx.set(p.id, s.context);
  }
}

// Filler words to skip at the start of a description when guessing the species.
const LEAD_FILLER = new Set([
  ...HEB_STOP,
  "עוד", "פורטרט", "פרוטרט", "פרורטרט", "מבט", "הערצה", "צילום", "תמונה",
  "יפה", "יפהפה", "יקר", "נחמד", "היום", "אתמול", "בוקר", "ערב", "כאן",
  "שם", "הנה", "וואו", "איזה", "עוף", "ציפור",
]);

/**
 * Best-effort guess of the bird name from a free-text Hebrew description.
 * Not perfect on purpose — the client validates/edits it. We skip common
 * filler/opening words, then take the first 1–2 content words.
 */
function guessSpecies(desc: string | null): string {
  if (!desc) return "";
  const words = desc.replace(/[.!?,;:]+/g, " ").split(/\s+/).filter(Boolean);
  const isFiller = (w: string) =>
    LEAD_FILLER.has(w) || LEAD_FILLER.has(w.replace(/^ה/, ""));
  const isStop = (w: string) =>
    /^\d+$/.test(w) || HEB_STOP.has(w) || HEB_STOP.has(w.replace(/^ה/, ""));
  if (words.length <= 2) return words.join(" ").trim();
  // Skip leading filler/stop words, then take up to 2 words, stopping at the
  // first stop word or number (so "סיקסק עם 6 …" → "סיקסק").
  let i = 0;
  while (i < words.length && isFiller(words[i])) i++;
  const name: string[] = [];
  for (let j = i; j < words.length && name.length < 2; j++) {
    if (isStop(words[j])) break;
    name.push(words[j]);
  }
  return (name.length ? name.join(" ") : words.slice(i, i + 2).join(" ")).trim();
}

// Words hinting at a time (season / part of day / etc.) — used to nudge the
// "rest" of a description toward the context field.
const TIME_HINTS = [
  "חורף", "קיץ", "אביב", "סתיו", // seasons
  "בוקר", "ערב", "צהריים", "לילה", "זריחה", "שקיעה", "יום", // day parts
  "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי",
  "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר", // months
];

/**
 * Split a description into a guessed bird name and the "rest" (place / date /
 * season / other). Best-effort — the client edits both fields.
 */
function splitDescription(desc: string | null): { name: string; rest: string } {
  if (!desc) return { name: "", rest: "" };
  const name = guessSpecies(desc);
  let rest = desc;
  if (name) {
    const idx = desc.indexOf(name);
    if (idx >= 0) rest = desc.slice(0, idx) + " " + desc.slice(idx + name.length);
  }
  rest = rest
    .replace(/\s+/g, " ")
    .replace(/^[\s.,;:!?\-–…]+|[\s.,;:!?\-–…]+$/g, "")
    .trim();
  return { name, rest };
}

/** Keep only clean species labels (reject caption-like text). */
function isCleanSpecies(s: string): boolean {
  const t = (s || "").trim();
  if (!t) return false;
  if (/["'“”…?!]/.test(t)) return false; // quotes / ellipsis → caption
  if (t.split(/\s+/).length > 3) return false; // too long → caption
  return true;
}

/** Light heuristic: does this text look like a time (year / season / etc.)? */
function looksLikeTime(text: string): boolean {
  if (!text) return false;
  if (/\b(19|20)\d{2}\b/.test(text)) return true; // a year like 2024
  return TIME_HINTS.some((h) => text.includes(h));
}
const DRIVE = "https://www.googleapis.com/drive/v3/files";
const DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files";
// Everything lives inside ONE project folder (created by the app). Albums are
// subfolders of it; myphotos.json + embeddings.json live inside it too. A small
// manifest at Drive root lists the projects.
const PROJECT_FOLDER = "birds";
const PROJECTS_MANIFEST = "myphotos-projects.json";
const INDEX_NAME = "myphotos.json";

const baseName = (p: string) => p.split(/[\\/]/).pop() || p;
function parseDate(s: string | null): string | null {
  if (!s) return null;
  const d = new Date(s.replace(/\bpm\b/i, "PM").replace(/\bam\b/i, "AM"));
  return isNaN(d.getTime()) ? null : d.toISOString();
}

/** Capture date (EXIF DateTimeOriginal) as ISO; falls back to the file's mtime. */
async function photoDate(file: File): Promise<string> {
  try {
    const x = await exifr.parse(file, ["DateTimeOriginal", "CreateDate"]);
    const d: unknown = x?.DateTimeOriginal || x?.CreateDate;
    if (d instanceof Date && !isNaN(d.getTime())) return d.toISOString();
  } catch {
    /* no EXIF */
  }
  return new Date(file.lastModified).toISOString();
}

const MONTHS_ABBR = [
  "Janv", "Févr", "Mars", "Avr", "Mai", "Juin",
  "Juil", "Août", "Sept", "Oct", "Nov", "Déc",
];
/** Format a date as "JJ Mois AAAA HH:MM" (e.g. "14 Sept 2026 17:54"). */
function formatDate(ts: string | null, raw: string | null): string {
  const d = ts ? new Date(ts) : raw ? new Date(raw) : null;
  if (!d || isNaN(d.getTime())) return raw || "";
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${dd} ${MONTHS_ABBR[d.getMonth()]} ${d.getFullYear()} ${hh}:${mm}`;
}

function parseFacebookHtml(text: string) {
  const doc = new DOMParser().parseFromString(text, "text/html");
  let sections = Array.from(doc.querySelectorAll("section._a6-g"));
  if (sections.length === 0) sections = Array.from(doc.querySelectorAll("section"));
  const meta = new Map<string, Meta>();
  const groups = new Map<string, number>();
  for (const sec of sections) {
    const h2 = sec.querySelector("h2")?.textContent?.trim() || "";
    const gm = h2.match(/posted in (.*)/);
    const group =
      (gm ? gm[1] : "").replace(/[‎‏]/g, "").replace(/\.$/, "").trim() || null;
    const date = sec.querySelector("._a72d")?.textContent?.trim() || null;
    let description: string | null = null;
    const walker = doc.createTreeWalker(sec, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const t = (walker.currentNode.nodeValue || "").trim();
      if (!t || t === h2 || t === date || t.includes("posted in")) continue;
      description = t;
      break;
    }
    const files = new Set<string>();
    sec.querySelectorAll("a[href], img[src]").forEach((el) => {
      const src = el.getAttribute("href") || el.getAttribute("src") || "";
      if (IMG_EXT.test(src)) files.add(baseName(src));
    });
    for (const f of files) {
      meta.set(f, { group, date, description });
      const key = group || "(inconnu)";
      groups.set(key, (groups.get(key) || 0) + 1);
    }
  }
  return { meta, groups };
}

export default function PhotoApp({ accessToken }: { accessToken?: string }) {
  const zipRef = useRef<JSZip | null>(null);
  const metaRef = useRef<Map<string, Meta>>(new Map());
  const urlsRef = useRef<string[]>([]);
  // Bookkeeping so we can re-organize files in Drive after a copy/reload.
  const driveRef = useRef<{
    indexFileId: string | null;
    extraDriveIds: string[]; // duplicate Drive files to clean up
    projectFolderId: string | null;
  }>({ indexFileId: null, extraDriveIds: [], projectFolderId: null });
  // Image-recognition state (SigLIP embeddings + k-NN).
  const extractorRef = useRef<any>(null);
  const embIndexRef = useRef<Map<string, { species: string; vec: Float32Array }>>(new Map());
  const embFileIdRef = useRef<string | null>(null);

  const [zipName, setZipName] = useState<string | null>(null);
  const [groups, setGroups] = useState<Map<string, number>>(new Map());
  const [groupFilter, setGroupFilter] = useState("");
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");
  const [prog, setProg] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedLink, setSavedLink] = useState<string | null>(null);
  const [okMsg, setOkMsg] = useState<string | null>(null);
  const [existingCount, setExistingCount] = useState<number | null>(null);
  const [driveFolders, setDriveFolders] = useState<{ name: string; count: number }[]>([]);
  const [projects, setProjects] = useState<{ name: string; folderId: string }[]>([]);
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [showImportHelp, setShowImportHelp] = useState(false);
  const [driveDown, setDriveDown] = useState(false);
  const [dupCount, setDupCount] = useState(0);
  const [indexCount, setIndexCount] = useState(0);
  const projectKey = currentProjectId || "local";
  const [pending, setPending] = useState<PendingOp[]>([]);
  useEffect(() => setPending(loadPending(projectKey)), [projectKey]);
  function addPending(op: PendingOp) {
    setPending((prev) => {
      const next = [...prev, op];
      storePending(projectKey, next);
      return next;
    });
  }
  function clearPending() {
    storePending(projectKey, []);
    setPending([]);
  }

  const [species, setSpecies] = useState<Map<string, string>>(new Map());
  const [context, setContext] = useState<Map<string, string>>(new Map());
  const [q, setQ] = useState("");
  const [word, setWord] = useState<string | null>(null);
  const [sort, setSort] = useState("date-desc");
  const [nameFilter, setNameFilter] = useState<"all" | "named" | "unnamed">("all");
  const [albumFilter, setAlbumFilter] = useState<string>("(tous)");
  const [viewMode, setViewMode] = useState<"grid" | "byName">("byName");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [groupName, setGroupName] = useState("");
  const [moveOpen, setMoveOpen] = useState(false);
  const [moveNewName, setMoveNewName] = useState("");
  const [moveFolders, setMoveFolders] = useState<
    { id: string; name: string; parent?: string }[]
  >([]);
  const [movePath, setMovePath] = useState<{ id: string; name: string }[]>([]);
  const [identifying, setIdentifying] = useState<Set<string>>(new Set());
  const [aiResults, setAiResults] = useState<Map<string, Record<string, string>>>(
    new Map(),
  );
  const [claudeEnabled, setClaudeEnabled] = useState(false);
  const [aiConf, setAiConf] = useState<Map<string, number>>(new Map());
  const [aiCands, setAiCands] = useState<Map<string, string[]>>(new Map());

  /** Ask an AI model to identify the bird; store its guess under `tag`. */
  async function identifyWith(
    p: Photo,
    tag: string,
    body: { provider: string; model?: string },
  ) {
    const key = `${p.id}:${tag}`;
    setIdentifying((s) => new Set(s).add(key));
    setError(null);
    try {
      const src = p.blob || (await (await fetch(p.url)).blob());
      const small = await downscaleImage(src, 768, 0.8);
      const image = await blobToDataURL(small);
      const res = await fetch("/api/identify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image, known: uniqueSpecies, ...body }),
      });
      const data = await res.json();
      const name = res.ok && data.name && data.name !== "?" ? data.name : "—";
      if (!res.ok) setError(`IA (${tag}) : ${data?.error || res.status}`);
      setAiResults((m) => {
        const n = new Map(m);
        n.set(p.id, { ...(n.get(p.id) || {}), [tag]: name });
        return n;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIdentifying((s) => {
        const n = new Set(s);
        n.delete(key);
        return n;
      });
    }
  }

  /** Auto-classify all unnamed photos (of the current album) with Gemini. */
  async function classifyUnnamedWithAI() {
    const targets = albumScoped.filter((p) => !(species.get(p.id) || "").trim());
    if (targets.length === 0) return;
    if (
      !window.confirm(
        `Analyser ${targets.length} photo(s) non classée(s) avec l'IA (Gemini, gratuit) ` +
          `et remplir leurs noms d'oiseau ?`,
      )
    ) {
      return;
    }
    setError(null);
    setOkMsg(null);
    setBusy("Classement automatique par IA…");
    setProg({ done: 0, total: targets.length });
    let done = 0;
    let ok = 0;
    let cursor = 0;
    // Low concurrency to respect Gemini's free-tier rate limits.
    const worker = async () => {
      while (cursor < targets.length) {
        const p = targets[cursor++];
        try {
          const src = p.blob || (await (await fetch(p.url)).blob());
          const small = await downscaleImage(src, 768, 0.8);
          const image = await blobToDataURL(small);
          const res = await fetch("/api/identify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image,
              provider: "gemini",
              model: "gemini-3.8-flash",
              known: uniqueSpecies,
            }),
          });
          const data = await res.json();
          if (res.ok && data.name && data.name !== "?") {
            setSpeciesFor(p.id, data.name);
            ok++;
          }
        } catch {
          /* skip */
        }
        done++;
        setProg({ done, total: targets.length });
      }
    };
    await Promise.all(Array.from({ length: 2 }, worker));
    setBusy(null);
    setProg(null);
    setOkMsg(
      `✅ IA : ${ok}/${targets.length} photo(s) nommée(s). Vérifiez, puis cliquez « Sauvegarder ».`,
    );
  }

  // ---- Image recognition: SigLIP embeddings + k-NN (+ Gemini fallback) ----

  async function getExtractor() {
    if (extractorRef.current) return extractorRef.current;
    setBusy("Chargement du modèle image (1ʳᵉ fois, ~30-60 s)…");
    const { pipeline, env } = await import("@xenova/transformers");
    (env as any).allowLocalModels = false;
    extractorRef.current = await pipeline("image-feature-extraction", SIGLIP_MODEL);
    return extractorRef.current;
  }

  async function embedUrl(url: string): Promise<Float32Array> {
    const extractor = await getExtractor();
    const out = await extractor(url);
    const v = Float32Array.from(out.data as Float32Array);
    let norm = 0;
    for (const x of v) norm += x * x;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= norm;
    return v;
  }

  async function writeRootJson(
    name: string,
    fileId: string | null,
    json: string,
    parent: string,
  ) {
    if (fileId) {
      await driveFetch(`${DRIVE_UPLOAD}/${fileId}?uploadType=media`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: json,
      });
      return fileId;
    }
    const form = new FormData();
    form.append(
      "metadata",
      new Blob([JSON.stringify({ name, parents: [parent] })], { type: "application/json" }),
    );
    form.append("file", new Blob([json], { type: "application/json" }));
    const r = await driveFetch(`${DRIVE_UPLOAD}?uploadType=multipart&fields=id`, {
      method: "POST",
      body: form,
    });
    return r.ok ? (await r.json()).id : null;
  }

  /** Return the currently-selected project folder, creating a default if none. */
  async function getProjectFolderId(): Promise<string> {
    if (driveRef.current.projectFolderId) return driveRef.current.projectFolderId;
    if (currentProjectId) {
      driveRef.current.projectFolderId = currentProjectId;
      return currentProjectId;
    }
    // No project yet → create the default one.
    const r = await driveFetch(`${DRIVE}?fields=id`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: PROJECT_FOLDER, mimeType: "application/vnd.google-apps.folder" }),
    });
    if (!r.ok) throw new Error("Création du dossier projet échouée.");
    const id = (await r.json()).id as string;
    driveRef.current.projectFolderId = id;
    setCurrentProjectId(id);
    await updateProjectsManifest(PROJECT_FOLDER, id);
    return id;
  }

  /** Read the projects manifest (list of work folders). */
  async function readProjects(): Promise<{ name: string; folderId: string }[]> {
    const files = await listAll(`trashed=false and name='${PROJECTS_MANIFEST}'`, "files(id,name)");
    if (!files[0]) return [];
    const r = await driveFetch(`${DRIVE}/${files[0].id}?alt=media`);
    if (!r.ok) return [];
    const j = await r.json();
    return Array.isArray(j?.projects) ? j.projects : [];
  }

  /** Refresh the "already N photos" + folder list, scoped to a project. */
  async function refreshDriveInfo(pid: string) {
    try {
      const st = await fetchDriveState(pid);
      setExistingCount(st.imageCount);
      const counts = new Map<string, number>();
      for (const im of st.images) {
        const a = im.album || "(racine)";
        counts.set(a, (counts.get(a) || 0) + 1);
      }
      setDriveFolders(
        [...counts.entries()]
          .map(([name, count]) => ({ name, count }))
          .sort((a, b) => a.name.localeCompare(b.name, "he")),
      );
    } catch {
      /* ignore */
    }
  }

  /** Switch the active work folder (resets caches + reloads its info). */
  function selectProject(folderId: string) {
    setCurrentProjectId(folderId);
    driveRef.current.projectFolderId = folderId;
    driveRef.current.indexFileId = null;
    embIndexRef.current = new Map();
    embFileIdRef.current = null;
    setIndexCount(0);
    reset();
    setExistingCount(null);
    setDriveFolders([]);
    refreshDriveInfo(folderId);
  }

  /** Create a new work folder (project) and switch to it. */
  async function createProject() {
    const name = newProjectName.trim();
    if (!name) return;
    setError(null);
    setOkMsg(null);
    setBusy("Création du dossier de travail…");
    try {
      const r = await driveFetch(`${DRIVE}?fields=id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, mimeType: "application/vnd.google-apps.folder" }),
      });
      if (!r.ok) throw new Error("Création du dossier échouée.");
      const id = (await r.json()).id as string;
      await updateProjectsManifest(name, id);
      setProjects(await readProjects());
      setNewProjectName("");
      setBusy(null);
      selectProject(id);
      setOkMsg(`✅ Dossier de travail « ${name} » créé et sélectionné.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(null);
    }
  }

  /** Maintain myphotos-projects.json at Drive root: the list of project folders. */
  async function updateProjectsManifest(name: string, folderId: string) {
    try {
      const files = await listAll(`trashed=false and name='${PROJECTS_MANIFEST}'`, "files(id,name)");
      let mid: string | null = files[0]?.id || null;
      let manifest: { projects: { name: string; folderId: string }[] } = { projects: [] };
      if (mid) {
        const r = await driveFetch(`${DRIVE}/${mid}?alt=media`);
        if (r.ok) {
          const j = await r.json();
          if (j && Array.isArray(j.projects)) manifest = j;
        }
      }
      if (!manifest.projects.some((p) => p.folderId === folderId)) {
        manifest.projects.push({ name, folderId });
        mid = await writeRootJson(PROJECTS_MANIFEST, mid, JSON.stringify(manifest, null, 2), "root");
      }
    } catch {
      /* manifest is best-effort */
    }
  }

  async function saveEmbeddings() {
    const obj: Record<string, { species: string; vec: number[] }> = {};
    for (const [id, e] of embIndexRef.current) {
      obj[id] = { species: e.species, vec: Array.from(e.vec, (x) => Math.round(x * 1e4) / 1e4) };
    }
    const parent = await getProjectFolderId();
    if (!embFileIdRef.current) {
      const files = await listAll(`trashed=false and name='${EMB_NAME}'`, "files(id,name,parents)");
      embFileIdRef.current = files.find((x) => x.parents?.[0] === parent)?.id || null;
    }
    embFileIdRef.current = await writeRootJson(EMB_NAME, embFileIdRef.current, JSON.stringify(obj), parent);
  }

  /** Move existing albums + json files into the project folder ("birds"). */
  async function migrateToProject() {
    if (!accessToken) {
      setError("Non connecté à Google (jeton manquant).");
      return;
    }
    setError(null);
    setOkMsg(null);
    setBusy(`Rangement dans « ${PROJECT_FOLDER} »…`);
    try {
      const projectId = await getProjectFolderId();
      const projectIds = new Set(projects.map((p) => p.folderId));
      projectIds.add(projectId);
      const files = await listAll(
        `trashed=false and (mimeType='application/vnd.google-apps.folder' or name='${INDEX_NAME}' or name='${EMB_NAME}')`,
        "files(id,name,mimeType,parents)",
      );
      // Move only "unfiled" items: not a project folder, not the manifest, not
      // already inside a project folder.
      const movable = files.filter(
        (f) =>
          !projectIds.has(f.id) &&
          f.name !== PROJECTS_MANIFEST &&
          !(f.parents?.[0] && projectIds.has(f.parents[0])),
      );
      setProg({ done: 0, total: movable.length });
      let done = 0;
      let moved = 0;
      for (const f of movable) {
        const qp = new URLSearchParams({ addParents: projectId, fields: "id" });
        if (f.parents?.[0]) qp.set("removeParents", f.parents[0]);
        try {
          const r = await driveFetch(`${DRIVE}/${f.id}?${qp}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          if (r.ok) moved++;
        } catch {
          /* skip */
        }
        done++;
        setProg({ done, total: movable.length });
      }
      const projName = projects.find((p) => p.folderId === projectId)?.name || PROJECT_FOLDER;
      setOkMsg(`✅ ${moved} élément(s) rangé(s) dans le dossier « ${projName} ».`);
      if (currentProjectId) refreshDriveInfo(currentProjectId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  async function loadEmbeddings(): Promise<boolean> {
    if (embIndexRef.current.size) return true;
    const pid = driveRef.current.projectFolderId ?? currentProjectId;
    const files = await listAll(`trashed=false and name='${EMB_NAME}'`, "files(id,name,parents)");
    const f = files.find((x) => !pid || x.parents?.[0] === pid) || null;
    if (!f) return false;
    embFileIdRef.current = f.id;
    const r = await driveFetch(`${DRIVE}/${f.id}?alt=media`);
    if (!r.ok) return false;
    const obj = await r.json();
    for (const id of Object.keys(obj)) {
      const e = obj[id];
      if (e?.vec) embIndexRef.current.set(id, { species: e.species, vec: Float32Array.from(e.vec) });
    }
    setIndexCount(embIndexRef.current.size);
    return embIndexRef.current.size > 0;
  }

  /** Compute + save embeddings for all CLASSIFIED photos (the reference base). */
  async function indexClassified() {
    const refs = photos.filter((p) => isCleanSpecies(species.get(p.id) || ""));
    if (refs.length === 0) {
      setError("Aucune photo classée à indexer (nommez-en d'abord ou rechargez depuis Drive).");
      return;
    }
    setError(null);
    setOkMsg(null);
    try {
      await getExtractor();
      setBusy("Indexation des photos classées…");
      setProg({ done: 0, total: refs.length });
      embIndexRef.current = new Map(); // fresh index (drop stale entries)
      let done = 0;
      for (const p of refs) {
        try {
          const vec = await embedUrl(p.url);
          embIndexRef.current.set(p.id, { species: (species.get(p.id) || "").trim(), vec });
        } catch {
          /* skip */
        }
        done++;
        setProg({ done, total: refs.length });
      }
      setIndexCount(embIndexRef.current.size);
      setBusy(`Sauvegarde de ${EMB_NAME}…`);
      await saveEmbeddings();
      setOkMsg(`✅ ${embIndexRef.current.size} photo(s) de référence indexée(s).`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  /**
   * Recognize unnamed photos by image similarity (k-NN). ALWAYS proposes the
   * nearest match (nothing left empty) and records a confidence %. When the
   * similarity is very low, tries Gemini and keeps its answer if it gives one.
   */
  async function recognizeUnnamed() {
    const targets = albumScoped.filter((p) => !(species.get(p.id) || "").trim());
    if (targets.length === 0) return;
    setError(null);
    setOkMsg(null);
    setBusy("Préparation…");
    try {
      if (!embIndexRef.current.size) {
        const ok = await loadEmbeddings();
        if (!ok) {
          setError("Aucun index trouvé. Cliquez d'abord « 🧠 Indexer classées ».");
          setBusy(null);
          return;
        }
      }
      await getExtractor();
      const refs = [...embIndexRef.current.values()];
      setBusy("Reconnaissance par image…");
      setProg({ done: 0, total: targets.length });
      let done = 0;
      let high = 0;
      let low = 0;
      let byGemini = 0;
      const conf = new Map<string, number>(aiConf);
      const cands = new Map<string, string[]>(aiCands);
      for (const p of targets) {
        try {
          const vec = await embedUrl(p.url);
          const sims = refs
            .map((r) => ({ species: r.species, sim: dot(vec, r.vec) }))
            .sort((a, b) => b.sim - a.sim)
            .slice(0, KNN_K);
          const score = new Map<string, number>();
          for (const s of sims) score.set(s.species, (score.get(s.species) || 0) + s.sim);
          const ranked = [...score.entries()].sort((a, b) => b[1] - a[1]);
          const best = ranked[0];
          const topSim = sims[0]?.sim ?? 0;

          // Low confidence → offer the top-3 distinct candidates to pick from.
          if (topSim < SIM_OK) {
            cands.set(p.id, ranked.slice(0, 3).map(([s]) => s));
          } else {
            cands.delete(p.id);
          }

          let chosen = best ? best[0] : "";
          // Very low similarity → ask Gemini; keep its answer if valid.
          if (topSim < 0.5) {
            try {
              const src = p.blob || (await (await fetch(p.url)).blob());
              const small = await downscaleImage(src, 768, 0.8);
              const image = await blobToDataURL(small);
              const res = await fetch("/api/identify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image, provider: "gemini", model: "gemini-3.8-flash", known: uniqueSpecies }),
              });
              const data = await res.json();
              if (res.ok && data.name && data.name !== "?") {
                chosen = data.name;
                byGemini++;
              }
            } catch {
              /* keep image guess */
            }
          }
          if (chosen) {
            setSpeciesFor(p.id, chosen);
            conf.set(p.id, topSim);
            if (topSim >= SIM_OK) high++;
            else low++;
          }
        } catch {
          /* skip */
        }
        done++;
        setProg({ done, total: targets.length });
      }
      setAiConf(conf);
      setAiCands(cands);
      setOkMsg(
        `✅ Reconnaissance : ${high} sûre(s), ${low} à vérifier, dont ${byGemini} via Gemini. ` +
          `La confiance (%) s'affiche sous chaque photo. Corrigez puis « Sauvegarder ».`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  // Two free Gemini models to compare, plus optional (paid) Claude.
  const AI_MODELS = [
    { tag: "g1", label: "G1", provider: "gemini", model: "gemini-3.8-flash", title: "Gemini 3.8 Flash (gratuit)" },
    { tag: "g2", label: "G2", provider: "gemini", model: "gemini-3.7-flash", title: "Gemini 3.7 Flash (gratuit)" },
  ];
  const [lb, setLb] = useState<Photo | null>(null);

  function toggleSelected(id: string) {
    setSelected((prev) => {
      const s = new Set(prev);
      if (s.has(id)) s.delete(id);
      else s.add(id);
      return s;
    });
  }
  /** Name the selection locally; written to Drive on « 💾 Save ». */
  function applyGroupName() {
    const name = groupName.trim();
    if (!name || selected.size === 0) return;
    const newSp = new Map(species);
    for (const id of selected) {
      newSp.set(id, name);
      persistEdit(id, name, context.get(id) || "");
    }
    addPending({ type: "name", ids: [...selected], species: name });
    setSpecies(newSp);
    setSelected(new Set());
    setGroupName("");
    setOkMsg(`📝 ${selected.size} photo(s) groupée(s) « ${name} » — cliquez « 💾 Save » pour enregistrer dans Drive.`);
  }

  /** Load all app-created folders (for the MoveTo tree browser). */
  async function openMove() {
    setError(null);
    setMoveNewName("");
    setMovePath([]);
    setMoveOpen(true);
    try {
      const fs = await listAll(
        "trashed=false and mimeType='application/vnd.google-apps.folder'",
        "files(id,name,parents)",
      );
      setMoveFolders(fs.map((f: any) => ({ id: f.id, name: f.name, parent: f.parents?.[0] })));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  /** Create a subfolder at the current location in the MoveTo browser. */
  async function createMoveSubfolder() {
    const name = moveNewName.trim().replace(/[\\/]/g, "-");
    if (!name) return;
    const parentId = movePath[movePath.length - 1]?.id;
    setBusy("Création du dossier…");
    try {
      const body: any = { name, mimeType: "application/vnd.google-apps.folder" };
      if (parentId) body.parents = [parentId];
      const r = await driveFetch(`${DRIVE}?fields=id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error("Création échouée.");
      const id = (await r.json()).id as string;
      if (!parentId) await updateProjectsManifest(name, id); // new top-level = project
      setMoveFolders((prev) => [...prev, { id, name, parent: parentId }]);
      setMoveNewName("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  /**
   * Move the selected photos into the given folder (album = its name) — locally
   * only; the Drive moves are queued and done on « 💾 Save ».
   */
  function moveSelectedInto(targetId: string, targetName: string) {
    const ids = new Set(selected);
    const toMove = photos.filter((p) => ids.has(p.id));
    if (toMove.length === 0) return;
    setError(null);
    addPending({
      type: "move",
      items: toMove.map((p) => ({ id: p.id, driveId: p.driveId })),
      targetId,
      targetName,
    });

    // Update local state: change album + id; migrate species/context.
    const idMap = new Map<string, string>();
    const newPhotos = photos.map((p) => {
      if (!ids.has(p.id)) return p;
      const nid = `${targetName}/${p.name}`;
      idMap.set(p.id, nid);
      return { ...p, album: targetName, id: nid };
    });
    const spNew = new Map(species);
    const ctxNew = new Map(context);
    for (const [o, n] of idMap) {
      if (spNew.has(o)) { spNew.set(n, spNew.get(o)!); spNew.delete(o); }
      if (ctxNew.has(o)) { ctxNew.set(n, ctxNew.get(o)!); ctxNew.delete(o); }
    }
    setPhotos(newPhotos);
    setSpecies(spNew);
    setContext(ctxNew);
    setSelected(new Set());
    setMoveOpen(false);
    setMoveNewName("");
    setOkMsg(
      `📝 ${toMove.length} photo(s) groupée(s) vers « ${targetName} » — cliquez « 💾 Save » pour enregistrer dans Drive.`,
    );
  }

  /**
   * Remove selected photos: from the list, and — if they exist in Drive — move
   * them to the Drive trash (recoverable) and refresh myphotos.json.
   */
  async function removeSelected() {
    if (selected.size === 0) return;
    const ids = new Set(selected);
    const toDelete = photos.filter((p) => ids.has(p.id));
    const driveIds = toDelete
      .map((p) => p.driveId)
      .filter((x): x is string => !!x);

    if (
      !window.confirm(
        `Êtes-vous sûr de vouloir effacer ces ${toDelete.length} photo(s) ?` +
          (driveIds.length
            ? `\n${driveIds.length} d'entre elles seront mises à la corbeille de Google Drive (récupérables).`
            : ""),
      )
    ) {
      return;
    }

    setError(null);
    setOkMsg(null);
    try {
      // Trash the ones that live in Drive.
      if (driveIds.length) {
        setBusy("Suppression dans Drive…");
        setProg({ done: 0, total: driveIds.length });
        let done = 0;
        let cursor = 0;
        const worker = async () => {
          while (cursor < driveIds.length) {
            const id = driveIds[cursor++];
            try {
              await driveFetch(`${DRIVE}/${id}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ trashed: true }),
              });
            } catch {
              /* skip */
            }
            done++;
            setProg({ done, total: driveIds.length });
          }
        };
        await Promise.all(Array.from({ length: Math.min(8, driveIds.length) }, worker));
      }

      // Remaining photos + refreshed index.
      const remaining = photos.filter((p) => !ids.has(p.id));
      if (driveRef.current.indexFileId || driveIds.length) {
        try {
          if (!driveRef.current.indexFileId) {
            driveRef.current.indexFileId = (await fetchDriveState()).indexFileId;
          }
          const index = remaining.map((p) => ({
            file: p.name,
            album: p.album,
            group: p.group,
            date: p.date,
            description: p.description,
            species: species.get(p.id) || "",
            context: context.get(p.id) || "",
          }));
          driveRef.current.indexFileId = await writeIndexFile(
            driveRef.current.indexFileId,
            JSON.stringify(index, null, 2),
          );
        } catch {
          /* index refresh best-effort */
        }
      }

      setPhotos(remaining);
      setSpecies((prev) => {
        const m = new Map(prev);
        for (const id of ids) m.delete(id);
        return m;
      });
      setContext((prev) => {
        const m = new Map(prev);
        for (const id of ids) m.delete(id);
        return m;
      });
      setSelected(new Set());
      setExistingCount((prev) =>
        prev !== null ? Math.max(0, prev - driveIds.length) : prev,
      );
      setOkMsg(
        `✅ ${toDelete.length} photo(s) retirée(s)` +
          (driveIds.length ? ` (dont ${driveIds.length} mise(s) à la corbeille Drive).` : "."),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  function setSpeciesFor(id: string, value: string) {
    setSpecies((prev) => {
      const m = new Map(prev);
      m.set(id, value);
      return m;
    });
    persistEdit(id, value, context.get(id) || "");
  }
  /** Move the species filter to the previous/next species in the list. */
  function stepSpecies(delta: number) {
    const names = speciesOptions.map(([s]) => s);
    if (names.length === 0) return;
    const cur = word ? names.indexOf(word) : -1; // -1 = "all"
    let next = cur + delta;
    if (next < -1) next = names.length - 1;
    if (next >= names.length) next = -1;
    setWord(next === -1 ? null : names[next]);
  }

  function setContextFor(id: string, value: string) {
    setContext((prev) => {
      const m = new Map(prev);
      m.set(id, value);
      return m;
    });
    persistEdit(id, species.get(id) || "", value);
  }

  function reset() {
    urlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    urlsRef.current = [];
    setPhotos([]);
    setSavedLink(null);
  }

  async function onZip(file: File) {
    setError(null);
    setBusy("Lecture du ZIP…");
    try {
      const zip = await JSZip.loadAsync(file);
      zipRef.current = zip;
      // Find the group posts HTML anywhere in the archive.
      let htmlPath: string | null = null;
      zip.forEach((path) => {
        if (/group_posts_and_comments\.html$/i.test(path)) htmlPath = path;
      });
      if (!htmlPath) {
        throw new Error(
          "Fichier group_posts_and_comments.html introuvable dans le ZIP.",
        );
      }
      const html = await zip.file(htmlPath)!.async("string");
      const { meta, groups } = parseFacebookHtml(html);
      metaRef.current = meta;
      setGroups(groups);
      setZipName(file.name);
      const top = [...groups.entries()].sort((a, b) => b[1] - a[1])[0];
      setGroupFilter(top ? top[0] : "(tous)");
      reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function showPhotos() {
    const zip = zipRef.current;
    if (!zip) return;
    setError(null);
    setBusy("Extraction des photos…");
    reset();
    try {
      // Map basename -> zip entry for quick lookup.
      const byName = new Map<string, JSZip.JSZipObject>();
      zip.forEach((path, f) => {
        if (!f.dir && IMG_EXT.test(path)) byName.set(baseName(path), f);
      });

      const next: Photo[] = [];
      let done = 0;
      const entries = [...metaRef.current.entries()].filter(
        ([, m]) =>
          groupFilter === "(tous)" || (m.group || "(inconnu)") === groupFilter,
      );
      for (const [name, m] of entries) {
        const f = byName.get(name);
        if (!f) continue;
        const blob = await f.async("blob");
        const url = URL.createObjectURL(blob);
        urlsRef.current.push(url);
        // ZIP photos go into a folder named after their Facebook group.
        const album = `MyPhotos — ${m.group || "photos"}`;
        next.push({
          id: `${album}/${name}`,
          name,
          album,
          url,
          blob,
          group: m.group,
          date: m.date,
          description: m.description,
          ts: parseDate(m.date),
        });
        done++;
        setProgress(`${done}/${entries.length}`);
      }
      // Split each description into a guessed name + the rest (context),
      // then override with any locally-saved edits (survives a refresh).
      const stored = loadStoredEdits();
      const sp = new Map<string, string>();
      const ctx = new Map<string, string>();
      let restored = 0;
      for (const p of next) {
        const { name, rest } = splitDescription(p.description);
        const s = stored[p.id];
        if (s && (typeof s.species === "string" || typeof s.context === "string")) {
          sp.set(p.id, typeof s.species === "string" ? s.species : name);
          ctx.set(p.id, typeof s.context === "string" ? s.context : rest);
          restored++;
        } else {
          sp.set(p.id, name);
          ctx.set(p.id, rest);
        }
      }
      replayPending(loadPending(projectKey), next, sp, ctx);
      setSpecies(sp);
      setContext(ctx);
      setPhotos(next);
      if (restored > 0) {
        setOkMsg(`↩️ ${restored} modification(s) précédente(s) restaurée(s) automatiquement.`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProgress("");
    }
  }

  /**
   * Import a local folder of images: creates an album named after the folder,
   * downscales each image to save space, and appends them to the gallery.
   */
  async function onFolder(fileList: FileList) {
    const files = Array.from(fileList).filter((f) => IMG_EXT.test(f.name));
    if (files.length === 0) {
      setError("Aucune image trouvée dans ce dossier.");
      return;
    }
    const rel = (files[0] as any).webkitRelativePath || "";
    const album = (rel.split("/")[0] || "Album").trim() || "Album";
    setError(null);
    setOkMsg(null);
    setBusy(`Import du dossier « ${album} » (réduction des tailles)…`);
    setProg({ done: 0, total: files.length });
    try {
      // Photos of this folder already in Drive: only their date gets fixed.
      // If the folder already exists in myphotos.json, NOTHING is imported:
      // the folder is only used to fill missing dates.
      let inDrive = new Set<string>();
      let fixOnly = false;
      if (accessToken) {
        const st = await fetchDriveState();
        inDrive = st.existingKeys;
        if (st.indexFileId) {
          const r = await driveFetch(`${DRIVE}/${st.indexFileId}?alt=media`);
          if (!r.ok) throw new Error(`Lecture de ${INDEX_NAME} impossible (${r.status}).`);
          const arr = await r.json().catch(() => []);
          fixOnly = Array.isArray(arr) && arr.some((m: any) => (m?.album || "") === album);
        }
        if (fixOnly) setBusy(`« ${album} » existe déjà — lecture des dates uniquement…`);
      }
      const dateFix = new Map<string, string>(); // file name -> capture date
      const stored = loadStoredEdits();
      const added: Photo[] = [];
      const spAdd = new Map<string, string>();
      const ctxAdd = new Map<string, string>();
      let done = 0;
      for (const file of files) {
        const iso = await photoDate(file); // read from the ORIGINAL (downscale drops EXIF)
        const id = `${album}/${file.name}`;
        if (fixOnly || inDrive.has(id)) {
          if (inDrive.has(id)) dateFix.set(file.name, iso);
          done++;
          setProg({ done, total: files.length });
          continue;
        }
        const blob = await downscaleImage(file);
        const url = URL.createObjectURL(blob);
        urlsRef.current.push(url);
        added.push({
          id,
          name: file.name,
          album,
          url,
          blob,
          group: album,
          date: iso,
          description: null,
          ts: iso,
        });
        const s = stored[id];
        spAdd.set(id, s?.species ?? "");
        ctxAdd.set(id, s?.context ?? "");
        done++;
        setProg({ done, total: files.length });
      }
      setPhotos((prev) => {
        const have = new Set(prev.map((p) => p.id));
        return [...prev, ...added.filter((p) => !have.has(p.id))];
      });
      setSpecies((prev) => {
        const m = new Map(prev);
        for (const [k, v] of spAdd) if (!m.has(k)) m.set(k, v);
        return m;
      });
      setContext((prev) => {
        const m = new Map(prev);
        for (const [k, v] of ctxAdd) if (!m.has(k)) m.set(k, v);
        return m;
      });
      let fixed = 0;
      if (dateFix.size) {
        setBusy(`Correction des dates dans ${INDEX_NAME}…`);
        fixed = await fixDriveDates(album, dateFix);
      }
      if (fixOnly) {
        setOkMsg(
          `✅ « ${album} » existe déjà : aucune photo importée, ` +
            `date manquante ajoutée pour ${fixed} photo(s) sur ${dateFix.size} dans Drive.`,
        );
        return;
      }
      setOkMsg(
        `✅ ${added.length} nouvelle(s) photo(s) importée(s) du dossier « ${album} »` +
          (dateFix.size
            ? `, ${dateFix.size} déjà dans Drive (date manquante ajoutée pour ${fixed}).`
            : "."),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  /**
   * Fill the MISSING date of photos already in Drive (album + file name) in
   * myphotos.json — existing dates and everything else are left untouched.
   * Also updates loaded photos. Returns the number of dates filled.
   */
  async function fixDriveDates(album: string, dates: Map<string, string>): Promise<number> {
    const st = await fetchDriveState();
    let arr: any[] = [];
    if (st.indexFileId) {
      const r = await driveFetch(`${DRIVE}/${st.indexFileId}?alt=media`);
      if (!r.ok) throw new Error(`Lecture de ${INDEX_NAME} impossible (${r.status}).`);
      const j = await r.json().catch(() => []);
      if (Array.isArray(j)) arr = j;
    }
    const seen = new Set<string>();
    const filled = new Map<string, string>();
    for (const m of arr) {
      if ((m?.album || "") !== album || !dates.has(m.file)) continue;
      seen.add(m.file);
      if (!m.date) {
        m.date = dates.get(m.file);
        filled.set(m.file, m.date);
      }
    }
    // Drive photos with no entry yet in myphotos.json: add a minimal one.
    for (const [file, date] of dates) {
      if (!seen.has(file)) {
        arr.push({ file, album, group: album, date, description: null, species: "", context: "" });
        filled.set(file, date);
      }
    }
    if (filled.size === 0) return 0;
    const json = JSON.stringify(arr, null, 2);
    if (st.indexFileId) {
      const w = await driveFetch(`${DRIVE_UPLOAD}/${st.indexFileId}?uploadType=media`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: json,
      });
      if (!w.ok) throw new Error(`Écriture de ${INDEX_NAME} impossible (${w.status}).`);
      driveRef.current.indexFileId = st.indexFileId;
    } else {
      driveRef.current.indexFileId = await writeIndexFile(null, json);
    }
    setPhotos((prev) =>
      prev.map((p) => {
        const d = p.album === album ? filled.get(p.name) : undefined;
        return d ? { ...p, date: d, ts: d } : p;
      }),
    );
    return filled.size;
  }

  async function driveFetch(url: string, init?: RequestInit) {
    const res = await fetch(url, {
      ...init,
      headers: { Authorization: `Bearer ${accessToken}`, ...(init?.headers || {}) },
    });
    if (res.status === 401) throw new Error("Session Google expirée — reconnectez-vous.");
    return res;
  }

  /** List ALL matching Drive files, following pagination (not just the first 1000). */
  async function listAll(q: string, fields: string): Promise<any[]> {
    const out: any[] = [];
    let pageToken: string | undefined;
    do {
      const params = new URLSearchParams({
        q,
        fields: `nextPageToken, ${fields}`,
        pageSize: "1000",
        spaces: "drive",
      });
      if (pageToken) params.set("pageToken", pageToken);
      const res = await driveFetch(`${DRIVE}?${params}`);
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(
          `Lecture Drive impossible (${res.status} : ${b?.error?.message || "erreur inconnue"}). ` +
            `Si le problème persiste, déconnectez-vous puis reconnectez-vous.`,
        );
      }
      const data = await res.json();
      out.push(...(data.files || []));
      pageToken = data.nextPageToken;
    } while (pageToken);
    return out;
  }

  /**
   * Read the current Drive state (app-created files only) WITHOUT downloading
   * image bytes: the album folders, the myphotos.json index, and which
   * album/filename pairs already exist. Supports multiple folders (albums).
   */
  async function fetchDriveState(projectId?: string | null) {
    const pid = projectId ?? driveRef.current.projectFolderId ?? currentProjectId;
    const files = await listAll(
      `trashed=false and (mimeType contains 'image/' or name='${INDEX_NAME}' or mimeType='application/vnd.google-apps.folder')`,
      "files(id,name,mimeType,parents)",
    );
    const folders = files.filter((f) => f.mimeType === "application/vnd.google-apps.folder");
    const imgs = files.filter((f) => (f.mimeType || "").startsWith("image/"));

    // Album folders of THIS project = folders whose parent is the project folder.
    const albumNameById = new Map<string, string>();
    const folderIdByName = new Map<string, string>();
    for (const fo of folders) {
      if (pid && fo.parents?.[0] !== pid) continue;
      albumNameById.set(fo.id, fo.name);
      if (!folderIdByName.has(fo.name)) folderIdByName.set(fo.name, fo.id);
    }

    // Images belonging to this project = those inside one of its album folders.
    const existingKeys = new Set<string>();
    const images = imgs
      .filter((f) => f.parents?.[0] && albumNameById.has(f.parents[0]))
      .map((f) => {
        const album = albumNameById.get(f.parents![0]) || "";
        existingKeys.add(`${album}/${f.name}`);
        return { id: f.id, name: f.name, album };
      });

    const idx = files.find((f) => f.name === INDEX_NAME && (!pid || f.parents?.[0] === pid));
    return {
      folderIdByName,
      existingKeys,
      images,
      indexFileId: idx?.id || null,
      imageCount: images.length,
    };
  }

  /**
   * Create or update myphotos.json inside the project folder. Returns its id.
   * MERGES with the existing file: entries for photos not loaded in the app
   * (other albums, partial load) are kept as long as the photo is still in
   * Drive, so saving never wipes the classification of unloaded photos.
   */
  async function writeIndexFile(
    indexFileId: string | null,
    json: string,
  ): Promise<string | null> {
    if (indexFileId) {
      const r = await driveFetch(`${DRIVE}/${indexFileId}?alt=media`);
      if (!r.ok) throw new Error(`Lecture de ${INDEX_NAME} impossible (${r.status}) — rien n'a été écrit.`);
      const old: any[] = await r.json().catch(() => []);
      const fresh: any[] = JSON.parse(json);
      const key = (m: any) => `${m.album || ""}/${m.file}`;
      const have = new Set(fresh.map(key));
      const inDrive = (await fetchDriveState()).existingKeys;
      const kept = Array.isArray(old)
        ? old.filter((m) => m?.file && !have.has(key(m)) && inDrive.has(key(m)))
        : [];
      json = JSON.stringify([...fresh, ...kept], null, 2);
      await driveFetch(`${DRIVE_UPLOAD}/${indexFileId}?uploadType=media`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: json,
      });
      return indexFileId;
    }
    const parent = await getProjectFolderId();
    return writeRootJson(INDEX_NAME, null, json, parent);
  }

  /** Build the metadata array for myphotos.json from current state. */
  function buildIndex() {
    return photos.map((p) => ({
      file: p.name,
      album: p.album,
      group: p.group,
      date: p.date,
      description: p.description,
      species: species.get(p.id) || "",
      context: context.get(p.id) || "",
    }));
  }

  // Every minute: verify the Google Drive connection is still valid.
  useEffect(() => {
    if (!accessToken) return;
    let stop = false;
    const check = async () => {
      try {
        const res = await fetch(`${DRIVE}?pageSize=1&fields=files(id)`, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
        if (!stop) setDriveDown(!res.ok);
      } catch {
        if (!stop) setDriveDown(true);
      }
    };
    check();
    const t = setInterval(check, 60000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [accessToken]);

  // On sign-in: load the list of work folders (projects), select the first,
  // and show its "already N photos" + folder list.
  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    (async () => {
      try {
        const list = await readProjects();
        if (cancelled) return;
        setProjects(list);
        if (list.length > 0) {
          const first = list[0].folderId;
          setCurrentProjectId(first);
          driveRef.current.projectFolderId = first;
          await refreshDriveInfo(first);
        }
      } catch {
        /* ignore — user can still use the app */
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  /**
   * Upload new photos + write myphotos.json. `parentById` sends a photo into a
   * specific folder (a queued move) instead of its album folder.
   */
  async function saveToDrive(parentById?: Map<string, string>): Promise<boolean> {
    if (!accessToken) {
      setError("Non connecté à Google (jeton manquant).");
      return false;
    }
    if (photos.length === 0) return false;
    const sp = species;
    setError(null);
    setOkMsg(null);
    setBusy("Vérification de votre Drive…");
    try {
      const info = driveRef.current;
      const existing = await fetchDriveState();
      info.indexFileId = existing.indexFileId;
      const folderIdByName = existing.folderIdByName;
      const projectId = await getProjectFolderId(); // album folders go inside it

      // Upload only local photos not already present in their album (dedup album/name).
      const toUpload = photos.filter(
        (p) => p.blob && !p.driveId && !existing.existingKeys.has(`${p.album}/${p.name}`),
      );
      const skipped = photos.length - toUpload.length;

      // Ensure a Drive folder exists for every album we're about to upload to.
      const albums = [...new Set(toUpload.filter((p) => !parentById?.has(p.id)).map((p) => p.album))];
      const toCreate = albums.filter((a) => !folderIdByName.has(a));
      if (toCreate.length) {
        setBusy("Préparation des dossiers…");
        setProg({ done: 0, total: toCreate.length });
        let fdone = 0;
        for (const a of toCreate) {
          const fRes = await driveFetch(`${DRIVE}?fields=id`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: a,
              mimeType: "application/vnd.google-apps.folder",
              parents: [projectId],
            }),
          });
          if (!fRes.ok) {
            const b = await fRes.json().catch(() => ({}));
            throw new Error(
              `Échec création du dossier « ${a} » : ${b?.error?.message || fRes.status}. ` +
                `Vérifiez que l’API Google Drive est activée et que l’accès est autorisé.`,
            );
          }
          folderIdByName.set(a, (await fRes.json()).id);
          fdone++;
          setProg({ done: fdone, total: toCreate.length });
        }
      }

      let done = 0;
      setBusy("Copie vers Drive…");
      setProg({ done: 0, total: toUpload.length });
      let cursor = 0;
      const worker = async () => {
        while (cursor < toUpload.length) {
          const p = toUpload[cursor++];
          try {
            const parent = parentById?.get(p.id) || folderIdByName.get(p.album)!;
            const form = new FormData();
            form.append(
              "metadata",
              new Blob([JSON.stringify({ name: p.name, parents: [parent] })], {
                type: "application/json",
              }),
            );
            form.append("file", p.blob!);
            await driveFetch(`${DRIVE_UPLOAD}?uploadType=multipart&fields=id`, {
              method: "POST",
              body: form,
            });
          } catch {
            /* skip */
          }
          done++;
          setProg({ done, total: toUpload.length });
        }
      };
      await Promise.all(Array.from({ length: Math.min(6, toUpload.length) }, worker));

      // Write the single metadata file (myphotos.json) inside the project.
      setBusy(`Enregistrement de ${INDEX_NAME}…`);
      const index = photos.map((p) => ({
        file: p.name,
        album: p.album,
        group: p.group,
        date: p.date,
        description: p.description,
        species: sp.get(p.id) || "",
        context: context.get(p.id) || "",
      }));
      info.indexFileId = await writeIndexFile(info.indexFileId, JSON.stringify(index, null, 2));

      setSavedLink("https://drive.google.com/drive/my-drive");
      setExistingCount(existing.imageCount + toUpload.length);
      setOkMsg(
        skipped > 0
          ? `✅ Terminé : ${toUpload.length} nouvelle(s) photo(s) dans ${albums.length} dossier(s), ${skipped} déjà présente(s).`
          : `✅ Terminé : ${toUpload.length} photo(s) copiée(s) dans ${albums.length} dossier(s).`,
      );
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  /**
   * « 💾 Save »: perform the queued Drive moves, then upload new photos and
   * write myphotos.json. The queue is cleared only if everything succeeded.
   */
  async function saveAll() {
    if (!accessToken) {
      setError("Non connecté à Google (jeton manquant).");
      return;
    }
    // Final destination per photo (a photo may have been moved several times).
    const targetByDrive = new Map<string, string>();
    const targetById = new Map<string, string>();
    const idNow = new Map<string, string>(); // id at move time -> current id
    for (const op of pending) {
      if (op.type !== "move") continue;
      for (const it of op.items) {
        if (it.driveId) targetByDrive.set(it.driveId, op.targetId);
        else {
          const cur = idNow.get(it.id) ?? it.id;
          const nid = `${op.targetName}/${cur.slice(cur.indexOf("/") + 1)}`;
          idNow.set(it.id, nid);
          targetById.set(nid, op.targetId);
        }
      }
    }
    // Local photos (not yet in Drive) go straight into their chosen folder.
    const parentById = new Map<string, string>();
    for (const p of photos) {
      const t = targetById.get(p.id);
      if (t && !p.driveId) parentById.set(p.id, t);
    }

    setError(null);
    setOkMsg(null);
    let failed = 0;
    const moves = [...targetByDrive.entries()];
    if (moves.length) {
      setBusy("Déplacement des photos dans Drive…");
      setProg({ done: 0, total: moves.length });
      let done = 0;
      for (const [driveId, targetId] of moves) {
        try {
          const r = await driveFetch(`${DRIVE}/${driveId}?fields=parents`);
          if (!r.ok) throw new Error(String(r.status));
          const parents: string[] = (await r.json()).parents || [];
          const others = parents.filter((x) => x !== targetId);
          if (others.length || !parents.includes(targetId)) {
            const qp = new URLSearchParams({ fields: "id" });
            if (!parents.includes(targetId)) qp.set("addParents", targetId);
            if (others.length) qp.set("removeParents", others.join(","));
            const m = await driveFetch(`${DRIVE}/${driveId}?${qp}`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({}),
            });
            if (!m.ok) throw new Error(String(m.status));
          }
        } catch {
          failed++;
        }
        done++;
        setProg({ done, total: moves.length });
      }
    }
    const ok = await saveToDrive(parentById);
    if (currentProjectId) refreshDriveInfo(currentProjectId);
    if (failed) {
      setError(`${failed} déplacement(s) ont échoué — les opérations restent en attente, réessayez « 💾 Save ».`);
    } else if (ok) {
      clearPending();
    }
  }

  async function loadFromDrive(onlyAlbum?: string) {
    if (!accessToken) {
      setError("Non connecté à Google (jeton manquant).");
      return;
    }
    setError(null);
    setOkMsg(null);
    setBusy("Lecture depuis Google Drive…");
    reset();
    try {
      const info = driveRef.current;
      const st = await fetchDriveState();
      info.indexFileId = st.indexFileId;

      // Read all metadata from myphotos.json, keyed by album/filename.
      const metaByKey = new Map<string, Meta>();
      if (st.indexFileId) {
        const r = await driveFetch(`${DRIVE}/${st.indexFileId}?alt=media`);
        if (r.ok) {
          const arr = await r.json();
          if (Array.isArray(arr))
            for (const m of arr)
              if (m?.file) metaByKey.set(`${m.album || ""}/${m.file}`, m);
        }
      }
      // Migration: if myphotos.json is absent/empty, recover from any old
      // per-folder index.json (matched by filename only).
      const metaByNameOnly = new Map<string, Meta>();
      if (metaByKey.size === 0) {
        const oldIdx = await listAll("trashed=false and name='index.json'", "files(id,name)");
        let best: any[] = [];
        for (const ix of oldIdx) {
          try {
            const r = await driveFetch(`${DRIVE}/${ix.id}?alt=media`);
            if (!r.ok) continue;
            const arr = await r.json();
            if (Array.isArray(arr) && arr.length > best.length) best = arr;
          } catch {
            /* skip */
          }
        }
        for (const m of best) if (m?.file) metaByNameOnly.set(m.file, m);
      }

      // Deduplicate by album/filename; collect extra Drive ids for cleanup.
      const seen = new Set<string>();
      const imgs: { id: string; name: string; album: string }[] = [];
      const extraDriveIds: string[] = [];
      const source = onlyAlbum ? st.images.filter((f) => f.album === onlyAlbum) : st.images;
      for (const f of source) {
        const key = `${f.album}/${f.name}`;
        if (seen.has(key)) {
          extraDriveIds.push(f.id);
          continue;
        }
        seen.add(key);
        imgs.push(f);
      }
      driveRef.current.extraDriveIds = extraDriveIds;
      setDupCount(extraDriveIds.length);

      const next: Photo[] = [];
      const sp = new Map<string, string>();
      const ctx = new Map<string, string>();
      let done = 0;
      setProg({ done: 0, total: imgs.length });

      // Download images in parallel (pool of workers) — far faster than 1-by-1.
      let cursor = 0;
      const worker = async () => {
        while (cursor < imgs.length) {
          const f = imgs[cursor++];
          try {
            const r = await driveFetch(`${DRIVE}/${f.id}?alt=media`);
            if (r.ok) {
              const blob = await r.blob();
              const url = URL.createObjectURL(blob);
              urlsRef.current.push(url);
              const m =
                metaByKey.get(`${f.album}/${f.name}`) || metaByNameOnly.get(f.name);
              const split = splitDescription(m?.description ?? null);
              const pid = `${f.album}/${f.name}`;
              next.push({
                id: pid,
                name: f.name,
                album: f.album,
                url,
                driveId: f.id,
                group: m?.group ?? null,
                date: m?.date ?? null,
                description: m?.description ?? null,
                ts: parseDate(m?.date ?? null),
              });
              sp.set(pid, ((m?.species ?? split.name) || "").trim());
              ctx.set(pid, ((m?.context ?? split.rest) || "").trim());
            }
          } catch {
            /* skip this image */
          }
          done++;
          setProg({ done, total: imgs.length });
        }
      };
      await Promise.all(Array.from({ length: Math.min(8, imgs.length) }, worker));

      // Locally-saved edits win over Drive metadata (most recent unsaved work).
      const stored = loadStoredEdits();
      for (const p of next) {
        const s = stored[p.id];
        if (!s) continue;
        if (typeof s.species === "string") sp.set(p.id, s.species);
        if (typeof s.context === "string") ctx.set(p.id, s.context);
      }
      // Re-apply groupings not yet saved to Drive (survive a refresh).
      const ops = loadPending(projectKey);
      replayPending(ops, next, sp, ctx);

      if (next.length === 0) setError("Aucune photo trouvée dans votre Drive.");
      else setOkMsg(`✅ ${next.length} photo(s) chargée(s) depuis votre Drive.`);
      setSpecies(sp);
      setContext(ctx);
      setPhotos(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setProg(null);
    }
  }

  /** Save the current names/contexts to myphotos.json (no file moves needed). */
  async function saveNames() {
    if (!accessToken) {
      setError("Non connecté à Google (jeton manquant).");
      return;
    }
    if (photos.length === 0) return;
    setError(null);
    setOkMsg(null);
    setBusy(`Enregistrement de ${INDEX_NAME}…`);
    try {
      const info = driveRef.current;
      if (!info.indexFileId) {
        // Find the index file (or we'll create it at root).
        const st = await fetchDriveState();
        info.indexFileId = st.indexFileId;
      }
      info.indexFileId = await writeIndexFile(
        info.indexFileId,
        JSON.stringify(buildIndex(), null, 2),
      );
      setOkMsg(`✅ Noms enregistrés dans ${INDEX_NAME}.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  /** Move duplicate Drive files (found at load) to the trash — recoverable. */
  async function cleanDuplicates() {
    if (!accessToken) {
      setError("Non connecté à Google (jeton manquant).");
      return;
    }
    const ids = driveRef.current.extraDriveIds;
    if (ids.length === 0) return;
    setError(null);
    setOkMsg(null);
    setBusy("Suppression des doublons…");
    setProg({ done: 0, total: ids.length });
    let done = 0;
    let deleted = 0;
    let cursor = 0;
    const worker = async () => {
      while (cursor < ids.length) {
        const id = ids[cursor++];
        try {
          const r = await driveFetch(`${DRIVE}/${id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ trashed: true }),
          });
          if (r.ok) deleted++;
        } catch {
          /* skip */
        }
        done++;
        setProg({ done, total: ids.length });
      }
    };
    await Promise.all(Array.from({ length: Math.min(8, ids.length) }, worker));

    driveRef.current.extraDriveIds = [];
    setDupCount(0);
    setExistingCount((prev) => (prev !== null ? Math.max(0, prev - deleted) : prev));
    setBusy(null);
    setProg(null);
    setOkMsg(`✅ ${deleted} doublon(s) mis à la corbeille (récupérables dans Google Drive).`);
  }

  const visible = useMemo(() => {
    let list = photos.slice();
    if (q) {
      const n = q.toLowerCase();
      list = list.filter((p) => (p.description || "").toLowerCase().includes(n));
    }
    if (word) list = list.filter((p) => (species.get(p.id) || "").trim() === word);
    if (albumFilter !== "(tous)") list = list.filter((p) => p.album === albumFilter);
    if (nameFilter === "unnamed")
      list = list.filter((p) => !(species.get(p.id) || "").trim());
    else if (nameFilter === "named")
      list = list.filter((p) => !!(species.get(p.id) || "").trim());
    const cmp: Record<string, (a: Photo, b: Photo) => number> = {
      "date-desc": (a, b) => (b.ts || "").localeCompare(a.ts || ""),
      "date-asc": (a, b) => (a.ts || "").localeCompare(b.ts || ""),
      desc: (a, b) => (a.description || "").localeCompare(b.description || "", "he"),
    };
    list.sort(cmp[sort]);
    return list;
  }, [photos, q, word, sort, nameFilter, albumFilter, species]);

  const albums = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of photos) counts.set(p.album, (counts.get(p.album) || 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0], "he"));
  }, [photos]);

  // Counts scoped to the selected album (or all).
  const albumScoped = useMemo(
    () => (albumFilter === "(tous)" ? photos : photos.filter((p) => p.album === albumFilter)),
    [photos, albumFilter],
  );
  const unnamedCount = useMemo(
    () => albumScoped.filter((p) => !(species.get(p.id) || "").trim()).length,
    [albumScoped, species],
  );
  const namedCount = albumScoped.length - unnamedCount;

  const uniqueSpecies = useMemo(() => {
    const set = new Set<string>();
    for (const v of species.values()) {
      const t = v.trim();
      if (t) set.add(t);
    }
    return [...set].sort((a, b) => a.localeCompare(b, "he"));
  }, [species]);

  // Species dropdown options (name + count), scoped to the selected album.
  const speciesOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of albumScoped) {
      const s = (species.get(p.id) || "").trim();
      if (s) counts.set(s, (counts.get(s) || 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0], "he"));
  }, [albumScoped, species]);

  const uniqueContexts = useMemo(() => {
    const set = new Set<string>();
    for (const v of context.values()) {
      const t = v.trim();
      if (t) set.add(t);
    }
    return [...set].sort((a, b) => a.localeCompare(b, "he"));
  }, [context]);

  // Group the visible photos by species name (for the "by bird" view).
  const grouped = useMemo(() => {
    const map = new Map<string, Photo[]>();
    for (const p of visible) {
      const key = (species.get(p.id) || "").trim() || "— Sans nom —";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(p);
    }
    // Groups sorted by species name, ascending.
    return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0], "he"));
  }, [visible, species]);

  // Split groups into runs: each multi-photo group alone, consecutive
  // single-photo groups together (laid out on 2 columns).
  const groupRuns = useMemo(() => {
    const runs: [string, Photo[]][][] = [];
    for (const g of grouped) {
      const last = runs[runs.length - 1];
      if (g[1].length === 1 && last && last[0][1].length === 1) last.push(g);
      else runs.push([g]);
    }
    return runs;
  }, [grouped]);

  // Reusable card renderer so grid and grouped views stay identical.
  const renderCard = (p: Photo) => (
    <figure key={p.driveId || p.id} className={`photo ${selected.has(p.id) ? "selected" : ""}`}>
      <input
        type="checkbox"
        className="sel-box"
        title="Sélectionner"
        checked={selected.has(p.id)}
        onChange={() => toggleSelected(p.id)}
        onClick={(e) => e.stopPropagation()}
      />
      <div className="ai-icons">
        {AI_MODELS.map((mdl) => (
          <button
            key={mdl.tag}
            className="ai-icon"
            title={mdl.title}
            disabled={identifying.has(`${p.id}:${mdl.tag}`)}
            onClick={() => identifyWith(p, mdl.tag, { provider: mdl.provider, model: mdl.model })}
          >
            {identifying.has(`${p.id}:${mdl.tag}`) ? "…" : mdl.label}
          </button>
        ))}
        {claudeEnabled && (
          <button
            className="ai-icon"
            title="Claude (payant)"
            disabled={identifying.has(`${p.id}:claude`)}
            onClick={() => identifyWith(p, "claude", { provider: "claude" })}
          >
            {identifying.has(`${p.id}:claude`) ? "…" : "C"}
          </button>
        )}
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={p.url} alt={p.description || ""} loading="lazy" onClick={() => setLb(p)} />
      <figcaption className="cap">
        {p.description && (
          <div className="cap-desc" dir="auto" title={p.description}>
            {p.description}
          </div>
        )}
        <EditableInput
          className="species-input"
          listId="species-list"
          placeholder="Nom de l’oiseau…"
          value={species.get(p.id) ?? ""}
          onChange={(val) => setSpeciesFor(p.id, val)}
        />
        {aiConf.has(p.id) && (
          <span
            className={`conf ${(aiConf.get(p.id) || 0) >= SIM_OK ? "ok" : "low"}`}
            title="Confiance de la reconnaissance par image"
          >
            🖼️ {Math.round((aiConf.get(p.id) || 0) * 100)}%
          </span>
        )}
        {aiCands.get(p.id) && aiCands.get(p.id)!.length > 0 && (
          <div className="ai-results">
            {aiCands.get(p.id)!.map((c) => (
              <button
                key={c}
                className="ai-chip"
                title="Proposition (image) — cliquer pour appliquer"
                onClick={() => setSpeciesFor(p.id, c)}
              >
                {c}
              </button>
            ))}
          </div>
        )}
        {aiResults.get(p.id) && Object.keys(aiResults.get(p.id)!).length > 0 && (
          <div className="ai-results">
            {Object.entries(aiResults.get(p.id)!).map(([tag, name]) => (
              <button
                key={tag}
                className="ai-chip"
                title={
                  selected.size > 0
                    ? `Proposition ${tag.toUpperCase()} — remplir « Nom du groupe » (${selected.size} sélectionnée·s)`
                    : `Proposition ${tag.toUpperCase()} — cliquer pour appliquer`
                }
                onClick={() => {
                  if (name === "—") return;
                  if (selected.size > 0) setGroupName(name);
                  else setSpeciesFor(p.id, name);
                }}
              >
                {tag.toUpperCase()}: {name}
              </button>
            ))}
          </div>
        )}
        <EditableInput
          className="species-input ctx-input"
          listId="context-list"
          placeholder="Lieu / date / saison…"
          value={context.get(p.id) ?? ""}
          onChange={(val) => setContextFor(p.id, val)}
        />
        {looksLikeTime(context.get(p.id) || "") && (
          <span className="tag-time">🕒 temps</span>
        )}
        <div className="date">{formatDate(p.ts, p.date)}</div>
      </figcaption>
    </figure>
  );

  return (
    <>
      {driveDown && (
        <div className="notice notice-error busy-sticky" style={{ top: 12, bottom: "auto" }}>
          ⚠️ Connexion à Google Drive perdue.{" "}
          <button
            className="btn btn-accent btn-sm"
            style={{ marginInlineStart: 8 }}
            onClick={() => window.location.reload()}
          >
            Recharger / reconnecter
          </button>
        </div>
      )}

      <div className="card">
        <div className="step">
          <span className="num">1</span> Dossier de travail
        </div>
        <div className="controls">
          {projects.length > 0 && (
            <select
              value={currentProjectId ?? ""}
              onChange={(e) => selectProject(e.target.value)}
              disabled={!!busy}
            >
              {projects.map((p) => (
                <option key={p.folderId} value={p.folderId}>
                  {p.name}
                </option>
              ))}
            </select>
          )}
          <input
            type="text"
            style={{ flex: "0 1 200px", minWidth: 140 }}
            placeholder="Nouveau dossier…"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") createProject();
            }}
          />
          <button
            className="btn btn-accent btn-sm"
            onClick={createProject}
            disabled={!!busy || !newProjectName.trim()}
          >
            + Créer
          </button>
        </div>
        {projects.length === 0 && (
          <p className="hint">
            Aucun dossier de travail. Créez-en un (ex. « birds ») pour y ranger vos
            imports.
          </p>
        )}
      </div>

      {existingCount !== null && existingCount > 0 && photos.length === 0 && (
        <div className="card">
          <div className="notice notice-info" style={{ marginBottom: 0 }}>
            📁 Vous avez déjà <strong>{existingCount} photo(s)</strong> dans votre
            Drive.{" "}
            <button
              className="btn btn-accent btn-sm"
              style={{ marginInlineStart: 8 }}
              onClick={() => {
                setNameFilter("all");
                loadFromDrive();
              }}
              disabled={!!busy}
            >
              Les afficher
            </button>
            <button
              className="btn btn-ghost btn-sm"
              style={{ marginInlineStart: 8 }}
              onClick={() => {
                setNameFilter("unnamed");
                loadFromDrive();
              }}
              disabled={!!busy}
            >
              Afficher que les non classés
            </button>
          </div>

          {driveFolders.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div className="hint" style={{ marginTop: 0, marginBottom: 6 }}>
                Ou chargez un dossier précis :
              </div>
              <div className="chips" dir="rtl">
                {driveFolders.map((f) => (
                  <button
                    key={f.name}
                    className="chip"
                    disabled={!!busy}
                    title={`Charger le dossier « ${f.name} »`}
                    onClick={() => {
                      setNameFilter("all");
                      loadFromDrive(f.name);
                    }}
                  >
                    📁 {f.name} ({f.count})
                  </button>
                ))}
              </div>
            </div>
          )}

        </div>
      )}

      <div className="card">
        <div className="step">
          <span className="num">2</span> Importer des photos
          <button
            className="help-btn"
            title="Aide"
            onClick={() => setShowImportHelp((v) => !v)}
          >
            ?
          </button>
        </div>
        {showImportHelp && (
          <div className="help-box">
            <p style={{ marginTop: 0 }}>
              Depuis un <strong>export Facebook (.zip)</strong> → dossier
              « MyPhotos — &lt;groupe&gt; », ou depuis un <strong>dossier local</strong>{" "}
              → dossier du même nom (images réduites). Tout est lu dans votre
              navigateur.
            </p>
            <p style={{ marginBottom: 4 }}>
              <strong>How to export your photos from a Facebook group</strong>
            </p>
            <ol style={{ margin: 0, paddingInlineStart: 18 }}>
              <li>Connectez-vous sur Facebook.</li>
              <li>
                Go to{" "}
                <a
                  href="https://accountscenter.facebook.com/info_and_permissions/dyi"
                  target="_blank"
                  rel="noreferrer"
                >
                  accountscenter.facebook.com/info_and_permissions/dyi
                </a>{" "}
                and click <strong>Export your information</strong>, then{" "}
                <strong>Create export</strong>.
              </li>
              <li>
                Select your profile and choose <strong>Export to device</strong>.
              </li>
              <li>
                Click <strong>Customize information</strong>, press{" "}
                <strong>Clear all</strong> in every section, then check only{" "}
                <strong>Groups</strong>.
              </li>
              <li>
                Set <strong>Date range: All time</strong>,{" "}
                <strong>Format: HTML</strong> and <strong>Media quality: High</strong>,
                then click <strong>Start export</strong>.
              </li>
              <li>
                Wait for the notification or email that the export is ready, then click{" "}
                <strong>Download</strong> and enter your password. The file stays
                available for 4 days.
              </li>
            </ol>
          </div>
        )}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <input
            type="file"
            accept=".zip,application/zip"
            style={{ display: "none" }}
            id="zip-input"
            onChange={(e) => e.target.files?.[0] && onZip(e.target.files[0])}
          />
          <label htmlFor="zip-input" className="btn btn-ghost">
            {zipName ? `✓ ${zipName}` : "Choisir un fichier ZIP"}
          </label>

          <input
            type="file"
            multiple
            style={{ display: "none" }}
            id="folder-input"
            {...({ webkitdirectory: "", directory: "" } as any)}
            onChange={(e) => e.target.files && onFolder(e.target.files)}
          />
          <label htmlFor="folder-input" className="btn btn-ghost">
            Importer un dossier local
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={claudeEnabled}
              onChange={(e) => setClaudeEnabled(e.target.checked)}
            />
            Activer Claude (payant)
          </label>
        </div>

        {groups.size > 0 && (
          <>
            <label style={{ display: "block", marginTop: 14 }}>Groupe</label>
            <select value={groupFilter} onChange={(e) => setGroupFilter(e.target.value)}>
              <option value="(tous)">(tous les groupes)</option>
              {[...groups.entries()]
                .sort((a, b) => b[1] - a[1])
                .map(([g, n]) => (
                  <option key={g} value={g}>
                    {g} ({n})
                  </option>
                ))}
            </select>
            <button className="btn btn-accent" onClick={showPhotos} disabled={!!busy} style={{ marginTop: 12 }}>
              Afficher les photos
            </button>
          </>
        )}
      </div>

      {photos.length > 0 && (
        <div className="card">
          <div className="step">
            <span className="num">3</span> Sauvegarder dans Google Drive
          </div>
          <p className="hint" style={{ marginTop: 0 }}>
            <strong>« 💾 Sauvegarder »</strong> téléverse les nouvelles photos dans le
            dossier de travail (un sous-dossier par album) <strong>et</strong> enregistre
            les noms/albums dans <strong>{INDEX_NAME}</strong>. À utiliser aussi bien après
            un import qu'après avoir renommé ou groupé des photos.
          </p>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="btn btn-accent" onClick={saveAll} disabled={!!busy}>
              💾 Sauvegarder
            </button>
            <button className="btn btn-ghost" onClick={() => loadFromDrive()} disabled={!!busy}>
              Recharger depuis mon Drive
            </button>
            {dupCount > 0 && (
              <button className="btn btn-danger-outline" onClick={cleanDuplicates} disabled={!!busy}>
                Nettoyer les doublons ({dupCount})
              </button>
            )}
          </div>
          {savedLink && (
            <div className="notice notice-info" style={{ marginTop: 12 }}>
              ✅ Copié dans votre Drive.{" "}
              <a href={savedLink} target="_blank" rel="noreferrer" style={{ color: "inherit" }}>
                Ouvrir le dossier
              </a>
            </div>
          )}
          {okMsg && (
            <div className="notice notice-info" style={{ marginTop: 12 }}>
              {okMsg}
            </div>
          )}
        </div>
      )}

      {busy && (
        <div className="notice notice-info busy-sticky">
          <div className="busy-row">
            <span className="spinner" />
            <span>
              {busy} {prog ? `${prog.done}/${prog.total}` : progress}
            </span>
          </div>
          {prog && prog.total > 0 && (
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${Math.round((prog.done / prog.total) * 100)}%` }}
              />
            </div>
          )}
        </div>
      )}
      {error && <div className="notice notice-error">{error}</div>}

      {photos.length > 0 && (
        <div className="card">
          <div className="step">
            <span className="num">✓</span> Galerie
            <span className="count-pill">
              {visible.length}/{photos.length}
            </span>
          </div>
          <div className="controls">
            <input
              type="search"
              placeholder="Rechercher dans les descriptions…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
            <select value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="date-desc">Date ↓ (récent)</option>
              <option value="date-asc">Date ↑ (ancien)</option>
              <option value="desc">Description A→Z</option>
            </select>
            <select value={viewMode} onChange={(e) => setViewMode(e.target.value as "grid" | "byName")}>
              <option value="byName">Affichage groupe</option>
              <option value="grid">Affichage grille</option>
            </select>
            {albums.length > 1 && (
              <select value={albumFilter} onChange={(e) => setAlbumFilter(e.target.value)}>
                <option value="(tous)">Albums : tous ({photos.length})</option>
                {albums.map(([a, n]) => (
                  <option key={a} value={a}>
                    {a} ({n})
                  </option>
                ))}
              </select>
            )}
          </div>
          <div className="controls">
            <select
              value={nameFilter}
              onChange={(e) => setNameFilter(e.target.value as "all" | "named" | "unnamed")}
            >
              <option value="all">Tous ({albumScoped.length})</option>
              <option value="named">Classés ({namedCount})</option>
              <option value="unnamed">Non Classés ({unnamedCount})</option>
            </select>
            <button
              className="btn btn-ghost btn-sm"
              onClick={indexClassified}
              disabled={!!busy}
              title="Calculer les empreintes image de vos photos classées (référence pour la reconnaissance)"
            >
              🧠 Indexer réf. image{indexCount ? ` (${indexCount})` : ` (${namedCount})`}
            </button>
            {unnamedCount > 0 && (
              <button
                className="btn btn-accent btn-sm"
                onClick={recognizeUnnamed}
                disabled={!!busy}
                title="Reconnaître les non classés par similarité d'image (secours Gemini)"
              >
                🖼️ Reconnaître par image ({unnamedCount})
              </button>
            )}
          </div>

          {pending.length > 0 && (
            <div className="notice notice-info" style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <span>
                📝 {pending.length} opération(s) de classement en attente — pas encore dans Drive.
              </span>
              <button className="btn btn-accent btn-sm" onClick={saveAll} disabled={!!busy}>
                💾 Save
              </button>
            </div>
          )}

          {selected.size > 0 && (
            <div className="batchbar">
              <span className="batch-count">{selected.size} sélectionnée(s)</span>
              <input
                type="text"
                list="species-list"
                dir="auto"
                placeholder="Nom du groupe / de l’oiseau…"
                value={groupName}
                onChange={(e) => setGroupName(e.target.value)}
              />
              <button className="btn btn-accent btn-sm" onClick={applyGroupName}>
                Grouper
              </button>
              <button className="btn btn-accent btn-sm" onClick={openMove}>
                Grouper vers un autre dossier
              </button>
              <button className="btn btn-danger-outline btn-sm" onClick={removeSelected}>
                Effacer
              </button>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setSelected(new Set())}
              >
                Cancel
              </button>
            </div>
          )}
          {speciesOptions.length > 0 && (
            <div className="controls" dir="rtl" style={{ justifyContent: "flex-start" }}>
              <button
                className="btn btn-ghost btn-sm"
                title="Espèce précédente"
                onClick={() => stepSpecies(-1)}
              >
                ◀
              </button>
              <select
                dir="rtl"
                value={word ?? ""}
                onChange={(e) => setWord(e.target.value || null)}
              >
                <option value="">כל המינים ({albumScoped.length})</option>
                {speciesOptions.map(([s, n]) => (
                  <option key={s} value={s}>
                    {s} ({n})
                  </option>
                ))}
              </select>
              <button
                className="btn btn-ghost btn-sm"
                title="Espèce suivante"
                onClick={() => stepSpecies(1)}
              >
                ▶
              </button>
            </div>
          )}

          {viewMode === "grid" ? (
            <div className="gallery">{visible.map(renderCard)}</div>
          ) : (
            groupRuns.map((run) => {
              const sections = run.map(([name, list]) => (
                <section key={name} className="group-section">
                  <h3 className="group-title" dir="auto">
                    {name} <span className="count-pill">{list.length}</span>
                  </h3>
                  <div className="gallery gallery-rows" dir="rtl">{list.map(renderCard)}</div>
                </section>
              ));
              // Consecutive single-photo groups sit side by side (2 columns on web).
              return run.length > 1 || run[0][1].length === 1 ? (
                <div key={run[0][0]} className="singles-run" dir="rtl">{sections}</div>
              ) : (
                sections
              );
            })
          )}

          <datalist id="species-list">
            {uniqueSpecies.map((s) => (
              <option key={s} value={s} />
            ))}
          </datalist>
          <datalist id="context-list">
            {uniqueContexts.map((s) => (
              <option key={s} value={s} />
            ))}
          </datalist>
        </div>
      )}

      {lb && (
        <div className="lb open" onClick={() => setLb(null)}>
          <button className="close" onClick={() => setLb(null)}>×</button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={lb.url} alt={lb.description || ""} />
          <div className="lb-cap" dir="auto">
            <div>{lb.description || "(sans description)"}</div>
            <div className="date">{formatDate(lb.ts, lb.date)}</div>
          </div>
        </div>
      )}

      {moveOpen && (() => {
        const appFolderIds = new Set(moveFolders.map((f) => f.id));
        const currentId = movePath[movePath.length - 1]?.id;
        const shown = moveFolders
          .filter((f) => (currentId ? f.parent === currentId : !f.parent || !appFolderIds.has(f.parent)))
          .sort((a, b) => a.name.localeCompare(b.name, "he"));
        return (
          <div className="lb open" onClick={() => setMoveOpen(false)}>
            <div className="modal" onClick={(e) => e.stopPropagation()} dir="auto">
              <h3 style={{ marginTop: 0 }}>Déplacer {selected.size} photo(s)</h3>

              {/* Breadcrumb */}
              <div className="crumbs">
                <button className="crumb" onClick={() => setMovePath([])}>
                  🏠 Racine
                </button>
                {movePath.map((seg, i) => (
                  <span key={seg.id}>
                    {" / "}
                    <button
                      className="crumb"
                      onClick={() => setMovePath(movePath.slice(0, i + 1))}
                    >
                      {seg.name}
                    </button>
                  </span>
                ))}
              </div>

              {/* Folder list at current level */}
              <div className="folder-list">
                {shown.length === 0 && (
                  <div className="hint">(aucun sous-dossier ici)</div>
                )}
                {shown.map((f) => (
                  <button
                    key={f.id}
                    className="folder-row"
                    onClick={() => setMovePath([...movePath, { id: f.id, name: f.name }])}
                  >
                    📁 {f.name} <span className="chev">›</span>
                  </button>
                ))}
              </div>

              {/* Create subfolder */}
              <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <input
                  type="text"
                  dir="auto"
                  placeholder={currentId ? "Nouveau sous-dossier…" : "Nouveau dossier de travail…"}
                  value={moveNewName}
                  onChange={(e) => setMoveNewName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") createMoveSubfolder();
                  }}
                />
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={createMoveSubfolder}
                  disabled={!!busy || !moveNewName.trim()}
                >
                  + Créer
                </button>
              </div>

              <div style={{ display: "flex", gap: 10, marginTop: 16, justifyContent: "space-between", alignItems: "center" }}>
                <span className="hint" style={{ margin: 0 }}>
                  {currentId ? `Destination : ${movePath[movePath.length - 1].name}` : "Entrez dans un dossier"}
                </span>
                <span style={{ display: "flex", gap: 10 }}>
                  <button className="btn btn-ghost" onClick={() => setMoveOpen(false)}>
                    Annuler
                  </button>
                  <button
                    className="btn btn-accent"
                    disabled={!currentId || !!busy}
                    onClick={() =>
                      moveSelectedInto(
                        movePath[movePath.length - 1].id,
                        movePath[movePath.length - 1].name,
                      )
                    }
                  >
                    Déplacer ici
                  </button>
                </span>
              </div>
            </div>
          </div>
        );
      })()}
    </>
  );
}
