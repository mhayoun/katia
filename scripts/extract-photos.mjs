#!/usr/bin/env node
/**
 * Extract photos from a Facebook "Download Your Information" (DYI) export ZIP.
 *
 * The export (JSON format) contains your group posts and their media. It bundles
 * ALL your groups; each post records which group it belongs to. This script can:
 *   - list what's inside (groups found + photo counts),
 *   - extract every photo, or
 *   - extract only the photos attached to posts of ONE target group.
 *
 * No npm dependencies: it shells out to the system `unzip`.
 *
 * Usage:
 *   node scripts/extract-photos.mjs <export.zip> --list
 *   node scripts/extract-photos.mjs <export.zip> --group 310659152306433 --out ./photos
 *   node scripts/extract-photos.mjs <export.zip> --out ./photos      # all photos
 *
 * --group accepts a group id OR a fragment of the group name. Matching is done
 * against the group-posts JSON, so media stored in a shared folder is still
 * filtered correctly.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join, basename } from "node:path";

const IMAGE_RE = /\.(jpe?g|png|gif|webp|heic|bmp|tiff?)$/i;

function parseArgs(argv) {
  const args = { zip: undefined, out: "./photos", group: undefined, list: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--list") args.list = true;
    else if (a === "--out") args.out = argv[++i];
    else if (a === "--group") args.group = argv[++i];
    else if (!a.startsWith("--") && !args.zip) args.zip = a;
  }
  return args;
}

/** List every entry path in the ZIP. */
function listEntries(zip) {
  const out = execFileSync("unzip", ["-Z1", zip], {
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 64,
  });
  return out.split("\n").filter(Boolean);
}

/** Read one entry's bytes from the ZIP without extracting to disk. */
function readEntry(zip, entry) {
  return execFileSync("unzip", ["-p", zip, entry], {
    maxBuffer: 1024 * 1024 * 256,
  });
}

/**
 * Facebook JSON escapes non-ASCII as UTF-8 bytes reinterpreted as latin1
 * ("mojibake"). This repairs strings so group names read correctly.
 */
function fixMojibake(s) {
  try {
    return Buffer.from(s, "latin1").toString("utf8");
  } catch {
    return s;
  }
}

const isImageUri = (v) => typeof v === "string" && IMAGE_RE.test(v);

/**
 * Walk any JSON value, carrying a "context" of the nearest group / date / text
 * seen while descending, and collect one rich record per image found:
 *   { uri, group, date (ISO string|null), description }
 *
 * Facebook's group-posts JSON is roughly:
 *   post: { timestamp, title: "X posted in <group>", data:[{post:"text"}],
 *           attachments:[{data:[{media:{uri, creation_timestamp, description}}]}] }
 * The context propagation makes it tolerant to schema/locale variations.
 */
function collectPostsMedia(node, ctx, acc) {
  if (Array.isArray(node)) {
    for (const item of node) collectPostsMedia(item, ctx, acc);
    return;
  }
  if (!node || typeof node !== "object") return;

  const next = { ...ctx };

  // Group label (kept as-is; the post "title" usually reads "X posted in <group>").
  for (const key of ["title", "group_name", "name"]) {
    if (typeof node[key] === "string" && node[key].length) {
      next.group = fixMojibake(node[key]);
      break;
    }
  }
  // Date (epoch seconds).
  for (const key of ["creation_timestamp", "timestamp"]) {
    if (typeof node[key] === "number" && node[key] > 0) {
      next.date = new Date(node[key] * 1000).toISOString();
      break;
    }
  }
  // Free text: post body or media caption.
  for (const key of ["post", "description", "caption"]) {
    if (typeof node[key] === "string" && node[key].length) {
      next.text = fixMojibake(node[key]);
      break;
    }
  }

  const uri = node.uri || node.media_uri || node.photo;
  if (isImageUri(uri)) {
    acc.push({
      uri,
      group: next.group ?? null,
      date: next.date ?? null,
      description: next.text ?? null,
    });
  }

  for (const key of Object.keys(node)) {
    collectPostsMedia(node[key], next, acc);
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.zip) {
    console.error("Usage: node scripts/extract-photos.mjs <export.zip> [--list] [--group <id|name>] [--out DIR]");
    process.exit(1);
  }
  if (!existsSync(args.zip)) {
    console.error(`ZIP introuvable: ${args.zip}`);
    process.exit(1);
  }

  const entries = listEntries(args.zip);
  const imageEntries = entries.filter((e) => IMAGE_RE.test(e));
  const groupJsonEntries = entries.filter(
    (e) => e.endsWith(".json") && /group/i.test(e),
  );

  // Parse group-posts JSON to map media -> group.
  const media = [];
  for (const je of groupJsonEntries) {
    try {
      const json = JSON.parse(readEntry(args.zip, je).toString("utf8"));
      collectPostsMedia(json, {}, media);
    } catch {
      // skip unparseable json
    }
  }

  // Build a per-group summary from the parsed media.
  const byGroup = new Map();
  for (const m of media) {
    const g = m.group || "(groupe inconnu)";
    byGroup.set(g, (byGroup.get(g) || 0) + 1);
  }

  if (args.list) {
    console.log(`\nEntrées totales : ${entries.length}`);
    console.log(`Images dans le ZIP : ${imageEntries.length}`);
    console.log(`Fichiers JSON de groupes : ${groupJsonEntries.length}`);
    const withDesc = media.filter((m) => m.description).length;
    const withDate = media.filter((m) => m.date).length;
    console.log(
      `Métadonnées : ${withDate}/${media.length} avec date, ${withDesc}/${media.length} avec description`,
    );
    console.log(`\nGroupes détectés (via JSON) :`);
    if (byGroup.size === 0) {
      console.log("  (aucun mapping média→groupe trouvé — voir note ci-dessous)");
    } else {
      for (const [g, n] of [...byGroup.entries()].sort((a, b) => b[1] - a[1])) {
        console.log(`  - ${g} : ${n} photo(s)`);
      }
    }
    console.log(
      `\nAstuce : relancez avec --group "<id ou nom>" --out ./photos pour extraire un seul groupe,\n` +
        `ou sans --group pour extraire toutes les photos.`,
    );
    return;
  }

  // Decide which image entries to extract.
  let toExtract;
  if (args.group) {
    const needle = args.group.toLowerCase();
    // URIs whose post's group matches the needle (id or name fragment).
    const wanted = new Set(
      media
        .filter((m) => (m.group || "").toLowerCase().includes(needle))
        .map((m) => m.uri),
    );
    // Match ZIP entries whose path ends with a wanted URI.
    toExtract = imageEntries.filter((e) =>
      [...wanted].some((u) => e.endsWith(u) || e.includes(u)),
    );
    if (toExtract.length === 0) {
      console.error(
        `Aucune photo trouvée pour le groupe "${args.group}".\n` +
          `Lancez d'abord --list pour voir les groupes disponibles.`,
      );
      process.exit(2);
    }
  } else {
    toExtract = imageEntries;
  }

  mkdirSync(args.out, { recursive: true });
  console.log(`Extraction de ${toExtract.length} photo(s) vers ${args.out} ...`);

  // Extract in batches to stay under arg-length limits.
  const BATCH = 200;
  for (let i = 0; i < toExtract.length; i += BATCH) {
    const batch = toExtract.slice(i, i + BATCH);
    execFileSync("unzip", ["-o", "-j", args.zip, ...batch, "-d", args.out], {
      stdio: "ignore",
      maxBuffer: 1024 * 1024 * 64,
    });
    process.stdout.write(`\r  ${Math.min(i + BATCH, toExtract.length)}/${toExtract.length}`);
  }

  // Write index.json: one metadata record per extracted photo.
  // Keyed by the flattened filename (files are extracted with `unzip -j`).
  const index = toExtract.map((entry) => {
    // Find the record whose uri this entry ends with.
    const rec = media.find((m) => entry.endsWith(m.uri) || entry.includes(m.uri));
    return {
      file: basename(entry),
      group: rec?.group ?? null,
      date: rec?.date ?? null,
      description: rec?.description ?? null,
    };
  });
  writeFileSync(join(args.out, "index.json"), JSON.stringify(index, null, 2));

  console.log(`\n✅ Terminé : ${toExtract.length} photo(s) + index.json dans ${args.out}`);
}

main();
