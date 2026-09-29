#!/usr/bin/env python3
"""
Extract photos + metadata from a Facebook "Download Your Information" export in
HTML format (group_posts_and_comments.html).

For each post it captures the group, date, description and the referenced image
files, filters to one target group, copies the images into an output folder and
writes an index.json (one record per photo: file, group, date, description).

Usage:
  python3 scripts/extract-from-html.py <html_file> [--root DIR] [--group SUBSTR]
                                       [--out DIR] [--list]

  --root   export root the image paths are relative to (default: 3 levels up
           from the html file, i.e. the export root)
  --group  keep only posts whose group contains this text (default: keep all)
  --out    output folder (default: ./katia-photos)
  --list   only print the groups found and photo counts, copy nothing
"""

import argparse
import html
import json
import os
import re
import shutil
import sys

SECTION_RE = re.compile(r'<section class="_a6-g".*?</section>', re.S)
H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
DATE_RE = re.compile(r'_a72d">(.*?)<', re.S)
IMG_RE = re.compile(
    r'(?:href|src)="(your_facebook_activity/[^"]+?\.(?:jpg|jpeg|png|gif|webp))"',
    re.I,
)
TEXT_RE = re.compile(r">([^<]+)<")
LRM = "‎‏"  # RTL/LTR marks Facebook injects around Hebrew


def clean(s: str) -> str:
    return html.unescape(s).strip().strip(LRM).strip()


def group_from_h2(h2_html: str):
    text = clean(re.sub(r"<[^>]+>", "", h2_html))
    m = re.search(r"posted in (.*)", text)
    grp = m.group(1) if m else text
    return clean(grp).rstrip(".").strip(LRM).strip()


def parse_sections(html_text: str):
    posts = []
    for sec in SECTION_RE.findall(html_text):
        h2 = H2_RE.search(sec)
        group = group_from_h2(h2.group(1)) if h2 else None
        h2_text = clean(re.sub(r"<[^>]+>", "", h2.group(1))) if h2 else ""

        date_m = DATE_RE.search(sec)
        date = clean(date_m.group(1)) if date_m else None

        images = []
        for img in IMG_RE.findall(sec):
            if img not in images:
                images.append(img)

        # Description = the first text node that is neither the header nor the date.
        description = None
        for piece in TEXT_RE.findall(sec):
            t = clean(piece)
            if not t or t == h2_text or t == date or "posted in" in t:
                continue
            description = t
            break

        posts.append(
            {"group": group, "date": date, "description": description, "images": images}
        )
    return posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html_file")
    ap.add_argument("--root")
    ap.add_argument("--group", default="")
    ap.add_argument("--out", default="./katia-photos")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.html_file):
        sys.exit(f"HTML introuvable: {args.html_file}")

    # Export root: image paths in the HTML are relative to it.
    root = args.root or os.path.abspath(
        os.path.join(os.path.dirname(args.html_file), "..", "..")
    )

    with open(args.html_file, encoding="utf-8") as f:
        posts = parse_sections(f.read())

    # Group summary.
    by_group = {}
    for p in posts:
        g = p["group"] or "(inconnu)"
        by_group[g] = by_group.get(g, 0) + len(p["images"])

    if args.list:
        print(f"Posts: {len(posts)} | racine export: {root}")
        print("Groupes détectés (nb photos) :")
        for g, n in sorted(by_group.items(), key=lambda kv: -kv[1]):
            print(f"  - {g} : {n}")
        return

    needle = args.group
    kept = [p for p in posts if (needle in (p["group"] or ""))] if needle else posts

    os.makedirs(args.out, exist_ok=True)
    index = []
    copied = 0
    missing = 0
    for p in kept:
        for rel in p["images"]:
            src = os.path.join(root, rel)
            name = os.path.basename(rel)
            if not os.path.isfile(src):
                missing += 1
                continue
            shutil.copy2(src, os.path.join(args.out, name))
            copied += 1
            index.append(
                {
                    "file": name,
                    "group": p["group"],
                    "date": p["date"],
                    "description": p["description"],
                }
            )

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"✅ {copied} photo(s) copiée(s) dans {args.out} (+ index.json)")
    if missing:
        print(f"⚠️  {missing} image(s) référencée(s) mais introuvable(s) sur le disque")


if __name__ == "__main__":
    main()
