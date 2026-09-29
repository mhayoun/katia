#!/usr/bin/env python3
"""
Generate a self-contained offline photo viewer/classifier from a folder that
contains images + index.json (as produced by extract-from-html.py).

The data is embedded directly into viewer.html so it works when opened via
file:// (no fetch of local JSON needed). Images are referenced relatively.

Usage:
  python3 scripts/make-viewer.py <folder>
"""

import datetime as dt
import json
import os
import re
import sys
from collections import Counter

HEB_STOP = {
    "עם", "של", "על", "את", "לא", "כן", "זה", "או", "גם", "יש", "אני", "הוא",
    "היא", "אבל", "כי", "אם", "מה", "מן", "עד", "בין", "אל", "הם", "הן",
}
WORD_RE = re.compile(r"[֐-׿A-Za-z]{2,}")


def parse_date(s):
    if not s:
        return None
    for fmt in ("%b %d, %Y %I:%M:%S %p", "%b %d, %Y %H:%M:%S", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(s.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 scripts/make-viewer.py <folder>")
    folder = sys.argv[1]
    index_path = os.path.join(folder, "index.json")
    if not os.path.isfile(index_path):
        sys.exit(f"index.json introuvable dans {folder}")

    with open(index_path, encoding="utf-8") as f:
        items = json.load(f)

    # Enrich each record with a sortable ISO date.
    for it in items:
        it["ts"] = parse_date(it.get("date"))

    # Build word frequency across descriptions for the filter chips.
    counter = Counter()
    for it in items:
        for w in WORD_RE.findall(it.get("description") or ""):
            if w not in HEB_STOP and not w.isdigit():
                counter[w] += 1
    top_words = [w for w, n in counter.most_common(40) if n >= 2]

    payload = {"items": items, "words": top_words}
    data_json = json.dumps(payload, ensure_ascii=False)

    html = TEMPLATE.replace("/*__DATA__*/", data_json)
    out = os.path.join(folder, "viewer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Visualiseur généré : {out}")
    print(f"   {len(items)} photo(s), {len(top_words)} mot(s)-clé(s) détecté(s)")
    print(f"   Ouvrez-le dans votre navigateur (double-clic).")


TEMPLATE = r"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Katia — Galerie</title>
<style>
  :root{
    --bg:#0f1115;--card:#171a21;--border:#262b36;--text:#e8eaed;--muted:#9aa3b2;
    --accent:#22c55e;--chip:#1f2530;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
  header{position:sticky;top:0;z-index:5;background:rgba(15,17,21,.95);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:14px 16px}
  h1{margin:0 0 10px;font-size:1.25rem}
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  input[type=search],select{background:#0f1319;border:1px solid var(--border);
    color:var(--text);border-radius:10px;padding:9px 12px;font-size:.95rem}
  input[type=search]{flex:1;min-width:200px}
  .count{color:var(--muted);font-size:.9rem;margin-inline-start:auto}
  .chips{display:flex;gap:8px;flex-wrap:wrap;padding:12px 16px 0}
  .chip{background:var(--chip);border:1px solid var(--border);color:var(--text);
    border-radius:999px;padding:5px 12px;font-size:.85rem;cursor:pointer}
  .chip.active{background:var(--accent);color:#06210f;border-color:var(--accent);font-weight:700}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
    gap:14px;padding:16px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;
    overflow:hidden;display:flex;flex-direction:column}
  .card img{width:100%;aspect-ratio:1/1;object-fit:cover;cursor:pointer;display:block;background:#0b0d11}
  .meta{padding:10px 12px}
  .desc{font-size:.92rem;line-height:1.35;margin:0 0 6px}
  .date{font-size:.78rem;color:var(--muted)}
  .empty{padding:40px 16px;text-align:center;color:var(--muted)}
  /* lightbox */
  .lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;
    align-items:center;justify-content:center;flex-direction:column;z-index:20;padding:20px}
  .lb.open{display:flex}
  .lb img{max-width:95vw;max-height:80vh;object-fit:contain;border-radius:8px}
  .lb .cap{color:#fff;margin-top:12px;max-width:800px;text-align:center}
  .lb .cap .date{color:#bbb}
  .lb .close{position:absolute;top:14px;inset-inline-end:18px;color:#fff;
    font-size:30px;cursor:pointer;background:none;border:none}
</style>
</head>
<body>
<header>
  <h1>🐦 Katia — Galerie photos</h1>
  <div class="controls">
    <input id="q" type="search" placeholder="Rechercher dans les descriptions…">
    <select id="sort">
      <option value="date-desc">Date ↓ (récent)</option>
      <option value="date-asc">Date ↑ (ancien)</option>
      <option value="desc">Description A→Z</option>
    </select>
    <span class="count" id="count"></span>
  </div>
</header>
<div class="chips" id="chips"></div>
<div class="grid" id="grid"></div>
<div class="empty" id="empty" style="display:none">Aucune photo ne correspond.</div>

<div class="lb" id="lb">
  <button class="close" id="lbClose">×</button>
  <img id="lbImg" src="" alt="">
  <div class="cap"><div id="lbDesc"></div><div class="date" id="lbDate"></div></div>
</div>

<script type="application/json" id="data">/*__DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const items = DATA.items;
const state = { q:'', sort:'date-desc', word:null };

const grid = document.getElementById('grid');
const empty = document.getElementById('empty');
const count = document.getElementById('count');

// chips
const chips = document.getElementById('chips');
DATA.words.forEach(w => {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = w;
  b.onclick = () => { state.word = (state.word === w ? null : w); syncChips(); render(); };
  b.dataset.w = w;
  chips.appendChild(b);
});
function syncChips(){
  [...chips.children].forEach(c => c.classList.toggle('active', c.dataset.w === state.word));
}

document.getElementById('q').addEventListener('input', e => { state.q = e.target.value.trim(); render(); });
document.getElementById('sort').addEventListener('change', e => { state.sort = e.target.value; render(); });

// lightbox
const lb = document.getElementById('lb');
document.getElementById('lbClose').onclick = () => lb.classList.remove('open');
lb.addEventListener('click', e => { if (e.target === lb) lb.classList.remove('open'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.classList.remove('open'); });
function openLb(it){
  document.getElementById('lbImg').src = it.file;
  document.getElementById('lbDesc').textContent = it.description || '(sans description)';
  document.getElementById('lbDate').textContent = it.date || '';
  lb.classList.add('open');
}

function filtered(){
  let list = items.slice();
  if (state.q){
    const q = state.q.toLowerCase();
    list = list.filter(it => (it.description||'').toLowerCase().includes(q));
  }
  if (state.word){
    list = list.filter(it => (it.description||'').includes(state.word));
  }
  const cmp = {
    'date-desc': (a,b) => (b.ts||'').localeCompare(a.ts||''),
    'date-asc':  (a,b) => (a.ts||'').localeCompare(b.ts||''),
    'desc':      (a,b) => (a.description||'').localeCompare(b.description||'', 'he'),
  }[state.sort];
  list.sort(cmp);
  return list;
}

function render(){
  const list = filtered();
  count.textContent = list.length + ' / ' + items.length + ' photo(s)';
  grid.innerHTML = '';
  empty.style.display = list.length ? 'none' : 'block';
  const frag = document.createDocumentFragment();
  for (const it of list){
    const card = document.createElement('div');
    card.className = 'card';
    const img = document.createElement('img');
    img.src = it.file; img.loading = 'lazy'; img.alt = it.description || '';
    img.onclick = () => openLb(it);
    const meta = document.createElement('div');
    meta.className = 'meta';
    const d = document.createElement('p'); d.className='desc';
    d.textContent = it.description || '(sans description)';
    const dt = document.createElement('div'); dt.className='date'; dt.textContent = it.date || '';
    meta.appendChild(d); meta.appendChild(dt);
    card.appendChild(img); card.appendChild(meta);
    frag.appendChild(card);
  }
  grid.appendChild(frag);
}
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
