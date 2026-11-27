"""Create a reproducible, method-blinded offline booklet with genuinely blank ratings."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from questionforge.experiment import fingerprint_file
from questionforge.review import GUIDE, ratings_csv

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "greedy", "beam4")


def prepare():
    rng = random.Random(42)
    items, mapping, sources = [], [], {}
    for dataset, experiment in [("public", "benchmark_test"), ("classroom", "novel_test")]:
        path = ROOT / f"results/{experiment}/predictions.jsonl"
        sources[path.relative_to(ROOT).as_posix()] = fingerprint_file(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        # Classroom translations remain in the same sampled topic groups.
        topics = sorted({r["group"] for r in rows})
        classroom_groups = set(rng.sample(topics, 3)) if dataset == "classroom" else None
        for language in ["en", "fr", "ar"]:
            candidates = sorted([r for r in rows if r["language"] == language and r["method"] == "baseline"], key=lambda r: r["id"])
            if classroom_groups is None:
                selected = rng.sample(candidates, 3)
            else:
                selected = [next(r for r in candidates if r["group"] == group) for group in sorted(classroom_groups)]
            by_key = {(r["id"], r["method"]): r for r in rows if r["language"] == language}
            for source in selected:
                for method in METHODS:
                    row = by_key[(source["id"], method)]
                    if row.get("error"):
                        raise ValueError("Do not hide failed generations from a review sample.")
                    opaque_id = hashlib.sha256(f"qf-review-v1:{dataset}:{row['id']}:{method}".encode()).hexdigest()[:12]
                    items.append({"item_id": opaque_id, "language": language, "context": row["context"],
                                  "question": row["question"], "answer": row["answer"]})
                    mapping.append({"item_id": opaque_id, "dataset": dataset, "source_id": row["id"],
                                    "group": row["group"], "language": language, "method": method})
    rng.shuffle(items)
    return items, mapping, sources


BOOKLET = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuestionForge · Independent question review</title>
<style>
body{margin:0;background:#f6f7f3;color:#173c3b;font:17px/1.65 Arial,sans-serif}main{max-width:960px;margin:auto;padding:40px 24px}
h1{font-size:36px;line-height:1.2}small,.muted{color:#60756c}section{background:white;border:1px solid #d8e2d7;border-radius:12px;padding:24px;margin:20px 0}
.passage{white-space:pre-wrap;background:#edf1e8;padding:20px;border-radius:8px}#question{font-size:25px;font-weight:bold}
button{background:#205c4a;color:white;border:0;border-radius:7px;padding:12px 18px;cursor:pointer;font-size:15px}button:disabled{opacity:.45}
input[type=text]{font-size:17px;padding:9px;border:1px solid #afc0b5;border-radius:5px}textarea{width:100%;box-sizing:border-box;min-height:80px;font:inherit}
fieldset{border:1px solid #d8e2d7;border-radius:8px;margin:18px 0}legend{text-transform:capitalize;font-weight:bold}label.score{display:inline-block;padding:10px 22px 10px 0}
.nav{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}progress{width:100%;accent-color:#205c4a}details{margin:20px 0}
</style><main>
<small>QUESTIONFORGE / INDEPENDENT REVIEW</small><h1>Does the question ask a supported fact?</h1>
<p>Judge each question independently from its passage and intended answer. Model names and automatic scores are hidden. A different valid question can still receive full credit.</p>
<details open><summary>Scoring guide · 0 poor, 1 partial, 2 good</summary><div id="guide"></div>
<p>A question with an unsupported premise cannot receive answerability 2 just because its answer appears in the passage. Rate the language used in the passage. Skip languages you cannot judge.</p></details>
<label>Reviewer pseudonym <input id="reviewer" type="text" placeholder="e.g. reviewer-1" maxlength="80"></label>
<p class="muted">Works offline. Drafts stay in this browser. Export your CSV and keep a copy; nothing is sent to a server.</p>
<div class="nav"><strong id="progressText"></strong><button id="export">Export ratings CSV</button></div><progress id="progress"></progress>
<section><div class="nav"><small id="identifier"></small><select id="jump" aria-label="Jump to question"></select></div>
<h3>Source passage</h3><div class="passage" id="context" dir="auto"></div>
<h3>Generated question</h3><div id="question" dir="auto"></div><p><strong>Intended answer:</strong> <span id="answer" dir="auto"></span></p>
<div id="scores"></div><label>Optional comment<textarea id="comment"></textarea></label>
</section><div class="nav"><button id="previous">← Previous</button><span id="saved" role="status"></span><button id="next">Next →</button></div>
<p class="muted">Prepared items are not completed annotations. Only ratings you actually enter are exported. Independent reviewers should work without consulting each other's ratings.</p>
</main><script>
const items=__ITEMS__, guide=__GUIDE__, key='questionforge-review-'+__HASH__;
let state={reviewer_id:'',ratings:{},position:0};try{Object.assign(state,JSON.parse(localStorage.getItem(key)||'{}'))}catch(e){}
let index=Math.max(0,Math.min(items.length-1,state.position||0));
const el=id=>document.getElementById(id), dimensions=['relevance','answerability','fluency'];
el('reviewer').value=state.reviewer_id;
Object.entries(guide).forEach(([name,text])=>{const p=document.createElement('p');p.textContent=name+': '+text;el('guide').appendChild(p)});
items.forEach((item,n)=>{const option=document.createElement('option');option.value=n;option.textContent='Question '+(n+1)+' · '+item.language.toUpperCase();el('jump').appendChild(option)});
function persist(){state.position=index;state.reviewer_id=el('reviewer').value.trim();try{localStorage.setItem(key,JSON.stringify(state));el('saved').textContent='Draft saved in this browser'}catch(e){el('saved').textContent='Browser storage unavailable; export CSV before closing'}
const count=items.filter(item=>dimensions.every(d=>[0,1,2].includes((state.ratings[item.item_id]||{})[d]))).length;
el('progressText').textContent=count+' / '+items.length+' fully rated';el('progress').max=items.length;el('progress').value=count}
function draw(){const item=items[index],rating=state.ratings[item.item_id]||{};el('identifier').textContent='ITEM '+(index+1)+' / '+items.length+' · '+item.language.toUpperCase()+' · '+item.item_id;
['context','question','answer'].forEach(field=>el(field).textContent=item[field]);el('jump').value=index;el('scores').replaceChildren();
dimensions.forEach(d=>{const field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=d;field.appendChild(legend);
[0,1,2].forEach(score=>{const label=document.createElement('label');label.className='score';const radio=document.createElement('input');radio.type='radio';radio.name=d;radio.value=score;radio.checked=rating[d]===score;
radio.onchange=()=>{state.ratings[item.item_id]??={};state.ratings[item.item_id][d]=score;persist()};label.append(radio,document.createTextNode(' '+score));field.appendChild(label)});el('scores').appendChild(field)});
el('comment').value=rating.comment||'';el('previous').disabled=index===0;el('next').disabled=index===items.length-1;persist()}
el('comment').oninput=()=>{state.ratings[items[index].item_id]??={};state.ratings[items[index].item_id].comment=el('comment').value;persist()};el('reviewer').oninput=persist;
el('previous').onclick=()=>{index--;draw()};el('next').onclick=()=>{index++;draw()};el('jump').onchange=()=>{index=Number(el('jump').value);draw()};
el('export').onclick=()=>{persist();if(!state.reviewer_id){alert('Enter a reviewer pseudonym first.');return}
const escape=value=>'"'+String(value??'').replaceAll('"','""')+'"';const header=['reviewer_id','item_id',...dimensions,'comment'];
const rows=items.map(item=>{const r=state.ratings[item.item_id]||{};return [state.reviewer_id,item.item_id,...dimensions.map(d=>r[d]??''),r.comment||''].map(escape).join(',')});
const blob=new Blob(['\uFEFF'+header.join(',')+'\r\n'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'});
const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='questionforge-ratings-'+state.reviewer_id.replace(/[^a-zA-Z0-9_-]/g,'_')+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)};
draw();
</script></html>'''


def main():
    items, mapping, sources = prepare()
    output = ROOT / "results/human_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    item_text = json.dumps(items, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(item_text.encode()).hexdigest()
    manifest = {"sampling_seed": 42, "items": len(items), "source_sha256": sources,
                "items_sha256": digest, "blinding": "Method, model, reference and automatic scores hidden in reviewer booklet.",
                "status": "prepared; no human ratings supplied by this script"}
    for name, value in [("items.json", items), ("researcher_mapping.json", mapping), ("manifest.json", manifest)]:
        target = output / name
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise ValueError("Existing review sample differs. Preserve it before preparing another sample.")
        target.write_text(content, encoding="utf-8")
    (output / "ratings_template.csv").write_text(ratings_csv([{"item_id": item["item_id"]} for item in items]), encoding="utf-8-sig")
    # Escape '<' so source text cannot terminate the script element in a browser.
    html = BOOKLET.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("<", "\\u003c"))
    html = html.replace("__GUIDE__", json.dumps(GUIDE)).replace("__HASH__", json.dumps(digest[:16]))
    (output / "review_booklet.html").write_text(html, encoding="utf-8")
    print(f"Prepared {len(items)} method-blinded items: {output / 'review_booklet.html'}")


if __name__ == "__main__":
    main()
