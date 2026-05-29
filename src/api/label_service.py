"""
Active Learning Labeling Service — FastAPI + Web UI
Browse uncertain samples, one-click label, export to training data.

Usage:
    python -m src.api.label_service
    Open http://localhost:8700
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.active_learning.queue import ActiveLearningQueue

app = FastAPI(title="Electrolyte Label Service")
queue = ActiveLearningQueue(queue_dir="./al_queue")
RAW_DATA_DIR = r"C:\Users\32333\raw_data_hierarchical"


@app.get("/")
async def index():
    return HTMLResponse(LABEL_UI_HTML)


@app.get("/api/stats")
async def stats():
    pending = queue.get_pending()
    labeled = queue.get_labeled()
    # Show prediction stats
    ng_count = sum(1 for s in pending if s.probability >= 0.5)
    ok_count = len(pending) - ng_count
    return {
        "pending": len(pending),
        "labeled": len(labeled),
        "uncertainty_threshold": queue.uncertainty_threshold,
        "predicted_ok": ok_count,
        "predicted_ng": ng_count,
        "ready_to_export": len(labeled),
    }


@app.get("/api/samples/pending")
async def pending_samples(page: int = Query(1, ge=1), per_page: int = Query(20, le=100)):
    samples = queue.get_pending()
    start = (page - 1) * per_page
    page_samples = samples[start:start + per_page]
    return {
        "total": len(samples),
        "page": page,
        "samples": [
            {
                "index": queue._samples.index(s),
                "image_url": f"/api/image/{queue._samples.index(s)}",
                "model_name": s.model_name,
                "predicted_label": "NG" if s.probability >= 0.5 else "OK",
                "probability": s.probability,
                "uncertainty": s.uncertainty,
                "timestamp": s.timestamp,
            }
            for s in page_samples
        ]
    }


@app.get("/api/samples/labeled")
async def labeled_samples():
    return [
        {
            "index": queue._samples.index(s),
            "label": s.label,
            "predicted": "NG" if s.probability >= 0.5 else "OK",
            "probability": s.probability,
            "labeled_at": s.labeled_at,
        }
        for s in queue.get_labeled()
    ]


@app.get("/api/image/{index}")
async def get_image(index: int):
    if 0 <= index < len(queue._samples):
        path = queue._samples[index].image_path
        if os.path.exists(path):
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "Image not found")


@app.post("/api/label/{index}")
async def label_sample(index: int, label: str = Query(..., regex="^(hierarchical|non_hierarchical)$")):
    if index < 0 or index >= len(queue._samples):
        raise HTTPException(404, "Sample not found")
    queue.label(index, label)
    return {"status": "ok", "index": index, "label": label}


@app.post("/api/batch-label")
async def batch_label(data: dict):
    """Batch label: {"labels": {"0": "hierarchical", "5": "non_hierarchical", ...}}"""
    count = 0
    for idx_str, label in data.get("labels", {}).items():
        idx = int(idx_str)
        if 0 <= idx < len(queue._samples):
            queue.label(idx, label)
            count += 1
    return {"status": "ok", "labeled": count}


@app.post("/api/export")
async def export_labeled():
    labeled_count = queue.labeled_count
    if labeled_count == 0:
        return {"status": "error", "message": "No labeled samples to export"}

    queue.export_labeled(RAW_DATA_DIR)
    return {
        "status": "ok",
        "exported": labeled_count,
        "target_dir": RAW_DATA_DIR,
    }


# ── Web UI ──
LABEL_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Electrolyte Label Tool</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:#1e293b;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155}
.header h1{font-size:18px}.header .stats{font-size:13px;color:#94a3b8}
.main{display:grid;grid-template-columns:1fr 300px;gap:16px;padding:20px;max-width:1400px;margin:0 auto}
.image-area{background:#1e293b;border-radius:10px;padding:20px;display:flex;flex-direction:column;align-items:center}
.image-area img{max-width:100%;max-height:65vh;border-radius:6px;object-fit:contain}
.image-info{margin-top:12px;text-align:center;font-size:13px;color:#94a3b8}
.actions{display:flex;gap:12px;margin-top:16px}
.btn{padding:12px 40px;border:none;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;transition:all .15s}
.btn:hover{transform:translateY(-1px)}
.btn-ok{background:#16a34a;color:#fff}.btn-ok:hover{background:#15803d}
.btn-ng{background:#dc2626;color:#fff}.btn-ng:hover{background:#b91c1c}
.btn-skip{background:#334155;color:#94a3b8}.btn-skip:hover{background:#475569}
.btn-export{background:#6366f1;color:#fff;padding:8px 20px;font-size:13px}
.sidebar{background:#1e293b;border-radius:10px;padding:16px}
.sidebar h3{font-size:14px;margin-bottom:12px;color:#94a3b8}
.progress{background:#334155;border-radius:4px;height:6px;margin:8px 0;overflow:hidden}
.progress-fill{background:#6366f1;height:100%;transition:width .2s}
.stat-item{display:flex;justify-content:space-between;padding:4px 0;font-size:12px}
.queue-list{max-height:400px;overflow-y:auto;font-size:11px}
.queue-item{padding:6px 8px;border-bottom:1px solid #1e293b;cursor:pointer;display:flex;justify-content:space-between}
.queue-item:hover{background:#1e293b}
.queue-item.active{background:#1e3a5f;border-left:2px solid #3b82f6}
.badge{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600}
.badge-ok{background:#16a34a33;color:#22c55e}
.badge-ng{background:#dc262633;color:#ef4444}
.keyboard-hint{font-size:11px;color:#64748b;margin-top:8px;text-align:center}
</style>
</head>
<body>
<div class="header">
  <h1>Electrolyte Label Tool</h1>
  <div class="stats">
    <span id="statPending">Pending: 0</span> |
    <span id="statLabeled">Labeled: 0</span> |
    <button class="btn-export" onclick="exportData()">Export to Training</button>
  </div>
</div>
<div class="main">
  <div class="image-area">
    <img id="mainImage" src="" alt="Select a sample">
    <div class="image-info" id="imageInfo">No sample selected</div>
    <div class="actions">
      <button class="btn btn-ok" onclick="label('non_hierarchical')" id="btnOk">OK — Non-Hierarchical (A)</button>
      <button class="btn btn-skip" onclick="nextSample()">Skip (S)</button>
      <button class="btn btn-ng" onclick="label('hierarchical')" id="btnNg">NG — Hierarchical (D)</button>
    </div>
    <div class="keyboard-hint">A = OK | D = NG | S = Skip | E = Export</div>
  </div>
  <div class="sidebar">
    <h3>Queue Progress</h3>
    <div class="progress"><div class="progress-fill" id="progressBar" style="width:0%"></div></div>
    <div class="stat-item"><span>Pending</span><span id="pendingCount">0</span></div>
    <div class="stat-item"><span>Labeled</span><span id="labeledCount">0</span></div>
    <div class="stat-item"><span>Predicted OK</span><span id="predOk">0</span></div>
    <div class="stat-item"><span>Predicted NG</span><span id="predNg">0</span></div>
    <hr style="border-color:#334155;margin:10px 0">
    <div class="queue-list" id="queueList"></div>
  </div>
</div>
<script>
let samples=[],currentIdx=-1,currentSampleIdx=-1;

async function loadStats(){const r=await fetch('/api/stats');const d=await r.json();
document.getElementById('statPending').textContent='Pending: '+d.pending;
document.getElementById('statLabeled').textContent='Labeled: '+d.labeled;
document.getElementById('pendingCount').textContent=d.pending;
document.getElementById('labeledCount').textContent=d.labeled;
document.getElementById('predOk').textContent=d.predicted_ok;
document.getElementById('predNg').textContent=d.predicted_ng;
const total=d.pending+d.labeled;
document.getElementById('progressBar').style.width=total?Math.round(d.labeled/total*100)+'%':'0%';
}

async function loadSamples(){const r=await fetch('/api/samples/pending?per_page=100');const d=await r.json();samples=d.samples;renderQueue();if(samples.length&&currentIdx<0)selectSample(0);}

function renderQueue(){const el=document.getElementById('queueList');
el.innerHTML=samples.map((s,i)=>`<div class="queue-item${i===currentIdx?' active':''}" onclick="selectSample(${i})">
<span><span class="badge ${s.predicted_label==='OK'?'badge-ok':'badge-ng'}">${s.predicted_label}</span> #${s.index}</span>
<span style="color:#64748b">${(s.uncertainty*100).toFixed(0)}% unsure</span></div>`).join('');}

function selectSample(i){currentIdx=i;const s=samples[i];currentSampleIdx=s.index;
document.getElementById('mainImage').src='/api/image/'+s.index+'?t='+Date.now();
document.getElementById('imageInfo').innerHTML=`#${s.index} | Model: ${s.model_name} | Predicted: <b style="color:${s.predicted_label==='OK'?'#22c55e':'#ef4444'}">${s.predicted_label}</b> | Probability: ${(s.probability*100).toFixed(1)}% | Uncertainty: ${(s.uncertainty*100).toFixed(0)}%`;
renderQueue();}

async function label(lbl){if(currentSampleIdx<0)return;await fetch('/api/label/'+currentSampleIdx+'?label='+lbl,{method:'POST'});
samples.splice(currentIdx,1);currentIdx=-1;if(samples.length)selectSample(Math.min(currentIdx+1,samples.length-1));loadStats();renderQueue();}

function nextSample(){if(samples.length){selectSample((currentIdx+1)%samples.length)}}

async function exportData(){if(!confirm('Export all labeled samples to training data?'))return;const r=await fetch('/api/export',{method:'POST'});const d=await r.json();alert(d.status==='ok'?`Exported ${d.exported} samples to ${d.target_dir}`:d.message);loadStats();}

document.addEventListener('keydown',e=>{
if(e.target.tagName==='INPUT')return;
if(e.key==='a'||e.key==='A')label('non_hierarchical');
if(e.key==='d'||e.key==='D')label('hierarchical');
if(e.key==='s'||e.key==='S')nextSample();
if(e.key==='e'||e.key==='E')exportData();
});

loadStats();loadSamples();
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Label Service starting at http://localhost:8700")
    print(f"Queue dir: {queue.queue_dir.absolute()}")
    print(f"Export target: {RAW_DATA_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8700, log_level="info")
