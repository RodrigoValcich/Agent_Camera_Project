"""
API FastAPI: endpoints REST + WebSocket para o dashboard.

Teclas:
  [0]          → grade 2x2
  [1][2][3][4] → câmera individual
  [I]          → ativa/desativa caixas de detecção na tela
  [ESC]        → voltar para grade / encerrar sistema
"""
import json
import asyncio
import logging
import os
import time
from typing import Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, desc

logger = logging.getLogger(__name__)

_ws_clients: list[WebSocket] = []

BBOX_COLORS = {
    "person":  (0,   255,  0),
    "vehicle": (255, 165,  0),
    "animal":  (0,   200, 255),
    "motion":  (0,     0, 255),
    "object":  (200, 200,   0),
}


async def broadcast_event(data: dict):
    if not _ws_clients:
        return
    msg = json.dumps(data)
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def _draw_detections(img, detections) -> None:
    for d in detections:
        if not hasattr(d, 'bbox') or d.bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = d.bbox
        color = BBOX_COLORS.get(getattr(d, 'category', ''), (200, 200, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.label} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def create_app(config: dict, capture, assistant, detection_data: dict = None) -> FastAPI:
    if detection_data is None:
        detection_data = {}

    bbox_ttl = 4.0
    state    = {"show_bbox": True}

    app = FastAPI(title="Camera AI Dashboard", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return _DASHBOARD_HTML

    @app.get("/stream/{camera_id}")
    async def stream(camera_id: str):
        cam = capture.get_camera(camera_id)
        if not cam:
            raise HTTPException(404, f"Câmera {camera_id} não encontrada")

        quality = config["dashboard"].get("stream_quality", 70)

        async def frame_generator():
            while True:
                img = cam.get_latest_frame()
                if img is None:
                    await asyncio.sleep(0.05)
                    continue

                if state["show_bbox"]:
                    entry = detection_data.get(camera_id)
                    if entry is not None:
                        detections, last_det = entry
                        if (time.time() - last_det) < bbox_ttl:
                            img = img.copy()
                            _draw_detections(img, detections)

                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buf.tobytes() +
                    b"\r\n"
                )
                await asyncio.sleep(1 / 25)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace;boundary=frame",
        )

    @app.post("/api/bbox-overlay/{bbox_state}")
    async def bbox_overlay(bbox_state: str):
        state["show_bbox"] = (bbox_state == "on")
        return {"show_bbox": state["show_bbox"]}

    @app.get("/api/cameras")
    async def cameras():
        return capture.all_status()

    @app.get("/api/events")
    async def events(limit: int = 50, camera_id: Optional[str] = None, event_type: Optional[str] = None):
        from src.database import get_session, Event

        async with get_session() as session:
            q = select(Event).order_by(desc(Event.created_at)).limit(limit)
            if camera_id:
                q = q.where(Event.camera_id == camera_id)
            if event_type:
                q = q.where(Event.event_type == event_type)
            result = await session.execute(q)
            rows   = result.scalars().all()

        return [
            {
                "id":          r.id,
                "camera_id":   r.camera_id,
                "camera_name": r.camera_name,
                "event_type":  r.event_type,
                "ai_summary":  r.ai_summary,
                "thumbnail":   r.thumbnail,
                "clip_path":   r.clip_path,
                "created_at":  r.created_at.timestamp() if r.created_at else None,
            }
            for r in rows
        ]

    @app.get("/api/thumbnail/{event_id}")
    async def thumbnail(event_id: str):
        from src.database import get_session, Event

        async with get_session() as session:
            result = await session.execute(select(Event).where(Event.id == event_id))
            event  = result.scalar_one_or_none()

        if not event or not event.thumbnail or not os.path.exists(event.thumbnail):
            raise HTTPException(404, "Thumbnail não encontrada")

        return FileResponse(event.thumbnail, media_type="image/jpeg")

    @app.post("/api/chat")
    async def chat(body: dict):
        message  = body.get("message", "")
        event_id = body.get("event_id")

        frames = []
        for cam_id in capture.camera_ids():
            cam = capture.get_camera(cam_id)
            if cam:
                img = cam.get_latest_frame()
                if img is not None:
                    frames.append((cam_id, img))

        event_context = None
        if event_id:
            from src.database import get_session, Event
            async with get_session() as session:
                result = await session.execute(select(Event).where(Event.id == event_id))
                ev = result.scalar_one_or_none()
            if ev:
                event_context = {
                    "event_type":  ev.event_type,
                    "camera_name": ev.camera_name,
                    "time":        ev.created_at.strftime("%d/%m %H:%M") if ev.created_at else "",
                    "ai_summary":  ev.ai_summary,
                }

        cam_names = [capture.get_camera(c[0]).name for c in frames if capture.get_camera(c[0])]
        full_message = f"[Câmeras: {', '.join(cam_names)}] " + message

        reply = await assistant.chat(
            full_message,
            frames=[f[1] for f in frames] if frames else None,
            event_context=event_context,
        )
        return {"reply": reply}

    @app.post("/api/shutdown")
    async def shutdown():
        logger.info("Encerramento solicitado pelo dashboard")
        asyncio.get_event_loop().call_later(0.5, lambda: os._exit(0))
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        _ws_clients.append(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            if ws in _ws_clients:
                _ws_clients.remove(ws)

    return app


# --------------------------------------------------------------------------- #

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Camera AI Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0f1117; color:#e0e0e0; font-family:'Segoe UI',sans-serif; height:100vh; display:flex; flex-direction:column; }

  header { background:#1a1d27; padding:10px 20px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #2a2d3a; flex-shrink:0; gap:12px; }
  header h1 { font-size:1.1rem; font-weight:600; color:#fff; flex-shrink:0; }

  /* Botões de câmera no header */
  #cam-buttons { display:flex; gap:6px; align-items:center; flex:1; }
  .cam-btn { display:flex; align-items:center; gap:5px; cursor:pointer; padding:4px 10px; border-radius:6px; font-size:0.75rem; background:#1e2130; border:1px solid #2a2d3a; color:#888; transition:all .2s; }
  .cam-btn:hover { background:#2a2d3a; color:#fff; }
  .cam-btn.active { background:#1d3a6b; border-color:#3b82f6; color:#60a5fa; }
  .cam-btn .dot { width:7px; height:7px; border-radius:50%; background:#444; }
  .cam-btn .dot.online { background:#22c55e; box-shadow:0 0 5px #22c55e88; }
  /* Botão mosaico */
  .cam-btn.mosaic { color:#94a3b8; }
  .cam-btn.mosaic.active { background:#1e2d1e; border-color:#22c55e; color:#22c55e; }

  #right-bar { display:flex; gap:8px; align-items:center; flex-shrink:0; }
  #bbox-indicator { font-size:0.7rem; padding:3px 10px; border-radius:12px; font-weight:500; transition:all .3s; cursor:pointer; }
  #bbox-indicator.on  { background:#22c55e22; color:#22c55e; border:1px solid #22c55e55; }
  #bbox-indicator.off { background:#ef444422; color:#ef4444; border:1px solid #ef444455; }

  .layout { display:flex; flex:1; overflow:hidden; min-height:0; }

  .cameras { flex:1; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:4px; padding:4px; background:#0a0c12; }
  .cameras.solo-mode { grid-template-columns:1fr; grid-template-rows:1fr; }
  .cameras.solo-mode .cam-cell { display:none; }
  .cameras.solo-mode .cam-cell.solo { display:block; }

  .cam-cell { position:relative; background:#111; border-radius:6px; overflow:hidden; cursor:pointer; border:2px solid transparent; transition:border-color .2s; }
  .cam-cell:hover { border-color:#3b82f655; }
  .cam-cell.solo { border-color:#3b82f6; }
  .cam-cell.detecting { border-color:#ef4444; animation:detect-pulse 1s ease; }
  @keyframes detect-pulse { 0%,100%{box-shadow:0 0 8px #ef444444} 50%{box-shadow:0 0 20px #ef4444aa} }

  .cam-cell img { width:100%; height:100%; object-fit:cover; display:block; background:#0a0c12; }
  .cam-label  { position:absolute; top:8px; left:8px; background:#000c; color:#fff; font-size:0.72rem; padding:2px 8px; border-radius:4px; }
  .cam-number { position:absolute; bottom:8px; left:8px; background:#000c; color:#666; font-size:0.65rem; padding:1px 6px; border-radius:4px; }
  .cam-badge  { position:absolute; top:8px; right:8px; font-size:0.65rem; padding:2px 7px; border-radius:4px; font-weight:600; display:none; }
  .cam-badge.show { display:block; }
  .badge-person  { background:#22c55e22; color:#22c55e; border:1px solid #22c55e55; }
  .badge-vehicle { background:#f9731622; color:#f97316; border:1px solid #f9731655; }
  .badge-animal  { background:#06b6d422; color:#06b6d4; border:1px solid #06b6d455; }

  #key-hint { position:fixed; bottom:10px; left:14px; color:#22c55e; font-size:0.68rem; pointer-events:none; text-shadow:0 0 8px #22c55e66; letter-spacing:.3px; transition:color .3s; }

  .sidebar { width:340px; display:flex; flex-direction:column; border-left:1px solid #1e2130; background:#13151f; flex-shrink:0; }
  .tabs { display:flex; border-bottom:1px solid #1e2130; flex-shrink:0; }
  .tab { flex:1; padding:10px; text-align:center; cursor:pointer; font-size:0.8rem; color:#666; border-bottom:2px solid transparent; transition:.2s; }
  .tab.active { color:#3b82f6; border-color:#3b82f6; }

  #events-panel { display:flex; flex-direction:column; flex:1; overflow:hidden; }
  #events-panel.hidden { display:none; }
  #events-list { flex:1; overflow-y:auto; padding:8px; display:flex; flex-direction:column; gap:6px; }
  #events-empty { color:#555; padding:20px; font-size:.8rem; text-align:center; }

  .event-card { background:#1a1d27; border-radius:8px; padding:10px; cursor:pointer; border:1px solid #2a2d3a; transition:border-color .2s; }
  .event-card:hover { border-color:#3b82f670; }
  .event-card.active { border-color:#3b82f6; }
  .event-card.new { animation:pulse .7s ease; }
  @keyframes pulse { 0%,100%{background:#1a1d27} 50%{background:#1e2540} }
  .event-header { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
  .event-icon { font-size:1rem; }
  .event-meta { flex:1; min-width:0; }
  .event-cam  { font-size:0.75rem; font-weight:600; color:#fff; }
  .event-time { font-size:0.65rem; color:#555; }
  .event-summary { font-size:0.72rem; color:#9ca3af; line-height:1.4; }
  .event-thumb { width:100%; border-radius:5px; margin-top:6px; max-height:90px; object-fit:cover; }

  #chat-panel { display:none; flex-direction:column; flex:1; overflow:hidden; }
  #chat-panel.active { display:flex; }
  #chat-messages { flex:1; overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:92%; padding:8px 12px; border-radius:10px; font-size:0.8rem; line-height:1.5; word-break:break-word; }
  .msg.user      { background:#1d4ed8; align-self:flex-end; color:#fff; }
  .msg.assistant { background:#1e2130; align-self:flex-start; color:#d1d5db; }

  .chat-input-row { padding:10px; border-top:1px solid #1e2130; display:flex; gap:6px; flex-shrink:0; }
  .chat-input-row input { flex:1; background:#1a1d27; border:1px solid #2a2d3a; border-radius:8px; padding:8px 12px; color:#e0e0e0; font-size:0.82rem; outline:none; min-width:0; }
  .chat-input-row input:focus { border-color:#3b82f6; }
  .chat-input-row button { background:#2563eb; color:#fff; border:none; border-radius:8px; padding:8px 14px; cursor:pointer; font-size:0.8rem; }
  .chat-input-row button:hover { background:#1d4ed8; }

  #shutdown-modal { display:none; position:fixed; inset:0; background:#000b; z-index:999; align-items:center; justify-content:center; }
  #shutdown-modal.show { display:flex; }
  .modal-box { background:#1a1d27; border:1px solid #2a2d3a; border-radius:14px; padding:30px 40px; text-align:center; }
  .modal-box h2 { font-size:1.1rem; margin-bottom:8px; color:#fff; }
  .modal-box p  { font-size:0.8rem; color:#888; margin-bottom:22px; }
  .modal-btns   { display:flex; gap:10px; justify-content:center; }
  .modal-btns button { padding:9px 22px; border-radius:8px; border:none; cursor:pointer; font-size:0.85rem; font-weight:500; }
  .btn-cancel  { background:#2a2d3a; color:#ccc; }
  .btn-cancel:hover  { background:#33374a; }
  .btn-confirm { background:#dc2626; color:#fff; }
  .btn-confirm:hover { background:#b91c1c; }

  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-track { background:#0f1117; }
  ::-webkit-scrollbar-thumb { background:#2a2d3a; border-radius:2px; }
</style>
</head>
<body>

<header>
  <h1>📷 Camera AI</h1>
  <div id="cam-buttons">
    <!-- Botão mosaico -->
    <div class="cam-btn mosaic active" id="btn-mosaic" onclick="switchMode(0)" title="Tecla 0">
      ⊞ Todas
    </div>
    <!-- Botões das câmeras gerados pelo JS -->
  </div>
  <div id="right-bar">
    <span id="bbox-indicator" class="on" onclick="toggleBbox()" title="Tecla I">🔲 Caixas ON</span>
  </div>
</header>

<div class="layout">
  <div class="cameras" id="cameras-grid"></div>
  <div class="sidebar">
    <div class="tabs">
      <div class="tab active" onclick="showTab('events')">Eventos</div>
      <div class="tab" onclick="showTab('chat')">Assistente IA</div>
    </div>

    <div id="events-panel">
      <div id="events-list">
        <p id="events-empty">Aguardando eventos…</p>
      </div>
    </div>

    <div id="chat-panel">
      <div id="chat-messages"></div>
      <div class="chat-input-row">
        <input id="chat-input" placeholder="Pergunte sobre as câmeras…" onkeydown="if(event.key==='Enter')sendChat()">
        <button onclick="sendChat()">Enviar</button>
      </div>
    </div>
  </div>
</div>

<div id="key-hint">[0] Mosaico · [1][2][3][4] Câmera · [I] Caixas · [ESC] Encerrar</div>

<div id="shutdown-modal">
  <div class="modal-box">
    <h2>⚠️ Encerrar sistema?</h2>
    <p>O servidor será desligado e as câmeras pararão de gravar.</p>
    <div class="modal-btns">
      <button class="btn-cancel" onclick="closeShutdown()">Cancelar</button>
      <button class="btn-confirm" onclick="confirmShutdown()">Encerrar</button>
    </div>
  </div>
</div>

<script>
const SHOW_BADGE = new Set(['person', 'vehicle', 'animal', 'object']);
const ICONS = { person:'🧍', vehicle:'🚗', animal:'🐾', object:'📦', motion:'👁️' };

let cameras         = [];
let selectedEventId = null;
let soloMode        = false;
let currentMode     = 0;
let showBbox        = true;
const badgeTimers   = {};
const detectTimers  = {};

// ------------------------------------------------------------------ //
// Grid + botões de câmera no header

async function loadCameras() {
  const res = await fetch('/api/cameras');
  cameras = await res.json();
  buildGrid();
  buildCamButtons();
}

function buildGrid() {
  const grid = document.getElementById('cameras-grid');
  grid.innerHTML = '';

  cameras.forEach((cam, i) => {
    const cell = document.createElement('div');
    cell.className = 'cam-cell';
    cell.id = 'cell-' + cam.id;

    const img = document.createElement('img');
    img.alt = cam.name;
    img.src = '/stream/' + cam.id;
    img.onerror = () => {
      setTimeout(() => { img.src = '/stream/' + cam.id + '?t=' + Date.now(); }, 2000);
    };
    cell.appendChild(img);

    cell.innerHTML += `
      <span class="cam-label">${cam.name}</span>
      <span class="cam-number">[${i+1}]</span>
      <span class="cam-badge" id="badge-${cam.id}"></span>
    `;
    cell.ondblclick = () => switchMode(i + 1);
    grid.appendChild(cell);
  });
}

function buildCamButtons() {
  const bar = document.getElementById('cam-buttons');
  // Mantém o botão mosaico, remove câmeras antigas
  bar.innerHTML = '';

  // Botão mosaico
  const mosaic = document.createElement('div');
  mosaic.className = 'cam-btn mosaic' + (currentMode === 0 ? ' active' : '');
  mosaic.id = 'btn-mosaic';
  mosaic.title = 'Tecla 0';
  mosaic.innerHTML = '⊞ Todas';
  mosaic.onclick = () => switchMode(0);
  bar.appendChild(mosaic);

  // Botões individuais por câmera
  cameras.forEach((cam, i) => {
    const btn = document.createElement('div');
    btn.className = 'cam-btn' + (currentMode === i + 1 ? ' active' : '');
    btn.id = 'btn-cam-' + cam.id;
    btn.title = `Tecla ${i+1}`;
    btn.innerHTML = `<div class="dot ${cam.online ? 'online' : ''}"></div>${cam.name}`;
    btn.onclick = () => switchMode(i + 1);
    bar.appendChild(btn);
  });
}

function updateCamButtons() {
  // Atualiza dots de online/offline sem rebuildar tudo
  cameras.forEach((cam, i) => {
    const btn = document.getElementById('btn-cam-' + cam.id);
    if (!btn) return;
    const dot = btn.querySelector('.dot');
    if (dot) dot.className = 'dot' + (cam.online ? ' online' : '');
    btn.className = 'cam-btn' + (currentMode === i + 1 ? ' active' : '');
  });
  const mosaic = document.getElementById('btn-mosaic');
  if (mosaic) mosaic.className = 'cam-btn mosaic' + (currentMode === 0 ? ' active' : '');
}

// ------------------------------------------------------------------ //
// Modo de exibição

function switchMode(mode) {
  currentMode = mode;
  const grid  = document.getElementById('cameras-grid');
  document.querySelectorAll('.cam-cell').forEach(c => c.classList.remove('solo'));

  if (mode === 0) {
    soloMode = false;
    grid.classList.remove('solo-mode');
    setHint('[0] Mosaico · [1][2][3][4] Câmera · [I] Caixas · [ESC] Encerrar');
  } else {
    soloMode = true;
    grid.classList.add('solo-mode');
    const cam = cameras[mode - 1];
    if (cam) {
      document.getElementById('cell-' + cam.id)?.classList.add('solo');
      setHint(`[${mode}] ${cam.name} — [0] Mosaico · ESC volta · [I] Caixas`);
    }
  }
  updateCamButtons();
}

// ------------------------------------------------------------------ //
// Toggle bboxes (tecla I)

function toggleBbox() {
  showBbox = !showBbox;
  fetch('/api/bbox-overlay/' + (showBbox ? 'on' : 'off'), {method:'POST'});
  const ind = document.getElementById('bbox-indicator');
  ind.textContent = showBbox ? '🔲 Caixas ON' : '🔲 Caixas OFF';
  ind.className   = showBbox ? 'on' : 'off';
  const hint = document.getElementById('key-hint');
  hint.style.color = showBbox ? '#22c55e' : '#ef4444';
  hint.textContent  = showBbox ? 'Caixas ATIVADAS' : 'Caixas DESATIVADAS';
  setTimeout(() => {
    hint.style.color = '#22c55e';
    hint.textContent = '[0] Mosaico · [1][2][3][4] Câmera · [I] Caixas · [ESC] Encerrar';
  }, 2000);
}

function setHint(text) {
  document.getElementById('key-hint').style.color = '#22c55e';
  document.getElementById('key-hint').textContent = text;
}

// ------------------------------------------------------------------ //
// Detecção — pisca borda; badge só para person/vehicle/animal

function flashDetection(cameraId, eventType) {
  const cell  = document.getElementById('cell-' + cameraId);
  const badge = document.getElementById('badge-' + cameraId);
  if (!cell) return;

  cell.classList.add('detecting');
  clearTimeout(detectTimers[cameraId]);
  detectTimers[cameraId] = setTimeout(() => cell.classList.remove('detecting'), 3000);

  if (!badge || !SHOW_BADGE.has(eventType)) return;
  badge.textContent = eventType;
  badge.className   = `cam-badge show badge-${eventType}`;
  clearTimeout(badgeTimers[cameraId]);
  badgeTimers[cameraId] = setTimeout(() => {
    badge.textContent = ''; badge.className = 'cam-badge';
  }, 5000);
}

// ------------------------------------------------------------------ //
// Teclas

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toLowerCase();
  if (key === '0') { switchMode(0); return; }
  if (['1','2','3','4'].includes(key)) { switchMode(parseInt(key)); return; }
  if (key === 'i') { toggleBbox(); return; }
  if (key === 'escape') {
    if (document.getElementById('shutdown-modal').classList.contains('show')) {
      closeShutdown();
    } else if (soloMode) {
      switchMode(0);
    } else {
      openShutdown();
    }
  }
});

// ------------------------------------------------------------------ //
// Encerrar

function openShutdown()  { document.getElementById('shutdown-modal').classList.add('show'); }
function closeShutdown() { document.getElementById('shutdown-modal').classList.remove('show'); }
async function confirmShutdown() {
  try { await fetch('/api/shutdown', {method:'POST'}); } catch(e) {}
  document.body.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;gap:16px;color:#555"><div style="font-size:3rem">⏻</div><div>Sistema encerrado.</div></div>`;
}

// ------------------------------------------------------------------ //
// Eventos

function fmtTime(ts) {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleTimeString('pt-BR', {hour:'2-digit', minute:'2-digit'});
}

async function loadEvents() {
  try {
    const res    = await fetch('/api/events?limit=50');
    const events = await res.json();
    const list   = document.getElementById('events-list');

    if (!events.length) {
      list.innerHTML = '<p id="events-empty" style="color:#555;padding:20px;font-size:.8rem;text-align:center">Nenhum evento registrado ainda.</p>';
      return;
    }

    list.innerHTML = '';
    events.forEach(ev => list.appendChild(makeEventCard(ev)));
  } catch(e) {
    console.error('Erro ao carregar eventos:', e);
  }
}

function makeEventCard(ev, isNew=false) {
  const card = document.createElement('div');
  card.className = 'event-card' + (isNew ? ' new' : '');
  card.id = 'ev-' + ev.id;

  const showBadge = SHOW_BADGE.has(ev.event_type);

  card.innerHTML = `
    <div class="event-header">
      <span class="event-icon">${ICONS[ev.event_type]||'📌'}</span>
      <div class="event-meta">
        <div class="event-cam">${ev.camera_name||ev.camera_id}</div>
        <div class="event-time">${fmtTime(ev.created_at)}</div>
      </div>
      ${showBadge ? `<span class="cam-badge show badge-${ev.event_type}">${ev.event_type}</span>` : ''}
    </div>
    <div class="event-summary">${ev.ai_summary||''}</div>
    ${ev.thumbnail ? `<img class="event-thumb" src="/api/thumbnail/${ev.id}" alt="thumb" onerror="this.style.display='none'">` : ''}
  `;

  card.onclick = () => {
    selectedEventId = ev.id;
    document.querySelectorAll('.event-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    const idx = cameras.findIndex(c => c.id === ev.camera_id);
    if (idx >= 0) switchMode(idx + 1);
  };
  return card;
}

// ------------------------------------------------------------------ //
// WebSocket

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = e => {
    const ev   = JSON.parse(e.data);
    const list = document.getElementById('events-list');
    // Remove placeholder se existir
    const empty = document.getElementById('events-empty');
    if (empty) empty.remove();

    list.prepend(makeEventCard(ev, true));
    flashDetection(ev.camera_id, ev.event_type);

    fetch('/api/cameras').then(r=>r.json()).then(c => {
      cameras = c;
      updateCamButtons();
    });
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ------------------------------------------------------------------ //
// Chat

function showTab(tab) {
  document.querySelectorAll('.tab').forEach((t,i) =>
    t.classList.toggle('active', (i===0?'events':'chat')===tab)
  );
  document.getElementById('events-panel').classList.toggle('hidden', tab==='chat');
  document.getElementById('chat-panel').classList.toggle('active', tab==='chat');
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = '';
  addMsg('user', msg);

  const body = {message: msg};
  if (selectedEventId) body.event_id = selectedEventId;

  const thinking = addMsg('assistant', '…');
  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    thinking.textContent = data.reply;
  } catch(e) {
    thinking.textContent = 'Erro ao conectar com o assistente.';
  }
}

function addMsg(role, text) {
  const box = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className   = 'msg ' + role;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

// ------------------------------------------------------------------ //
loadCameras();
loadEvents();
connectWS();
setInterval(() => {
  fetch('/api/cameras').then(r=>r.json()).then(c => { cameras=c; updateCamButtons(); });
  // Recarrega eventos a cada 30s para pegar novos que vieram enquanto WS estava fora
  loadEvents();
}, 30000);
</script>
</body>
</html>
"""