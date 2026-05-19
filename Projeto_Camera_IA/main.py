"""
main.py — ponto de entrada do sistema de câmeras com IA.
Uso: python main.py → abre http://localhost:8000
"""
import asyncio
import logging
import sys
import time
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


async def main():
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    import os
    for folder in ["recordings", "logs", "thumbnails"]:
        os.makedirs(folder, exist_ok=True)

    # Banco de dados
    from src.database import init_db
    await init_db(config)
    logger.info("Banco de dados inicializado ✓")

    # Câmeras
    from src.capture.camera import CaptureManager
    capture = CaptureManager(config)
    capture.start_all()
    logger.info(f"Câmeras iniciadas: {capture.camera_ids()} ✓")

    # Gravação contínua
    from src.recording.recorder import RecordingManager
    recorder = RecordingManager(config)
    recorder.set_event_loop(asyncio.get_event_loop())
    recorder.init_continuous(capture.camera_ids())
    logger.info("Gravação contínua iniciada ✓")

    # Assistente IA
    from src.assistant.ai_assistant import AIAssistant
    assistant = AIAssistant(config)
    logger.info("Assistente IA pronto ✓")

    # Dicionário: camera_id → (lista de detecções, timestamp)
    # Guarda DADOS da detecção, não o frame anotado — bboxes são desenhados ao vivo no stream
    detection_data: dict = {}

    # Callback quando clipe de evento fica pronto
    async def on_clip_ready(event, clip_path: str, thumb_path: str):
        import json
        from src.database import get_session, Event

        detected = ", ".join(set(d.label for d in event.detections)) if event.detections else event.event_type
        summary  = f"{detected} detectado em {event.camera_name}"

        async with get_session() as session:
            db_event = Event(
                id          = event.id,
                camera_id   = event.camera_id,
                camera_name = event.camera_name,
                event_type  = event.event_type,
                detected    = json.dumps([
                    {"label": d.label, "confidence": round(d.confidence, 2)}
                    for d in event.detections
                ]),
                confidence  = max((d.confidence for d in event.detections), default=0),
                clip_path   = clip_path,
                thumbnail   = thumb_path,
                ai_summary  = summary,
            )
            session.add(db_event)
            await session.commit()

        logger.info(f"Evento: [{event.camera_name}] {event.event_type} — {summary}")

        from src.api.routes import broadcast_event
        await broadcast_event({
            "id":          event.id,
            "camera_id":   event.camera_id,
            "camera_name": event.camera_name,
            "event_type":  event.event_type,
            "summary":     summary,
            "thumbnail":   thumb_path.replace("\\", "/"),
            "clip":        clip_path.replace("\\", "/"),
            "timestamp":   event.timestamp,
        })

    recorder.on_clip_ready = on_clip_ready

    # Callback de detecção — guarda só os dados da detecção (não o frame)
    def on_detection(event):
        cam = capture.get_camera(event.camera_id)
        if cam:
            recorder.on_event(event, cam)

        # Guarda lista de detecções + timestamp para o stream desenhar ao vivo
        if event.detections:
            detection_data[event.camera_id] = (event.detections, time.time())

    # Pipeline de detecção
    from src.detection.detector import DetectionPipeline
    pipeline = DetectionPipeline(config, on_event=on_detection)
    pipeline.start(capture.queue)
    logger.info("Pipeline de detecção iniciado ✓")

    # Thread que alimenta frames para gravação contínua
    import threading

    def feed_frames():
        while True:
            try:
                frame = capture.queue.get(timeout=1.0)
                recorder.feed_frame(frame.camera_id, frame.image)
            except Exception:
                pass

    threading.Thread(target=feed_frames, daemon=True, name="feed").start()

    # Dashboard web
    import uvicorn
    from src.api.routes import create_app

    app = create_app(config, capture, assistant, detection_data)

    host = config["dashboard"]["host"]
    port = config["dashboard"]["port"]

    logger.info(f"\n{'='*50}")
    logger.info(f"  Dashboard: http://localhost:{port}")
    logger.info(f"{'='*50}\n")

    server_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(server_config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Encerrando sistema…")
        pipeline.stop()
        recorder.stop_all()
        capture.stop_all()


if __name__ == "__main__":
    asyncio.run(main())