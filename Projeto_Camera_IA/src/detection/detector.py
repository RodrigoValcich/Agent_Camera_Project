"""
Pipeline de detecção: YOLOv8 (objetos) + diferença de frames (movimento).

Detecta automaticamente se há GPU NVIDIA disponível e escolhe o melhor modelo:
  - GPU NVIDIA  → yolov8s.pt  (small, mais preciso)
  - Só CPU      → yolov8n.pt  (nano, mais rápido)

Roda numa thread dedicada consumindo frames da fila do CaptureManager.
"""
import cv2
import time
import uuid
import threading
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics não instalado — rode: pip install ultralytics")

logger = logging.getLogger(__name__)

# Mapeamento de classes COCO para categorias do sistema
PERSON_CLASSES  = {"person"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
ANIMAL_CLASSES  = {"dog", "cat", "bird"}

# Cores dos bounding boxes por categoria
COLORS = {
    "person":  (0,   255,  0),    # verde
    "vehicle": (255, 165,  0),    # laranja
    "animal":  (0,   200, 255),   # ciano
    "object":  (200, 200,  0),    # amarelo
    "motion":  (0,     0, 255),   # vermelho
}


# --------------------------------------------------------------------------- #

@dataclass
class Detection:
    label:      str
    confidence: float
    bbox:       list      # [x1, y1, x2, y2]
    category:   str = "" # person / vehicle / animal / object


@dataclass
class DetectionEvent:
    id:          str               = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id:   str               = ""
    camera_name: str               = ""
    event_type:  str               = ""   # person / vehicle / animal / motion / object
    detections:  list              = field(default_factory=list)
    frame:       Optional[object]  = None
    timestamp:   float             = field(default_factory=time.time)

    def summary(self) -> str:
        counts = {}
        for d in self.detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        parts = [f"{v}x {k}" for k, v in counts.items()]
        return f"[{self.camera_name}] {', '.join(parts) or self.event_type}"


# --------------------------------------------------------------------------- #

def _gpu_available() -> bool:
    try:
        return YOLO_AVAILABLE and torch.cuda.is_available()
    except Exception:
        return False


def _select_device() -> tuple:
    """Detecta GPU e escolhe modelo + device. Retorna (model_name, device)."""
    if not YOLO_AVAILABLE:
        return "yolov8n.pt", "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        logger.info(f"GPU detectada: {gpu} → yolov8s.pt + CUDA")
        return "yolov8s.pt", "cuda"
    else:
        logger.info("Sem GPU NVIDIA → yolov8n.pt + CPU")
        return "yolov8n.pt", "cpu"


def _draw_boxes(frame, detections) -> np.ndarray:
    """Desenha bounding boxes anotados no frame."""
    out = frame.copy()
    for d in detections:
        if d.bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = d.bbox
        color = COLORS.get(d.category, COLORS["object"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.label} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #

class MotionDetector:
    """Detecção de movimento por diferença de frames — leve, sem GPU."""

    def __init__(self, sensitivity: float = 0.02):
        self.sensitivity = sensitivity
        self._prev = {}

    def detect(self, camera_id: str, frame) -> float:
        """Retorna fração de pixels com movimento (0.0 – 1.0)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if camera_id not in self._prev:
            self._prev[camera_id] = gray
            return 0.0

        diff   = cv2.absdiff(self._prev[camera_id], gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        ratio  = thresh.sum() / (thresh.size * 255)
        self._prev[camera_id] = gray
        return float(ratio)


# --------------------------------------------------------------------------- #

class ObjectDetector:
    """Detecção de objetos com YOLOv8, com seleção automática de GPU."""

    def __init__(self, config: dict):
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics não instalado.\n"
                "Execute: pip install ultralytics torch"
            )

        det_cfg = config["detection"]
        self.confidence  = det_cfg["confidence"]
        self.target      = set(det_cfg["classes"])
        self.detect_size = 640   # resolução de entrada do YOLO

        cfg_model = det_cfg.get("model", "auto")
        if cfg_model == "auto":
            self.model_name, self.device = _select_device()
        else:
            self.model_name = cfg_model
            self.device     = "cuda" if _gpu_available() else "cpu"

        logger.info(f"Carregando {self.model_name} no device '{self.device}'…")
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        logger.info("Modelo YOLO carregado ✓")

    def detect(self, frame) -> list:
        small = cv2.resize(frame, (self.detect_size, self.detect_size))
        results = self.model(small, conf=self.confidence, verbose=False, device=self.device)[0]

        h_ratio = frame.shape[0] / self.detect_size
        w_ratio = frame.shape[1] / self.detect_size

        detections = []
        for box in results.boxes:
            label = results.names[int(box.cls)]
            if label not in self.target:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * w_ratio); x2 = int(x2 * w_ratio)
            y1 = int(y1 * h_ratio); y2 = int(y2 * h_ratio)

            category = (
                "person"  if label in PERSON_CLASSES  else
                "vehicle" if label in VEHICLE_CLASSES else
                "animal"  if label in ANIMAL_CLASSES  else
                "object"
            )
            detections.append(Detection(
                label=label, confidence=float(box.conf),
                bbox=[x1, y1, x2, y2], category=category,
            ))

        return detections


# --------------------------------------------------------------------------- #

class DetectionPipeline:
    """
    Consome frames da fila do CaptureManager e emite DetectionEvents.

    Roda em thread própria. Usa YOLO para objetos e MotionDetector como fallback.
    Processa 1 a cada N frames para não sobrecarregar CPU/GPU.

    Uso:
        pipeline = DetectionPipeline(config, on_event=meu_callback)
        pipeline.start(frame_queue)
        # para encerrar:
        pipeline.stop()
    """

    def __init__(self, config: dict, on_event: Callable):
        det_cfg           = config["detection"]
        self.on_event     = on_event
        self.motion_sens  = det_cfg["motion_sensitivity"]
        self.min_interval = det_cfg.get("min_detection_interval", 2)
        self._last_event  = {}   # camera_id → timestamp
        self._frame_count = {}   # camera_id → contador

        # Sem GPU: processa 1 a cada 4 frames. Com GPU: todos os frames
        self._skip = 3 if not _gpu_available() else 0

        self.motion = MotionDetector(self.motion_sens)

        try:
            self.yolo = ObjectDetector(config)
        except Exception as e:
            logger.warning(f"YOLO indisponível ({e}) — usando só detecção de movimento")
            self.yolo = None

        self._stop   = threading.Event()
        self._thread = None
        self._queue  = None

    def start(self, frame_queue):
        self._queue = frame_queue
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="detection")
        self._thread.start()
        mode = "GPU" if _gpu_available() else "CPU"
        logger.info(f"DetectionPipeline iniciado [{mode}, skip={self._skip}]")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("DetectionPipeline encerrado")

    def _run(self):
        while not self._stop.is_set():
            try:
                frame_obj = self._queue.get(timeout=1.0)
            except Exception:
                continue
            self.process(frame_obj)

    def process(self, frame_obj):
        cam_id = frame_obj.camera_id
        img    = frame_obj.image
        now    = time.time()

        # Throttle por câmera
        if now - self._last_event.get(cam_id, 0) < self.min_interval:
            return None

        # Skip frames para economizar CPU/GPU
        count = self._frame_count.get(cam_id, 0) + 1
        self._frame_count[cam_id] = count
        if self._skip > 0 and count % (self._skip + 1) != 0:
            return None

        detections = []
        event_type = None

        # 1. YOLO
        if self.yolo:
            detections = self.yolo.detect(img)
            if detections:
                labels = {d.category for d in detections}
                event_type = (
                    "person"  if "person"  in labels else
                    "vehicle" if "vehicle" in labels else
                    "animal"  if "animal"  in labels else
                    "object"
                )

        # 2. Fallback: movimento
        if not detections:
            motion = self.motion.detect(cam_id, img)
            if motion >= self.motion_sens:
                event_type = "motion"
                detections = [Detection("motion", motion, [0, 0, 0, 0], "motion")]

        if not event_type:
            return None

        annotated = _draw_boxes(img, detections)
        event = DetectionEvent(
            camera_id   = cam_id,
            camera_name = frame_obj.camera_name,
            event_type  = event_type,
            detections  = detections,
            frame       = annotated,
            timestamp   = now,
        )

        self._last_event[cam_id] = now
        logger.info(event.summary())
        self.on_event(event)
        return event