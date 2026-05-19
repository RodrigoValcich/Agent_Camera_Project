"""
Módulo de captura de câmeras via RTSP usando OpenCV.
Compatível com DVRs Dahua/Intelbras usando o padrão:
  rtsp://usuario:senha@ip:porta/cam/realmonitor?channel=N&subtype=0

Credenciais lidas do .env (mesmo formato do código original do usuário).
Cada câmera roda numa thread separada e publica frames numa fila compartilhada.
"""
import os
import cv2
import threading
import time
import logging
from queue import Queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def build_rtsp_url(channel: int, subtype: int = 0) -> str:
    """
    Monta a URL RTSP no padrão Dahua/Intelbras a partir do .env.
    subtype=0 → stream principal (alta resolução)
    subtype=1 → stream secundário (menor resolução, mais leve)
    """
    usuario = os.getenv("usuario")
    senha   = os.getenv("senha")
    ip      = os.getenv("ip")
    porta   = os.getenv("porta", "554")
    return f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel={channel}&subtype={subtype}"


# --------------------------------------------------------------------------- #

@dataclass
class Frame:
    camera_id: str
    camera_name: str
    image: np.ndarray
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #

class CameraCapture:
    """
    Captura contínua de uma câmera RTSP em thread dedicada.
    Mantém um buffer circular de frames para gravação pré-evento.
    Reconecta automaticamente em caso de queda do DVR.
    """

    def __init__(
        self,
        camera_cfg: dict,
        frame_queue: Queue,
        buffer_seconds: int = 10,
        fps: int = 15,
        detection_subtype: int = 1,   # subtype=1 para detecção (leve)
        recording_subtype: int = 0,   # subtype=0 para gravação (qualidade total)
    ):
        self.id       = camera_cfg["id"]
        self.name     = camera_cfg["name"]
        channel       = camera_cfg["channel"]

        # Duas URLs: uma leve para detecção, outra full para gravação
        self.url_detect = build_rtsp_url(channel, subtype=detection_subtype)
        self.url_record = build_rtsp_url(channel, subtype=recording_subtype)

        self.queue   = frame_queue
        self.fps     = fps
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap:    Optional[cv2.VideoCapture]  = None

        # Buffer circular para frames pré-evento
        self.pre_buffer: list[Frame] = []
        self.pre_buffer_max = buffer_seconds * fps
        self._buf_lock = threading.Lock()

        # Status para o dashboard
        self.online = False
        self.last_frame_time: Optional[float] = None

    # ------------------------------------------------------------------ #
    # API pública

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"cam-{self.id}"
        )
        self._thread.start()
        logger.info(f"[{self.id}] '{self.name}' iniciada → ch{self._channel_from_url()}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        self.online = False
        logger.info(f"[{self.id}] encerrada")

    def get_pre_buffer(self) -> list[Frame]:
        """Retorna cópia do buffer pré-evento (thread-safe)."""
        with self._buf_lock:
            return list(self.pre_buffer)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Frame mais recente para exibição no dashboard."""
        with self._buf_lock:
            if self.pre_buffer:
                return self.pre_buffer[-1].image.copy()
        return None

    def status(self) -> dict:
        age = (time.time() - self.last_frame_time) if self.last_frame_time else None
        return {
            "id":      self.id,
            "name":    self.name,
            "online":  self.online,
            "lag_sec": round(age, 1) if age is not None else None,
        }

    # ------------------------------------------------------------------ #
    # Internos

    def _channel_from_url(self) -> str:
        import re
        m = re.search(r"channel=(\d+)", self.url_detect)
        return m.group(1) if m else "?"

    def _open(self, url: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # evita lag acumulado
        return cap

    def _run(self):
        retry_delay   = 5
        frame_interval = 1.0 / self.fps

        while not self._stop.is_set():
            self._cap = self._open(self.url_detect)

            if not self._cap.isOpened():
                logger.warning(f"[{self.id}] sem conexão, retry em {retry_delay}s")
                self.online = False
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)   # backoff até 60s
                continue

            logger.info(f"[{self.id}] conectado ✓")
            self.online    = True
            retry_delay    = 5   # reset backoff

            while not self._stop.is_set():
                t0 = time.time()
                ret, img = self._cap.read()

                if not ret:
                    logger.warning(f"[{self.id}] frame falhou — reconectando…")
                    self.online = False
                    break

                self.last_frame_time = time.time()
                frame = Frame(camera_id=self.id, camera_name=self.name, image=img)

                # Atualiza buffer pré-evento
                with self._buf_lock:
                    self.pre_buffer.append(frame)
                    if len(self.pre_buffer) > self.pre_buffer_max:
                        self.pre_buffer.pop(0)

                # Envia para fila de detecção (descarta se cheia)
                try:
                    self.queue.put_nowait(frame)
                except Exception:
                    pass

                # Throttle de FPS
                elapsed = time.time() - t0
                wait    = frame_interval - elapsed
                if wait > 0:
                    time.sleep(wait)

            self._cap.release()


# --------------------------------------------------------------------------- #

class CaptureManager:
    """
    Gerencia as 4 câmeras do DVR e expõe uma fila unificada de frames.

    Uso:
        manager = CaptureManager(config)
        manager.start_all()

        while True:
            frame = manager.queue.get()   # Frame com .camera_id, .image, .timestamp
            process(frame)
    """

    def __init__(self, config: dict):
        self.config = config
        self.queue: Queue[Frame] = Queue(maxsize=80)
        self._cameras: dict[str, CameraCapture] = {}

        fps = config["recording"]["fps"]
        pre = config["recording"]["pre_event_seconds"]

        for cam_cfg in config["cameras"]:
            if not cam_cfg.get("enabled", True):
                continue
            self._cameras[cam_cfg["id"]] = CameraCapture(
                cam_cfg,
                frame_queue=self.queue,
                buffer_seconds=pre,
                fps=fps,
            )

        logger.info(f"CaptureManager: {len(self._cameras)} câmeras configuradas")

    def start_all(self):
        for cam in self._cameras.values():
            cam.start()

    def stop_all(self):
        for cam in self._cameras.values():
            cam.stop()

    def get_camera(self, camera_id: str) -> Optional[CameraCapture]:
        return self._cameras.get(camera_id)

    def camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

    def all_status(self) -> list[dict]:
        return [cam.status() for cam in self._cameras.values()]

    def grid_frame(self, width: int = 640, height: int = 360) -> np.ndarray:
        """
        Monta grade 2x2 com os frames mais recentes — igual ao seu código original.
        Útil para debug ou preview rápido.
        """
        cells = []
        for cam in self._cameras.values():
            img = cam.get_latest_frame()
            if img is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(img, f"{cam.name} - Sem sinal",
                            (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
            else:
                img = cv2.resize(img, (width, height))
            cells.append(img)

        # Completa até 4 células
        while len(cells) < 4:
            cells.append(np.zeros((height, width, 3), dtype=np.uint8))

        row1 = np.hstack((cells[0], cells[1]))
        row2 = np.hstack((cells[2], cells[3]))
        return np.vstack((row1, row2))