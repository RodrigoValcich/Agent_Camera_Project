"""
Módulo de gravação.

Dois modos:
  - Contínua: grava tudo desde que a aplicação está aberta (um arquivo por câmera por hora)
  - Por evento: grava clipes pré+pós detecção (modo anterior, mantido para referência)

O modo padrão agora é CONTÍNUO.
"""
import cv2
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from src.capture.camera import CameraCapture
    from src.detection.detector import DetectionEvent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Gravação contínua

class ContinuousRecorder:
    """
    Grava o stream de uma câmera continuamente.
    Cria um novo arquivo a cada hora para não gerar arquivos gigantes.
    """

    def __init__(self, camera_id: str, camera_name: str, config: dict):
        rec = config["recording"]
        self.camera_id   = camera_id
        self.camera_name = camera_name
        self.fps         = rec["fps"]
        self.output_dir  = Path(rec["output_dir"]) / camera_id / "continuous"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._writer:    Optional[cv2.VideoWriter] = None
        self._file_path: Optional[str]             = None
        self._current_hour = -1
        self._lock = threading.Lock()

    def write(self, img: np.ndarray):
        """Escreve um frame. Cria novo arquivo se virou a hora."""
        now  = datetime.now()
        hour = now.hour

        with self._lock:
            if hour != self._current_hour or self._writer is None:
                self._rotate(img, now)

            if self._writer:
                self._writer.write(img)

    def stop(self):
        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None
        logger.info(f"[{self.camera_id}] gravação contínua encerrada")

    def _rotate(self, img: np.ndarray, now: datetime):
        """Fecha arquivo atual e abre novo para a hora corrente."""
        if self._writer:
            self._writer.release()

        self._current_hour = now.hour
        ts = now.strftime("%Y%m%d_%H0000")
        self._file_path = str(self.output_dir / f"{self.camera_id}_{ts}.mp4")

        h, w = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._file_path, fourcc, self.fps, (w, h))
        logger.info(f"[{self.camera_id}] novo arquivo: {self._file_path}")


# --------------------------------------------------------------------------- #
# Gravação de clipes por evento (mantida para thumbnails)

class ClipRecorder:
    """Grava um clipe curto ao redor de um evento detectado (para thumbnail)."""

    def __init__(self, event, pre_frames: list, config: dict):
        rec = config["recording"]
        self.event        = event
        self.pre_frames   = pre_frames
        self.post_seconds = rec["post_event_seconds"]
        self.fps          = rec["fps"]
        self.output_dir   = Path(rec["output_dir"])
        self._collecting  = True
        self._post_buf:   list[np.ndarray] = []
        self._lock        = threading.Lock()
        self._start_time  = time.time()

    def add_post_frame(self, img: np.ndarray):
        with self._lock:
            if not self._collecting:
                return
            if time.time() - self._start_time >= self.post_seconds:
                self._collecting = False
                return
            self._post_buf.append(img.copy())

    def is_collecting(self) -> bool:
        return self._collecting

    def save(self) -> tuple[str, str]:
        self._collecting = False
        cam_id = self.event.camera_id
        ts     = datetime.fromtimestamp(self.event.timestamp).strftime("%Y%m%d_%H%M%S")
        stem   = f"{cam_id}_{ts}_{self.event.event_type}"

        cam_dir = self.output_dir / cam_id / "events"
        cam_dir.mkdir(parents=True, exist_ok=True)

        clip_path  = str(cam_dir / f"{stem}.mp4")
        thumb_path = str(cam_dir / f"{stem}_thumb.jpg")

        # Thumbnail
        thumb_frame = cv2.resize(self.event.frame, (480, 270))
        cv2.imwrite(thumb_path, thumb_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Clipe
        all_frames = [f.image for f in self.pre_frames] + [self.event.frame] + self._post_buf
        if all_frames:
            h, w   = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(clip_path, fourcc, self.fps, (w, h))
            for img in all_frames:
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                writer.write(img)
            writer.release()

        logger.info(f"Clipe de evento salvo: {clip_path}")
        return clip_path, thumb_path


# --------------------------------------------------------------------------- #

class RecordingManager:
    """
    Gerencia gravação contínua de todas as câmeras
    e clipes de eventos para thumbnails.
    """

    def __init__(self, config: dict, on_clip_ready=None):
        self.config        = config
        self.on_clip_ready = on_clip_ready
        self._continuous:  dict[str, ContinuousRecorder] = {}
        self._active_clips: dict[str, ClipRecorder]      = {}
        self._lock = threading.Lock()
        self._loop = None

    def set_event_loop(self, loop):
        self._loop = loop

    def init_continuous(self, camera_ids: list[str]):
        """Inicia gravação contínua para todas as câmeras."""
        for cam_id in camera_ids:
            # Busca nome da câmera na config
            name = cam_id
            for c in self.config["cameras"]:
                if c["id"] == cam_id:
                    name = c.get("name", cam_id)
                    break
            self._continuous[cam_id] = ContinuousRecorder(cam_id, name, self.config)
            logger.info(f"[{cam_id}] gravação contínua iniciada")

    def feed_frame(self, camera_id: str, img: np.ndarray):
        """
        Alimenta um frame para:
        1. Gravação contínua
        2. Pós-buffer de clipes de evento ativos
        """
        # Gravação contínua
        rec = self._continuous.get(camera_id)
        if rec:
            rec.write(img)

        # Pós-buffer de evento
        with self._lock:
            clip = self._active_clips.get(camera_id)
        if clip and clip.is_collecting():
            clip.add_post_frame(img)

    def on_event(self, event: "DetectionEvent", camera: "CameraCapture"):
        """Inicia gravação de clipe de evento (para thumbnail)."""
        cam_id     = event.camera_id
        pre_frames = camera.get_pre_buffer()
        clip       = ClipRecorder(event, pre_frames, self.config)

        with self._lock:
            self._active_clips[cam_id] = clip

        threading.Timer(
            self.config["recording"]["post_event_seconds"] + 1,
            self._finalize_clip,
            args=(cam_id, clip),
        ).start()

    def _finalize_clip(self, cam_id: str, clip: ClipRecorder):
        clip_path, thumb_path = clip.save()
        with self._lock:
            if self._active_clips.get(cam_id) is clip:
                del self._active_clips[cam_id]

        if self.on_clip_ready and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.on_clip_ready(clip.event, clip_path, thumb_path),
                self._loop,
            )

    def stop_all(self):
        for rec in self._continuous.values():
            rec.stop()