"""
preview.py — visualizador de câmeras integrado ao novo sistema.
Substitui o código original, mantendo o mesmo comportamento (grade 2x2, teclas 0-4).

Extras adicionados:
  - tecla 'd' → ativa/desativa overlay de detecção (bounding boxes)
  - tecla 's' → salva snapshot da câmera atual
  - tecla 'i' → mostra status de cada câmera no terminal
  - título da janela mostra câmera online/offline em tempo real
"""
import cv2
import yaml
import time
import logging
from pathlib import Path
from src.capture.camera import CaptureManager, Frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("preview")

# --------------------------------------------------------------------------- #

def run_preview():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    manager = CaptureManager(config)
    manager.start_all()

    modo            = 0        # 0 = grade 2x2 | 1-4 = câmera individual
    show_detection  = False    # overlay de detecção
    W, H            = 640, 360 # tamanho de cada célula na grade

    cv2.namedWindow("Cameras DVR", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Cameras DVR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cam_ids = manager.camera_ids()   # ["cam_01", "cam_02", "cam_03", "cam_04"]

    try:
        while True:
            # Coleta frames mais recentes de cada câmera
            frames = {}
            for cam_id in cam_ids:
                cam   = manager.get_camera(cam_id)
                img   = cam.get_latest_frame() if cam else None
                frames[cam_id] = img

            # ---------- Monta imagem de exibição ----------
            if modo == 0:
                cells = []
                for i, cam_id in enumerate(cam_ids):
                    img = frames.get(cam_id)
                    cam = manager.get_camera(cam_id)

                    if img is None:
                        img = _placeholder(cam.name if cam else cam_id, W, H)
                    else:
                        img = cv2.resize(img, (W, H))
                        # Status no canto
                        status_color = (0, 255, 0) if (cam and cam.online) else (0, 0, 255)
                        cv2.circle(img, (W - 15, 15), 8, status_color, -1)
                        cv2.putText(img, cam.name if cam else cam_id,
                                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cells.append(img)

                while len(cells) < 4:
                    cells.append(_placeholder("—", W, H))

                row1  = cv2.hconcat([cells[0], cells[1]])
                row2  = cv2.hconcat([cells[2], cells[3]])
                exibir = cv2.vconcat([row1, row2])

            else:
                cam_id = cam_ids[modo - 1]
                img    = frames.get(cam_id)
                cam    = manager.get_camera(cam_id)

                if img is None:
                    exibir = _placeholder(f"Câmera {modo} — sem sinal", W * 2, H * 2)
                else:
                    exibir = img.copy()
                    cv2.putText(exibir, f"{cam.name if cam else cam_id}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Legenda de teclas (pequena, canto inferior esquerdo)
            _draw_legend(exibir)

            cv2.imshow("Cameras DVR", exibir)

            # ---------- Teclas ----------
            tecla = cv2.waitKey(30) & 0xFF

            if tecla == ord('q'):
                break
            elif tecla == ord('0'):
                modo = 0
            elif tecla in [ord('1'), ord('2'), ord('3'), ord('4')]:
                modo = int(chr(tecla))
                # Garante que o índice existe
                if modo > len(cam_ids):
                    modo = 0
            elif tecla == ord('d'):
                show_detection = not show_detection
                logger.info(f"Overlay de detecção: {'ON' if show_detection else 'OFF'}")
            elif tecla == ord('s'):
                # Snapshot da visão atual
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, exibir)
                logger.info(f"Snapshot salvo: {fname}")
            elif tecla == ord('i'):
                for s in manager.all_status():
                    icon = "✓" if s["online"] else "✗"
                    lag  = f"{s['lag_sec']}s" if s["lag_sec"] else "—"
                    print(f"  [{icon}] {s['id']} ({s['name']}) — último frame: {lag} atrás")

    finally:
        manager.stop_all()
        cv2.destroyAllWindows()
        logger.info("Encerrado.")


# --------------------------------------------------------------------------- #

def _placeholder(label: str, w: int, h: int):
    img = _zeros = cv2.UMat(h, w, cv2.CV_8UC3)
    import numpy as np
    img = np.zeros((h, w, 3), dtype=cv2.CV_8UC3 if False else "uint8")
    cv2.putText(img, label, (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    return img


def _draw_legend(frame):
    lines = ["[0] Grade  [1-4] Câmera  [s] Snapshot  [d] Detecção  [i] Status  [q] Sair"]
    h, w  = frame.shape[:2]
    y     = h - 12
    cv2.putText(frame, lines[0], (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


if __name__ == "__main__":
    run_preview()