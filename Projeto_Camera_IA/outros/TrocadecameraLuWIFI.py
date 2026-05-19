import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
usuario = os.getenv('usuario')
senha = os.getenv('senha')
ip = os.getenv('ip')
porta = os.getenv('porta')

urls = [
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=2&subtype=0",
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=3&subtype=0",
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=4&subtype=0",
]

caps = [cv2.VideoCapture(url) for url in urls]

modo = 0  # 0 = grade 2x2, 1-4 = câmera individual

cv2.namedWindow("Cameras DVR", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Cameras DVR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Lê todos os frames
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            vazio = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(vazio, f"Camera {i+1} - Sem sinal", (100, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frames.append(vazio)

    if modo == 0:
        # Grade 2x2
        f = [cv2.resize(f, (640, 360)) for f in frames]
        linha1 = np.hstack((f[0], f[1]))
        linha2 = np.hstack((f[2], f[3]))
        exibir = np.vstack((linha1, linha2))
    else:
        # Câmera individual tela cheia
        exibir = frames[modo - 1]
        cv2.putText(exibir, f"Camera {modo}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Cameras DVR", exibir)

    # Teclas
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('0'):
        modo = 0
    elif tecla == ord('1'):
        modo = 1
    elif tecla == ord('2'):
        modo = 2
    elif tecla == ord('3'):
        modo = 3
    elif tecla == ord('4'):
        modo = 4

for cap in caps:
    cap.release()
cv2.destroyAllWindows()