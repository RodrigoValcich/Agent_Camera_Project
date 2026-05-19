import cv2
import numpy as np

usuario = "admin"
senha = "Malli84"
ip = "192.168.1.4"
porta = "554"

# URLs das 4 câmeras
urls = [
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=2&subtype=0",
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=3&subtype=0",
    f"rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel=4&subtype=0",
]

# Abre as 4 câmeras
caps = [cv2.VideoCapture(url) for url in urls]

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
        else:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Sem sinal", (220, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frames.append(frame)

    # Monta grade 2x2
    linha1 = np.hstack((frames[0], frames[1]))
    linha2 = np.hstack((frames[2], frames[3]))
    grade = np.vstack((linha1, linha2))

    cv2.imshow("Cameras DVR", grade)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()