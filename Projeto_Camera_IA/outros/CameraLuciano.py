import cv2
usuario = "admin"
ip = "192.168.1.4"
porta = "554"
senha = 'Malli84'
url = f"rtsp://{usuario}:{senha}@{ip}:{porta}/"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera DVR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()