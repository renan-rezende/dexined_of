import cv2 as cv
import numpy as np
import os

input_folder = 'pelotas_dataset/images/detected'
output_folder = 'pelotas_dataset/images/masks'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.bmp')):
        continue

    path = os.path.join(input_folder, filename)
    img = cv.imread(path)

    if img is None:
        print(f"[ERRO] Falha ao ler: {filename}")
        continue

    # Mascara para regiões quase brancas (tolerância)
    # Aqui consideramos qualquer valor acima de 250 em todos os canais como branco
    lower = np.array([250, 250, 250])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(img, lower, upper)

    # Salva a imagem binária (só o que era branco puro)
    out_path = os.path.join(output_folder, f"mask_{filename}")
    cv.imwrite(out_path, mask)
    print(f"[OK] Máscara salva: {out_path}")
