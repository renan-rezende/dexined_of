import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Diretórios de entrada e saída
input_folder = 'images/gray_images'
output_folder = 'images/detected'

# Cria pasta de saída, se não existir
os.makedirs(output_folder, exist_ok=True)

# Parâmetros de área (em pixels)
area_min = 100
area_max = 3000  # ajuste conforme necessário

# Itera sobre os arquivos da pasta
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        continue

    path = os.path.join(input_folder, filename)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERRO] Não foi possível ler {filename}")
        continue

    # Pré-processamento
    img_blur = cv.GaussianBlur(img, (7, 7), 0)
    ret, thresh = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh_inv = cv.bitwise_not(thresh)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh_inv, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
    ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.watershed(img_color, markers)

    # Filtragem por área e contorno
    result_img = img_color.copy()
    for label in np.unique(markers):
        if label <= 1:
            continue

        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[markers == label] = 255

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area_min < area < area_max:
                cv.drawContours(result_img, [cnt], -1, (255, 255, 255), 1)

    # Salvar imagem resultante
    out_path = os.path.join(output_folder, f'detect_{filename}')
    cv.imwrite(out_path, result_img)
    print(f"[OK] Processado e salvo: {out_path}")
