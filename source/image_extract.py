import cv2
import os

def crop_image(RGB_image, gray_image):
    
    # Coordenadas e tamanho do recorte
    x = 300
    y = 0
    width = 1280  # múltiplo de 16
    height = 720  # múltiplo de 16
    
    # Aplica o recorte
    crop_RGB= RGB_image[y:y+height, x:x+width]
    crop_gray= gray_image[y:y+height, x:x+width]
    
    return crop_RGB, crop_gray

def main():
    #Caminhos dos arquivos/pastas
    video_path = "videos\Disco10b.avi"
    RGB_folder = "pelet_dateset\RGB_images"
    gray_folder = "images\gray_images"
    # Cria a pasta de saída se não existir
    os.makedirs(RGB_folder, exist_ok=True)
    os.makedirs(gray_folder, exist_ok=True)

    # Abre o vídeo
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        

    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break  # Sai do loop se não conseguir ler o frame

        # Converte o frame para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Recorta os frames
        frame, gray_frame = crop_image(frame, gray_frame)
        
        # Salva o frame como imagem
        gray_frame_filename = os.path.join(gray_folder, f"frame_{frame_count:05d}.jpg")
        RGB_frame_filename = os.path.join(RGB_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(gray_frame_filename, gray_frame)
        cv2.imwrite(RGB_frame_filename, frame)
        frame_count += 1

    video.release()
    print(f"Extração concluída!")


if __name__ == "__main__":
    main()