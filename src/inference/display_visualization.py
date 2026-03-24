import time

import cv2
import numpy as np
from PIL import Image

import torch


def draw_lines_and_intersections(img_np, predicted_mask_resized, inference_color_palette, intersections_lines, min_pixels=100):
    lines = {}
    intersections = {}

    # Si la imagen es BGR y la paleta es RGB, convierte los colores de la paleta a BGR
    def to_bgr(color):
        return (color[2], color[1], color[0])

    # Dibuja líneas ajustadas a los contornos de cada clase
    for class_index, color in inference_color_palette.items():
        if class_index == 0:
            continue
        mask_binary = (predicted_mask_resized == class_index).astype(np.uint8)
        if np.count_nonzero(mask_binary) < min_pixels:
            continue
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = max(contours, key=cv2.contourArea)
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = vx.item(), vy.item(), x.item(), y.item()
        lines[class_index] = (vx, vy, x, y)
        cv2.line(img_np, (int(x - vx * 1000), int(y - vy * 1000)), (int(x + vx * 1000), int(y + vy * 1000)), to_bgr(color), 4)

    # Dibuja intersecciones
    for index, (c1, c2) in intersections_lines.items():
        if c1 in lines and c2 in lines:
            vx1, vy1, x1, y1 = lines[c1]
            vx2, vy2, x2, y2 = lines[c2]
            A = np.array([[vx1, -vx2], [vy1, -vy2]])
            b = np.array([x2 - x1, y2 - y1])
            t, s = np.linalg.solve(A, b)
            x_int = int(x2 + vx2 * s)
            y_int = int(y2 + vy2 * s)
            intersections[index] = (x_int, y_int)
            cv2.circle(img_np, (x_int, y_int), 8, (255, 255, 255), -1)
            cv2.putText(img_np, f'{index}', (x_int + 10, y_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    return img_np, intersections

class VideoInference:
    def __init__(self, video_path, model, transform, device, inference_palette=None):
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_path = video_path
        self.model = model
        self.transform = transform
        self.device = device
        self.inference_palette = inference_palette

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(2):
                _ = self.model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            output = self.model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        predicted_mask_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for class_index, color in self.inference_palette.items():
            mask = (predicted_mask_resized == class_index)
            predicted_mask_color[mask] = color
        predicted_mask_color = cv2.cvtColor(predicted_mask_color, cv2.COLOR_RGB2BGR)
        return frame, predicted_mask_resized, predicted_mask_color

    def get_size(self):
        return self.width, self.height

    def release(self):
        self.cap.release()




