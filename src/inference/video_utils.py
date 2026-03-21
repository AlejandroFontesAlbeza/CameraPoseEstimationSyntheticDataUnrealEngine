import time

import cv2
import numpy as np
from PIL import Image

import torch


def mask_generator(video_path, model, transform, device, resize=None, inference_palette=None):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        width, height = frame.shape[1], frame.shape[0]
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            output = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            print(f'Inference time: {end-start}')
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        predicted_mask_color = np.zeros((height, width, 3), dtype=np.uint8)
        for class_index, color in inference_palette.items():
            predicted_mask_color[predicted_mask_resized == class_index] = color
        yield frame, predicted_mask_color
    cap.release()



