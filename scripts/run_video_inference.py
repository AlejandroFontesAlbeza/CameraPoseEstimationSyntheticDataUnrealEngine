import os
import cv2
import time
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from unet.unet import Unet
from config import INFERENCE_VIDEO_PATH, INFERENCE_MODEL_PATH
from utils.palette import inference_color_palette


def inference_video(video_path, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Unet(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(video_path)
    width, height = 640, 480
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
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

        for class_index, color in inference_color_palette.items():
            predicted_mask_color[predicted_mask_resized == class_index] = color

        combined = np.hstack((frame_resized, predicted_mask_color))
        cv2.imshow("Original + Predicted", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_video(INFERENCE_VIDEO_PATH, INFERENCE_MODEL_PATH)
