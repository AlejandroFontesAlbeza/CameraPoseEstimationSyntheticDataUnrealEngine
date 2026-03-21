import cv2
import numpy as np

import torch
from torchvision import transforms

from unet.unet import Unet
from config import INFERENCE_VIDEO_PATH, INFERENCE_MODEL_PATH
from utils.palette import inference_color_palette
from inference.video_utils import mask_generator

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Unet(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(INFERENCE_MODEL_PATH, map_location=device))
    print("Model loaded successfully.")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])


    resize = (640,480)
    show_mask = False

    gen = mask_generator(INFERENCE_VIDEO_PATH, model,
                        transform, device, resize=resize,
                        inference_palette=inference_color_palette)

    for frame, predicted_mask_color in gen:
        if show_mask:
            display = np.hstack((frame, predicted_mask_color))
        else:
            display = frame
        cv2.imshow('Inference', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
