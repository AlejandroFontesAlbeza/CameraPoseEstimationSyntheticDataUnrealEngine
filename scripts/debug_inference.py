from utils.dict_utils import inference_color_palette
from unet.unet import Unet


import os
import time
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as Rot


import torch
from torchvision import transforms



def inference(model_path, input_path):

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

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    width, height = img.size
    print(f"Input image size: {img.size}")
    input_tensor = transform(img).unsqueeze(0).to(device)

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
        mask = (predicted_mask_resized == class_index)
        predicted_mask_color[mask] = color

    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.imshow(img_np)
    plt.subplot(1,2,2)
    plt.imshow(predicted_mask_color)
    plt.show()




if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "models", "unet_modelNew.pth")
    input_path = os.path.join(script_dir, "..", "data", "tennisMatch", "frames", "frame0019.png")
    inference(model_path, input_path=input_path)
