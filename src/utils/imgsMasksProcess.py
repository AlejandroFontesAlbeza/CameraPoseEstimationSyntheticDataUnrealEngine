from PIL import Image
import numpy as np
import os
from tqdm import tqdm


from concurrent.futures import ProcessPoolExecutor

exactColorPalette = {
    (0, 0, 0): 0,       # Background
    (255, 0, 0): 1,     # Class 1
    (0, 255, 0): 2,     # Class 2
    (0, 0, 255): 3,     # Class 3
    (255, 255, 0): 4,   # Class 4
    (255, 0, 255): 5,   # Class 5
    (0, 255, 255): 6,   # Class 6
}

rangeColorPalette = {
    "class7": {"range": ((183,189), (0,0), (250,255)), "classIndex": 7}, # Class 7 needs to be range
    "class8": {"range": ((200,255), (113,192), (0,0)), "classIndex": 8}, # Class 8 needs to be range
    "class9": {"range": ((0,0), (221,255), (162,189)), "classIndex": 9} # Class 9 needs to be range
}

def dataProcess(args):

    inputMaskPath, outputMaskPath, inputImagePath, outputImagePath = args
    mask = Image.open(inputMaskPath).convert("RGB")
    mask_pixels = np.array(mask)
    height, width = mask_pixels.shape[:2]
    grayMask = np.zeros((height, width), dtype=np.uint8)

    r_channel = mask_pixels[:, :, 0]
    g_channel = mask_pixels[:, :, 1]
    b_channel = mask_pixels[:, :, 2]

    # Exact color matches
    for color, classIndex in exactColorPalette.items():
        match = np.all(mask_pixels == color, axis=-1)  # shape (H, W)
        grayMask[match] = classIndex

    # Range-based matches
    for classInfo in rangeColorPalette.values():
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = classInfo["range"]
        classIndex = classInfo["classIndex"]
        match = (
            (r_channel >= r_min) & (r_channel <= r_max) &
            (g_channel >= g_min) & (g_channel <= g_max) &
            (b_channel >= b_min) & (b_channel <= b_max)
        )
        grayMask[match] = classIndex
    grayMaskImage = Image.fromarray(grayMask)
    grayMaskImageResized = grayMaskImage.resize((512,512), resample=Image.NEAREST)
    grayMaskImageResized.save(outputMaskPath)

    image = Image.open(inputImagePath).convert("RGB")
    imageResized = image.resize((512,512), resample=Image.BILINEAR)
    imageResized.save(outputImagePath)
    return 0

def processFolders(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath, numWorkers=None):

    tasks = buildTasks(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath)

    with ProcessPoolExecutor(max_workers=numWorkers) as executor:
        list(
            tqdm(
                executor.map(dataProcess, tasks),
                total=len(tasks),
                desc="Processing images and masks"
            )
        )

def buildTasks(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath):
    tasks = []

    for filename in sorted(os.listdir(masksUEFolderPath)):
        if filename.lower().endswith(".png"):
            tasks.append((
                os.path.join(masksUEFolderPath, filename),
                os.path.join(masksFolderPath, filename),
                os.path.join(imagesUEFolderPath, filename),
                os.path.join(imagesFolderPath, filename)
            ))
    return tasks


if __name__ == "__main__":
    imagesUEFolderPath = "../dataset/trainDataset/imagesUE"
    masksUEFolderPath = "../dataset/trainDataset/masksUE"
    imagesFolderPath = "../dataset/trainDataset/images"
    masksFolderPath = "../dataset/trainDataset/masks"
    processFolders(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath, os.cpu_count())

