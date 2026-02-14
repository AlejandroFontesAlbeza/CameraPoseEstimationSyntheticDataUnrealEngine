import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from camera_pose.unet.training.architecture import CustomDataset, Unet
from tqdm import tqdm
import os


def main(img_path, mask_path, num_classes, batch_size, num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training on:', device)

    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(img_path, mask_path, img_transform)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.6,0.4], generator=generator)

    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    model = Unet(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print('training...')

        for index, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            masks = masks.to(device)

            
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_average_loss = train_loss / (index + 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_average_loss = val_loss / (index + 1)
        scheduler.step()
        
        print(f'Epoch: {epoch}, train_loss: {train_average_loss}, val_loss: {val_average_loss}, lr: {scheduler.get_last_lr()[0]}') 

    torch.save(model.state_dict(), 'unet_model.pth')
    print('model saved as unet_model.pth') 
    print("prueba para comprobar que no esta tracked egg-info")



if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    img_path = os.path.join(base_dir, "camera_pose/dataset/trainDataset/images")
    mask_path = os.path.join(base_dir, "camera_pose/dataset/trainDataset/masks")
    main(img_path=img_path, mask_path=mask_path, num_classes=10, batch_size=2, num_epochs=20)