import torch

from training.metrics import epoch_trained
from training.utils import device_selection, get_img_transform, get_data_loaders, get_model, optimizations
import argparse
import config_training


def main():

    parser = argparse.ArgumentParser(description='Train a UNet model for image segmentation')

    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for segmentation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--finetuning', action='store_true', help='Whether to perform fine-tuning')
    parser.add_argument('--step_lr', action='store_true', help='Whether to use StepLR')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for StepLR')

    args = parser.parse_args()

    train_img_path = config_training.DATASET_IMAGES_TRAIN_DIR
    train_mask_path = config_training.DATASET_MASKS_TRAIN_DIR
    valid_img_path = config_training.DATASET_IMAGES_VALID_DIR
    valid_mask_path = config_training.DATASET_MASKS_VALID_DIR
    model_path = config_training.MODEL_PATH
    new_model_path = config_training.NEW_MODEL_PATH

    num_classes = args.num_classes
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    finetuning = args.finetuning
    step_lr = args.step_lr
    step_size = args.step_size


    print('Training parameters:')
    print(f'num_classes: {num_classes}')
    print(f'lr: {lr}')
    print(f'batch_size: {batch_size}')
    print(f'num_epochs: {num_epochs}')
    print(f'finetuning: {finetuning}')
    print(f'model_path: {model_path}')
    print(f'new_model_path: {new_model_path}')
    print(f'step_lr: {step_lr}')
    print(f'step_size: {step_size}')

    device = device_selection()

    img_transform = get_img_transform()

    train_loader, val_loader = get_data_loaders(train_img_path, train_mask_path,
                                                valid_img_path, valid_mask_path,
                                                img_transform, batch_size)

    model = get_model(num_classes, device, finetuning, model_path)
    optimizer, scheduler, criterion = optimizations(model, lr, step_size)

    for epoch in range(num_epochs):

        train_average_loss, val_average_loss, miou = epoch_trained(model, num_classes, ignore_index=0,
                                                            train_loader=train_loader,
                                                            val_loader=val_loader, criterion=criterion,
                                                            optimizer=optimizer, device=device)

        if step_lr == True:
            scheduler.step()
        else:
            print('Not using StepLR, lr is fixed')

        print(f'Epoch: {epoch} / {num_epochs}')
        print(f'Training Loss: {train_average_loss:.4f}')
        print(f'Validation Loss: {val_average_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'mIoU: {miou * 100:.2f}%')

    print('model saved')
    torch.save(model.state_dict(), new_model_path)


if __name__ == "__main__":
    main()



