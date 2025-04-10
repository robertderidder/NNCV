"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    RandomVerticalFlip,
)

from model import Model
from diceloss import MultiDiceLoss

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch Deeplab")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr1", type=float, default=0.01, help="Learning rate classifier")
    parser.add_argument("--lr2", type=float, default=0.001, help="Learning rate backbone")
    parser.add_argument("--decay",type=float,default=0.7,help="decay")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="deeplab", help="Experiment ID for Weights & Biases")

    return parser

def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-robustness_challenge",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model().to(device)
    
    for param in model.model.backbone.parameters():
        param.requires_grad = True  # Unfreeze the backbone
        
    for param in model.model.classifier.parameters():
        param.requires_grad = True
          
    class PaintingByNumbersTransform:
      def __init__(self, id_to_color=None):
          self.id_to_color = id_to_color  # Dictionary mapping class IDs to colors
          
      def random_recolor(self, label_img):
          """Assigns random colors to segmentation labels."""
          h, w = label_img.shape[1:]
          recolored = torch.zeros((3, h, w), dtype=torch.uint8)  # Create an empty RGB image
                  
          unique_labels = label_img.unique()
          color_map = {label.item(): torch.randint(0, 256, (3,), dtype=torch.uint8) for label in unique_labels}

          for label, color in color_map.items():
              mask = (label_img[0] == label)  # label_img shape is [1, h, w]
              for c in range(3):
                  recolored[c][mask] = color[c]
             
          return recolored
  
      def __call__(self, img, target):
          if torch.rand(1).item() > 0.5:
              # Load the actual ground truth color image
              gt_color = self.random_recolor(target)
  
              # Blend image and color segmentation map
              alpha = torch.rand(1).item() * 0.29 + 0.7  # Random alpha between 0.7 and 0.99
              blended_img = alpha * img + (1 - alpha) * gt_color.float() / 255.0
              return blended_img, target
          
          return img, target  # If not applying transformation, return original


    # Define the transforms to apply to the data
    transform1 = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #Parameters required for deeplabV3
        PaintingByNumbersTransform(),
        RandomVerticalFlip(p=0.5),
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform1
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform1
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the loss function
    #criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    criterion = MultiDiceLoss()

    # Define the optimizer
    lr1 = args.lr1
    lr2 = args.lr2
    
    optimizer1 = AdamW([
    {"params": model.model.classifier.parameters(), "lr": lr1}  # Higher LR for classifier
    ])

    optimizer2 = AdamW([
    {"params": model.model.backbone.parameters(), "lr": lr2}  # Lower LR for backbone
    ])

    scheduler = lr_scheduler.StepLR(optimizer1, 10, gamma=args.decay, last_epoch=-1)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
    
        last_lr1 = scheduler.get_last_lr()[0]  # Returns a list, take the first value
        last_lr2 = lr2


        #print(f"Epoch {epoch+1:04}/{args.epochs:04}, lr = {last_lr1:.3E}")
        print(f"Epoch {epoch+1:04}/{args.epochs:04}, lr_classifier = {last_lr1:.3E}, lr_backbone = {last_lr2:.3E}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            outputs = model.model(images)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer1.param_groups[0]['lr'],
                "testrate": last_lr2,
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model.model(images)['out']
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

        scheduler.step()
                    
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
