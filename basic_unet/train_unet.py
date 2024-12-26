import argparse
import logging
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from glasses.models.segmentation.unet import UNet
from ms_dataset import MSDataset, get_train_transforms, get_val_transforms
from torchinfo import summary  # Import torchsummary
from tqdm import tqdm
import torchmetrics.functional as tmf

class FBetaLoss(torch.nn.Module):
    def __init__(self, beta, epsilon=1e-6):
        super(FBetaLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, pred, target):
        """Computes the F-beta loss.

        Args:
            pred: The predicted output.
            target: The ground truth output.
            beta: The beta parameter.
            epsilon: A small value to avoid division by zero.

        Returns:
            The F-beta loss.
        """

        pred = Fn.sigmoid(pred)
        # Flatten the prediction and target tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute the intersection between the predicted output and the ground truth output
        intersection = (pred * target).sum()

        # Compute the precision
        precision = intersection / (pred.sum() + self.epsilon)

        # Compute the recall
        recall = intersection / (target.sum() + self.epsilon)

        # Compute the F-beta loss
        f_beta_loss = (1 + self.beta ** 2) * (precision * recall) / ((self.beta ** 2) * precision + recall + self.epsilon)

        # Return the F-beta loss
        return 1 - f_beta_loss
    
# Set up logging
def setup_logging(log_file='training.log'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    return logger

def save_model(model, optimizer, epoch, path):
    """ Save the model and optimizer state at a given path """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def log_model_architecture(model, input_size, logger):
    # Get model summary as a string
    model_summary = summary(model, input_size, verbose=0)
    logger.info(f"Model Architecture:\n{model_summary}")

def train(args, logger):
    # Initialize the UNet model
    model = UNet(n_classes=1, in_channels=1)  # Modify according to your needs
    model = model.to(args.device)

    # Log model architecture
    log_model_architecture(model, input_size=(1,1, 256, 256), logger=logger)  # Adjust input_size to match your input dimensions

    # Initialize the loss function
    criterion = FBetaLoss(beta=args.beta)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load datasets
    train_dataset = MSDataset(csv_file=args.train_csv, root_dir=args.root_dir, transform=get_train_transforms())
    val_dataset = MSDataset(csv_file=args.val_csv, root_dir=args.root_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", unit="batch")
        for images, masks in train_loader_tqdm:
            images, masks = images.to(args.device), masks.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            dice_score = tmf.dice(outputs.data, masks.int(), multiclass=False).item()
            

            running_dice += dice_score
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice_score:.4f}")

        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {avg_train_loss:.4f}, Training Dice: {avg_train_dice:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", unit="batch")

        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images, masks = images.to(args.device), masks.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                # Calculate dice score
                dice_score = tmf.dice(outputs.data, masks.int(), multiclass=False).item()
                val_dice += dice_score
                val_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice_score:.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.info("Validation loss improved. Resetting patience counter.")
            model_save_path = os.path.join(args.save_dir, f"checkpoint.pt")
            save_model(model, optimizer, epoch, model_save_path)
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UNet model")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the CSV file for the training data")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to the CSV file for the validation data")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing the images and masks")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--patience", type=int, default=15, help="Number of patience epochs for early stopping")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on (cuda or cpu)")
    parser.add_argument("--log_file", type=str, default="training.log", help="File to log training details")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta value for the F-beta loss function")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the best model weights")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)

    # Start training
    logger.info("Starting training...")
    train(args, logger)
    logger.info("Training finished.")