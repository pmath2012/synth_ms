import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import sys
from glasses.models.segmentation import unet
from monai.networks.nets import unetr, swin_unetr
sys.path.append("..")
from models.vision_transformer import ViTSeg
from loss.utils import get_loss_function
from datasets import get_synth_loader, get_train_loaders
from training import train_model, validate_model

def check_keys(model, pretrain_path):
    # Check for matching keys
    pretrain = torch.load(pretrain_path)
    pretrained_keys = set(pretrain.keys())
    model_keys = set(model.state_dict().keys())

    missing_keys = model_keys - pretrained_keys
    unexpected_keys = pretrained_keys - model_keys
    print(f"Loading : {pretrain_path}")
    if len(missing_keys) > 0:
        print("Missing keys in Siamese network:", missing_keys)
    else: 
        print("No Missing keys")

    if len(unexpected_keys) > 0:
        print("Unexpected keys in pre-trained UNet:", unexpected_keys)
    else:
        print("No unexpected keys")


def parse_args():
    parser = argparse.ArgumentParser(description='Train baselines')
    parser.add_argument('--model_name', type=str, default='unet', help='Model name to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained model')
    parser.add_argument('--pretrain_path', type=str, default="", help='Path to pre-trained model')
    parser.add_argument('--loss', type=str, choices=["dice", "dfl", "dce", 'f0.5', 'f1', 'f2'], default='f0.5', help='Mask loss function')
    parser.add_argument('--data_directory', type=str, default='', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--training_file', type=str, default='train.csv', help='Training file')
    parser.add_argument('--validation_file', type=str, default='valid.csv', help='Validation file')
    parser.add_argument('--pretraining', action='store_true', help="flag for pre-training with synthetic data")
    parser.add_argument('--method', type=str, help="method for training")
    parser.add_argument('--dataset', default="svuh", type=str, help="dataset for training, can be svuh or ms")
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    print('-'*100)
    print(f"\n\n\nTraining {args.model_name} \n\n\n")
    print('-'*100)

    learning_rate = args.learning_rate
    loss = get_loss_function(args.loss)
    epochs = args.epochs
    pretrained = args.pretrained
    pretrain_path = args.pretrain_path
    data_directory = args.data_directory
    train_file = args.training_file
    valid_file = args.validation_file
    batch_size = args.batch_size
    method = args.method
    ds = args.dataset
    data_directory = args.data_directory
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    
    # Initialize TensorBoard
    writer = SummaryWriter()

    if args.model_name == 'VIT_R18':
        model = ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet18', enc_depth=2, dec_depth=2)
    elif args.model_name == 'VIT_R50':
        model = ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet50', enc_depth=2, dec_depth=2)
    elif args.model_name == "unet" or args.model_name == "UNet":
        model = unet.UNet(in_channels=1, n_classes=1)
    elif args.model_name == "unetr":
        model = unetr.UNETR(in_channels=1, out_channels=1, img_size=256, spatial_dims=2, dropout_rate=0.1)
    elif args.model_name == "swin_unetr":
        model = swin_unetr.SwinUNETR(in_channels=1, out_channels=1, img_size=256, spatial_dims=2, drop_rate=0.1)
    else:
        raise ValueError("Unsupported model name")

    if pretrained:
        check_keys(model, pretrain_path)
        model.load_state_dict(torch.load(pretrain_path), strict=False)
    
    if args.pretraining:
        train_dataloader = get_synth_loader(train_file, data_directory, batch_size)
    else:
        train_dataloader, valid_dataloader = get_train_loaders(train_file, valid_file, data_directory, batch_size)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    ckpt=f"{args.model_name}_{args.loss}_epochs_{epochs}_{method}_{ds}.pth"
    
    # lists to keep track of losses and accuracies
    best_loss = 0
    best_epoch = 0

    # start the training
    model = model.to(device)

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_losses, train_epoch_acc, train_epoch_dice, train_epoch_f1 = train_model(model, train_dataloader,
                                                optimizer, loss ,device)
        if not args.pretraining:
            valid_epoch_losses, valid_epoch_acc, valid_epoch_dice, valid_epoch_f1 = validate_model(model, valid_dataloader,
                                                    loss, device)

        writer.add_scalar('Train/Loss', train_epoch_losses, epoch)
        writer.add_scalar('Train/Accuracy', train_epoch_acc, epoch)
        writer.add_scalar('Train/Dice', train_epoch_dice, epoch)
        writer.add_scalar('Train/F1', train_epoch_f1, epoch)
        print(f"Training : {train_epoch_losses:.3f}, training acc: {train_epoch_acc:.3f}, dice : {train_epoch_dice:.3f}, f1: {train_epoch_f1:.3f}")

        if args.pretraining:
            running_loss = train_epoch_losses
        else:
            running_loss = valid_epoch_losses
            writer.add_scalar('Validation/Loss', valid_epoch_losses, epoch)
            writer.add_scalar('Validation/Accuracy', valid_epoch_acc, epoch)
            writer.add_scalar('Validation/Dice', valid_epoch_dice, epoch)
            writer.add_scalar('Validation/F1', valid_epoch_f1, epoch)
            print(f"Validation : {valid_epoch_losses:.3f}, validation acc: {valid_epoch_acc:.3f}, dice : {valid_epoch_dice:.3f}, f1: {valid_epoch_f1:.3f}")
        
        if epoch == 0:
            print("saving first model")
            best_loss=running_loss
            best_epoch=epoch
            torch.save(model.state_dict(), ckpt)
        elif running_loss < best_loss:
            print(f"loss improved from {best_loss} to {running_loss} : saving best")
            best_loss=running_loss
            best_epoch=epoch
            torch.save(model.state_dict(), ckpt)
        else:
            print(f"loss did not improve from {best_loss} @ {best_epoch+1}")

        print('-'*100)
    writer.close()
