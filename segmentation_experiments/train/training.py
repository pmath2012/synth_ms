import torch
import numpy as np
from torchmetrics.functional import dice
from torchmetrics.functional.classification import accuracy, binary_f1_score
from tqdm import tqdm

def inner_loop(model, data, criterion, optimizer, device='cpu', train=True):
    running_metrics = {'loss':0, 'accuracy': 0, 'f1': 0, 'dice': 0}
    image, labels = data['image'], data['mask']
    image = image.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if train:
        optimizer.zero_grad()
    # forward pass
    outputs_m = model(image)
    loss = criterion(outputs_m, labels)
    running_metrics['loss'] = loss.item()
    running_metrics['accuracy'] = accuracy(outputs_m.data, labels, task='binary').cpu().numpy()
    running_metrics['f1'] = binary_f1_score(outputs_m.data, labels).cpu().numpy()
    running_metrics['dice'] = dice(outputs_m.data, labels).cpu().numpy()
    if train:
        loss.backward()
        optimizer.step()
    return running_metrics


def train_model(model, trainloader, optimizer, criterion, device='cpu'):
    model.train()
    print('Training')
    epoch_metrics = {'loss':[], 'accuracy': [], 'f1': [], 'dice': []}
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        running_metrics = inner_loop(model, data, criterion, optimizer, device)
        for key in running_metrics.keys():
            epoch_metrics[key].append(running_metrics[key])
    
    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])

    return epoch_loss, epoch_acc, epoch_dice, epoch_f1

# validation
def validate_model(model, testloader, criterion, device='cpu'):
    model.eval()
    print('Validation')
    epoch_metrics = {'loss':[], 'accuracy': [], 'f1': [], 'dice': []}
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            running_metrics = inner_loop(model, data, criterion, None, device, train=False)
            for key in running_metrics.keys():
                epoch_metrics[key].append(running_metrics[key])

    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])
    return epoch_loss, epoch_acc, epoch_dice, epoch_f1
