from .fbeta_loss import FBetaLoss
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss 


def get_loss_function(loss_name):
    if loss_name == 'f0.5':
        loss = FBetaLoss(beta=0.5)
    elif loss_name == 'f1':
        loss = FBetaLoss(beta=1.0)
    elif loss_name == 'f2':
        loss = FBetaLoss(beta=2.0)
    elif loss_name == 'dice' or loss_name == 'Dice' :
        loss = DiceLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    elif loss_name == 'DiceFocalLoss' or loss_name == "dfl":
        loss = DiceFocalLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    elif loss_name == 'DiceCELoss' or loss_name == "dce":
        loss = DiceCELoss(include_background=False, to_onehot_y=False, sigmoid=True)
    else:
        raise ValueError("Unsupported  loss")

    return loss