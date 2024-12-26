import torch
import torch.nn.functional as Fn

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