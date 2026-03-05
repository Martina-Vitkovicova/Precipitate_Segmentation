import torch
import torch.nn.functional as funct


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        bce = funct.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

    def description(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"

    def __repr__(self):
        return self.description()


class FocalDiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha, self.gamma, self.smooth = alpha, gamma, smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce = funct.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (self.alpha * (1 - pt) ** self.gamma * bce).mean()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + self.smooth) / (
                probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        return 0.5 * (focal + dice.mean())

    def description(self):
        return f"FocalDiceLoss(alpha={self.alpha}, gamma={self.gamma}, smooth={self.smooth})"

    def __repr__(self):
        return self.description()


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.3, smooth=1e-6, bce_weight=0.):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.smooth, self.bce_weight = smooth, bce_weight

    def forward(self, logits, target):
        target = target.float()
        p = torch.sigmoid(logits)
        tp = (p * target).sum(dim=(1, 2, 3))
        fp = (p * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - p) * target).sum(dim=(1, 2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        ft = (1.0 - tversky) ** self.gamma
        bce = funct.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        return self.bce_weight * bce + (1.0 - self.bce_weight) * ft.mean()

    def description(self):
        return ("FocalTverskyLoss("
                f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
                f"smooth={self.smooth}, bce_weight={self.bce_weight})")

    def __repr__(self):
        return self.description()
