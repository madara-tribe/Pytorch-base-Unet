import torch 
import torch.nn as nn

class MetaLoss():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        
    def rgb_loss(self, y_true, y_pred):
        loss = self.criterion(y_true, y_pred)
        return loss

    def dice_loss(self, y_true, y_pred):
        smooth = 1.
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

    def bce_dice_loss(self, y_true, y_pred):
        return self.bce_loss(y_true, y_pred) + (1 - self.dice_loss(y_true, y_pred))


import torch 
import torch.nn as nn

class MetaLoss():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        
    def rgb_loss(self, y_true, y_pred):
        loss = self.criterion(y_true, y_pred)
        return loss

    def dice_loss(self, y_true, y_pred):
        smooth = 1.
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

    def bce_dice_loss(self, y_true, y_pred):
        return self.bce_loss(y_true, y_pred) + (1 - self.dice_loss(y_true, y_pred))
