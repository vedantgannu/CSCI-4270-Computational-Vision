'''
Skeleton model class. You will have to implement the classification and regression layers, along with the forward definition.
'''


from torch import nn
from torchvision import models
import torch


class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()

        # Pretrained backbone. If you are on the cci machine then this will not be able to automatically download
        #  the pretrained weights. You will have to download them locally then copy them over.
        #  During the local download it should tell you where torch is downloading the weights to, then copy them to 
        #  ~/.cache/torch/checkpoints/ on the supercomputer.
        resnet = models.resnet18(pretrained=True)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone weights. 
        for param in self.backbone.parameters():
            param.requires_grad = False

        # TODO: Implement the fully connected layers for classification and regression.
        self.classification_block = nn.Sequential(
            nn.Flatten(),#Flatten everything except for the batch
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),#index 0 is the "nothing" class
        )
        #4C classes means 16 regression values excluding the "nothing" class
        self.regression_block = nn.Sequential(
            nn.Flatten(),#Flatten everything except for the batch
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 16)
        )

    def forward(self, x):
        # TODO: Implement forward. Should return a (batch_size x num_classes) tensor for classification
        #           and a (batch_size x num_classes x 4) tensor for the bounding box regression.    
        logits_backbone = self.backbone(x)
        logits_classification = self.classification_block(logits_backbone)#Batch size x 5
        logits_regression = self.regression_block(logits_backbone)#Batch size x 16
        return logits_classification, logits_regression
