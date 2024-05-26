import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True) -> None:
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 4 + 1 + num_classes
        self.img_size = img_size
        self.feature_length = [img_size[0]//8,img_size[0]//16,img_size[0]//32]
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.7
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.cuda = cuda

    def forward(self, x):
        return x