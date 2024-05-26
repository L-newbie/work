'''
Author: weidong.he
Date: 2024-05-25 11:20:24
LastEditTime: 2024-05-25 23:41:50
'''
import torch.nn as nn
import torch
from ..config import cfg


def yolo_decode(output, num_classes, anchors, num_anchors, scale_x_y=1):
    '''
        output: (B, A * n_ch, H, W)
    '''
    # [bx,by,bw,bh,obj,cls] -> [bx,by,bw,bh] [obj] [classes]
    n_ch = 4 + 1 + num_classes
    #几个头
    A = num_anchors
    #batch = bn
    B = output.size(0)
    # 图像height width
    H = output.size(2)
    W = output.size(3)
    #(B, A * n_ch, H, W) -> (B, A, n_ch, H, W)
    output = output.view(B, A, n_ch, H, W)
    #(B, A, n_ch, H, W) -> (B, A, H, W, n_ch)
    output = output.permute(0, 1, 3, 4, 2)
    #使内存是连续的
    output = output.contigous()
    #分别获取[bx,by,bw,bh,obj,cls]
    bx, by = output[..., 0], [..., 1]
    bw, bh = output[..., 2], [..., 3]
    obj, cls = output[..., 4], output[..., 5:]
    #使用sigmod映射到0-1之间变为概率值
    bx = torch.sigmoid(bx)
    by = torch.sigmoid(by)
    obj = torch.sigmoid(obj)
    cls = torch.sigmoid(cls)
    #进行指数计算、缩放因子、sigmod
    bw = torch.exp(bw) * scale_x_y - 0.5 * (scale_x_y - 1)
    bh = torch.exp(bh) * scale_x_y - 0.5 * (scale_x_y - 1)
    #构造[1, 3, 19*19, 1] [0,1...18]的数组
    grad_x = torch.arange(W, dtype=torch.float).repeat(1, 3, W, 1)
    grad_y = torch.arange(H, dtype=torch.float).repeat(1, 3, W, 1).permute(0, 1, 3, 2)
    bx += grad_x
    by += grad_y
    
    for i in range(num_anchors):
        bw[:, i, :, :] *= anchors[i*2]
        bh[:, i, :, :] *= anchors[i*2+1]

    bx = (bx / W).unsqueeze(-1)
    by = (by / H).unsqueeze(-1)
    bw = (bw / W).unsqueeze(-1)
    bh = (bh / H).unsqueeze(-1)
    boxes = torch.cat((bx, by, bw, bh), dim=-1).reshape(B, A * H * W, 4)
    obj = obj.unsqueeze(-1).reshape(B, A*H*W, 1)
    cls =cls.reshape(B, A*H*W, num_classes)
    outputs = torch.cat([boxes, obj, cls], dim=-1)
    return outputs



class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=80, anchors=[], num_anchors=9, stride=32, scale_x_y=1):
        super(YoloLayer, self).__init__()
        #[[6, 7, 8], [3, 4, 5], [0, 1, 2]] -> [6, 7, 8]
        self.anchor_mask = anchor_mask
        #类别
        self.num_classes = num_classes
        #[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        self.anchors = anchors
        #9
        self.num_anchors = num_anchors
        #18/9=2
        self.anchor_step = len(anchors) // num_anchors
        #32
        self.stride = stride
        #1
        self.scale_x_y = scale_x_y
    
    def forward(self, output):
        if cfg.is_train:
            return output
        masked_anchors =[]
        for m in self.anchor_mask:
            masked_anchors += self.anchors[ m * self.anchor_step: (m + 1) * self.anchor_step]
        #[142, 110, 192, 243, 459, 401] 将像素值换算成网格单位
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        #
        data = yolo_decode(output, self.num_classes, masked_anchors, len(self.anchor_mask), self.scale_x_y)
        return data