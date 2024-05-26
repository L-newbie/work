from easydict import EasyDict

cfg = EasyDict()

cfg.anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]

cfg.anchors_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

cfg.strides = [32, 16, 8]

cfg.is_train = True