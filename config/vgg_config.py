from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.vgg = edict()

__C.vgg.vgg16 = [64, 64, "M",  128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ]
__C.vgg.vgg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', ]