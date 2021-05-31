import torch
from models.fext_model.vgg_model import Vgg_Net
from config.vgg_config import cfg


vgg_layers = cfg.vgg.vgg19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Vgg_Net(in_ch=3, n_classes=2, vgg_model=vgg_layers).to(device)
x = torch.randn(1, 3, 224, 244)
print(model)