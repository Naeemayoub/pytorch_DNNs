import torch
from models.fext_model.vgg_model import Vgg_Net
from config.vgg_config import cfg
from pytorch_model_summary import summary
vgg_layers = cfg.vgg.vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Vgg_Net(in_ch=3, n_classes=2, vgg_model=vgg_layers).to(device)
x = torch.randn(1, 3, 224, 224)
print(summary(model, x))