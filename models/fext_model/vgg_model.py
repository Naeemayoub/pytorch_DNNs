import torch
import torch.nn as nn

class Vgg_Net(nn.Module):
    def __init__(self, in_ch, n_classes, vgg_model):
        super(Vgg_Net, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.vgg_block = self.conv_vgg(vgg_model)
        self.fcl = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes),
        )
    def forward(self, x):
        x = self.vgg_block(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcl(x)
        return x

    def conv_vgg(self, arch):
        layers = []
        in_ch = self.in_ch
        for x in arch:
            if type(x) == int:
                features = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=features,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(x)
                ]
                in_ch = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)
