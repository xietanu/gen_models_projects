import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding="same", dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding="same", dilation=dilation)

    def forward(self, x):
        return self.conv2(self.relu(self.bn1(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1, 2, 4), latent_size=20):
        super().__init__()
        self.chs = chs
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(chs[-1] * 7 * 7, latent_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
        x = self.linear(x.reshape((-1, self.chs[-1] * 7 * 7)))
        x = self.sigmoid(x) * 2 - 1
        return x


class Decoder(nn.Module):
    def __init__(self, chs=(4, 2, 1), latent_size=20):
        super().__init__()
        self.chs = chs
        self.linear = nn.Linear(latent_size, chs[0] * 7 * 7)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x):
        x = self.linear(x).reshape((-1, self.chs[0], 7, 7))
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = self.dec_blocks[i](x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, enc_chs=(1, 2, 4), dec_chs=(4, 2, 1), latent_size=20, num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs, latent_size=latent_size)
        self.decoder = Decoder(dec_chs, latent_size=latent_size)
        self.head = nn.Conv2d(dec_chs[-1], num_class, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # run the model
        ftrs = self.encoder(x)
        out = self.decoder(ftrs)
        out = self.head(out)

        return self.sigmoid(out) * 2 - 1

class Classifier(nn.Module):
    def __init__(self, enc_chs=(1, 2, 4), latent_size=20, num_class=10):
        super().__init__()
        self.encoder = Encoder(enc_chs, latent_size)
        self.classifier_head = nn.Linear(latent_size, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.classifier_head(self.encoder(x))
        
    def classify(self, x):
        with torch.no_grad():
            return np.argmax(self.softmax(self(x)).cpu().numpy(), axis=1)
        