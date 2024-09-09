import torch
import torch.nn as nn


def get_configs(arch='vgg16'):
    if arch == 'vgg11':
        configs = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        configs = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        configs = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        configs = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")

    return configs


class VGGAutoEncoder(nn.Module):

    def __init__(self, configs, mid_f=False):
        super(VGGAutoEncoder, self).__init__()
        self.mid_f = mid_f
        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = VGGEncoder(configs=configs, enable_bn=True, mid_f=mid_f)
        self.decoder = VGGDecoder(configs=configs[::-1], enable_bn=True, mid_f=mid_f)

    def forward(self, x):
        if self.mid_f:
            z, encoder_mid_f = self.encoder(x)
            x = self.decoder(z, encoder_mid_f)
            return x, encoder_mid_f
        else:
            z = self.encoder(x)
            x = self.decoder(z)
            return x


class VGG(nn.Module):

    def __init__(self, configs, num_classes=1000, img_size=224, enable_bn=False):
        super(VGG, self).__init__()

        self.encoder = VGGEncoder(configs=configs, enable_bn=enable_bn)

        self.img_size = img_size / 32

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(self.img_size * self.img_size * 512), out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.encoder(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class VGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False, mid_f=False):
        super(VGGEncoder, self).__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")
        self.mid_f = mid_f

        self.conv1 = EncoderBlock(input_dim=3, output_dim=64, hidden_dim=64, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        if self.mid_f:
            out = [x1, x2, x3, x4, x5]
            return x5, out
        else:
            return x5


class VGGDecoder(nn.Module):

    def __init__(self, configs, enable_bn=False, mid_f=False):
        super(VGGDecoder, self).__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=64, output_dim=3, hidden_dim=64, layers=configs[4], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()

        self.mid_f = mid_f

    def forward_once_midf(self, x, encoder_mid_f):

        [ex1, ex2, ex3, ex4, ex5] = encoder_mid_f

        x1 = self.conv1(x)
        x1 = x1 + ex4
        x2 = self.conv2(x1)
        x2 = x2 + ex3
        x3 = self.conv3(x2)
        x3 = x3 + ex2
        x4 = self.conv4(x3)
        x4 = x4 + ex1
        x5 = self.conv5(x4)
        x6 = self.gate(x5)

        return x6

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

    def forward(self, x, encoder_mid_f=None):
        if self.mid_f:
            x = self.forward_once_midf(x, encoder_mid_f)
        else:
            x = self.forward_once(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d DecoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)


class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):

        return self.layer(x)


if __name__ == "__main__":
    configs = get_configs("vgg16")
    input = torch.randn((5, 3, 224, 224))

    model = VGGAutoEncoder(configs, mid_f=True)
    output = model(input)

    # vgg16:   512, 7, 7

    from torchsummary import summary

    summary(model, input, depth=3)

    # from thop import profile
    #
    # macs, params = profile(grids, inputs=(img, ref), verbose=False)
    # print(f"macs = {macs / 1e9}G")
    # print(f"params = {params / 1e6}M")
