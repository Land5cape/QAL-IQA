# coding=utf-8
import numpy as np
import os
import sys

import pyiqa
import timm
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import inspect
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '\models')

# print(sys.path)

from models.builder import BuildAutoEncoder
from tools.utils import downsample, weigth_init

eps = 1e-8


class PretrainFidelityLoss(torch.nn.Module):

    def __init__(self):
        super(PretrainFidelityLoss, self).__init__()

    def forward(self, pred, n_gop):  # n_gop == n_aug + 1
        assert pred.shape[0] % n_gop == 0, 'pred.shape[0] % n_aug != 0'

        # print(pred.shape, label.shape)
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(1)

        label = np.tile(np.arange(n_gop, 0, -1), pred.shape[0] // n_gop)
        label = torch.Tensor(label).to(pred.device)
        label = label.unsqueeze(1)

        pred = pred - pred.t()
        label = label - label.t()

        triu_idx = torch.triu_indices(pred.size(0), pred.size(0), offset=1)

        pred = pred[triu_idx[0], triu_idx[1]]
        label = label[triu_idx[0], triu_idx[1]]

        g = 0.5 * (torch.sign(label) + 1)  # remove negative values

        constant = torch.sqrt(torch.Tensor([2.])).to(pred.device)
        p = 0.5 * (torch.erf(pred / constant) + 1)
        # torch.erf(x) is the error function that returns 2 * CDF(x) - 1

        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
        loss = torch.mean(loss)

        return loss


class FidelityLoss(torch.nn.Module):

    def __init__(self):
        super(FidelityLoss, self).__init__()

    def forward(self, pred, label):
        # print(pred.shape, label.shape)
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(1)
            label = label.unsqueeze(1)

        pred = pred - pred.t()
        label = label - label.t()

        triu_idx = torch.triu_indices(pred.size(0), pred.size(0), offset=1)

        pred = pred[triu_idx[0], triu_idx[1]]
        label = label[triu_idx[0], triu_idx[1]]

        g = 0.5 * (torch.sign(label) + 1)  # remove negative values

        constant = torch.sqrt(torch.Tensor([2.])).to(pred.device)
        p = 0.5 * (torch.erf(pred / constant) + 1)
        # torch.erf(x) is the error function that returns 2 * CDF(x) - 1

        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
        loss = torch.mean(loss)

        return loss


class RecoverBlsLoss(torch.nn.Module):
    def __init__(self):
        super(RecoverBlsLoss, self).__init__()

    def forward(self, y_rec, y_trs):
        l = torch.abs(torch.mean(y_rec ** 2 - 2 * y_trs))
        return l


# class RecoverBlsLoss(torch.nn.Module):
#     def __init__(self, loss_net):
#         super(RecoverBlsLoss, self).__init__()
#         self.loss_net = loss_net
#
#     def gram_matrix(self, x, normalize=True):
#
#         (b, ch, h, w) = x.size()
#         features = x.view(b, ch, h * w)
#         features_t = features.transpose(1, 2)
#         gram = features.bmm(features_t)  # b, ch, ch
#         if normalize:
#             gram /= ch * h * w
#         return gram
#
#     def forward(self, dst, rec):  # TODO from gram matrix to bls
#         # print(dst.shape, rec.shape)
#
#         return_layer = 'conv4'
#
#         x1 = dst
#         x2 = rec
#         for name, module in self.loss_net.named_children():
#             x1 = module(x1)
#             x2 = module(x2)
#             if name == return_layer:
#                 break
#
#         G1 = self.gram_matrix(x1)
#         G2 = self.gram_matrix(x2)
#
#         g1_norm = torch.linalg.norm(G1, dim=(1, 2))
#         g2_norm = torch.linalg.norm(G2, dim=(1, 2))
#
#         size = G1.size()
#         Nl = size[1] * size[2]  # Or C x C = C^2
#
#         normalize_term = (torch.square(g1_norm) + torch.square(g2_norm)) / Nl


class RecoverQualityLoss(torch.nn.Module):

    def __init__(self, quality_loss, best_quality=1):
        super(RecoverQualityLoss, self).__init__()
        self.best_quality = best_quality
        self.quality_loss = quality_loss

    def forward(self, q):
        # q: [b, 1]
        b = torch.full(q.shape, self.best_quality).to(torch.float).to(q.device)
        return self.quality_loss(q, b)


class CNN1x1Layer(nn.Module):

    def __init__(self, input_dim, output_dim, downsample=None):
        super(CNN1x1Layer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))
        if downsample:
            downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=downsample, stride=downsample,
                          padding=0),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True))
            self.layer = nn.Sequential(*self.layer, *downsample_layer)

    def forward(self, x):
        return self.layer(x)


class RegLayer(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(RegLayer, self).__init__()

        self.layer = nn.Sequential(
            # nn.Linear(in_features=input_dim, out_features=input_dim),
            # nn.BatchNorm1d(input_dim),
            # # nn.Dropout(p=0.5),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=input_dim, out_features=hid_dim),
            # nn.BatchNorm1d(hid_dim),
            # # nn.Dropout(p=0.5),
            # nn.Linear(in_features=hid_dim, out_features=1)
            nn.Linear(in_features=input_dim, out_features=1),
        )

    def forward(self, x):
        return self.layer(x)


class QualityPredictor(torch.nn.Module):  #
    def __init__(self, encoder_arch='vgg16', mid_f=False):
        super(QualityPredictor, self).__init__()
        # resnet: 2048 * 7 * 7
        # vgg: 512 * 7 * 7

        self.mid_f = mid_f
        self.avg_pool = None

        if not mid_f:
            if encoder_arch == 'vgg16':
                c_dims = 512
                hw_size = 7
                in_dims = c_dims * hw_size * hw_size
            elif encoder_arch == 'resnet50':
                c_dims = 2048
                in_dims = c_dims
                self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc = RegLayer(in_dims, 128)

        else:
            if encoder_arch == 'vgg16':
                # [-1, 64, 112, 112] [-1, 128, 56, 56] [-1, 256, 28, 28] [-1, 512, 14, 14] [-1, 512, 7, 7]
                dims = [64, 128, 256, 512, 512]
                dims_out = [64, 64, 64, 64, 64]

            elif encoder_arch == 'resnet50':
                # [-1, 64, 112, 112] [-1, 256, 56, 56] [-1, 512, 28, 28] [-1, 1024, 14, 14] [-1, 2048, 7, 7]
                dims = [64, 256, 512, 1024, 2048]
                dims_out = [64, 64, 64, 64, 64]
                dims_out = [64, 64, 64, 64, 64]
            self.conv1 = CNN1x1Layer(dims[0], dims_out[0], downsample=16)
            self.conv2 = CNN1x1Layer(dims[1], dims_out[1], downsample=8)
            self.conv3 = CNN1x1Layer(dims[2], dims_out[2], downsample=4)
            self.conv4 = CNN1x1Layer(dims[3], dims_out[3], downsample=2)
            self.conv5 = CNN1x1Layer(dims[4], dims_out[4])

            dims_sum = sum(dims_out)
            self.conv_last = CNN1x1Layer(dims_sum, dims_sum)
            self.conv_attent = nn.Sequential(
                nn.Conv2d(in_channels=dims_sum, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )
            self.avg_pool_last = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_last = RegLayer(dims_sum, 128)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward_last_layer(self, x):
        if self.avg_pool:
            x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_multi_layer(self, x):
        x1, x2, x3, x4, x5 = x
        # print(x5)
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        f3 = self.conv3(x3)
        f4 = self.conv4(x4)
        f5 = self.conv5(x5)
        fs = torch.cat([f1, f2, f3, f4, f5], dim=1)

        # f = self.conv_last(fs)
        f = self.avg_pool_last(fs)
        f = torch.flatten(f, 1)
        q = self.fc_last(f)

        # f = self.conv_last(fs)
        # w = self.conv_attent(fs)
        # q = (f*w).sum(dim=2).sum(dim=2)/w.sum(dim=2).sum(dim=2)

        return q

    def forward(self, x):
        if self.mid_f:
            x = self.forward_multi_layer(x)
        else:
            x = self.forward_last_layer(x)
        return x


class QualityPredictor2(torch.nn.Module):  #
    def __init__(self, backbone='resnet18'):
        super(QualityPredictor2, self).__init__()
        # resnet: 2048 * 7 * 7
        # vgg: 512 * 7 * 7

        self.backbone = timm.create_model('resnet18', pretrained=True).cuda()
        self.fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=128),
            nn.BatchNorm1d(128),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1)
        )

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.backbone(x)
        q = self.fc(q)
        return q


class AutoEncoder(torch.nn.Module):

    def __init__(self, arch='vgg16', mid_f=False, device=torch.device('cpu')):
        super(AutoEncoder, self).__init__()
        self.mid_f = mid_f
        self.ae = BuildAutoEncoder(arch=arch, mid_f=mid_f, device=device).module
        self.encoder = self.ae.encoder
        self.decoder = self.ae.decoder

    def forward(self, x):
        if self.mid_f:
            z, encoder_mid_f = self.encoder(x)
            x = self.decoder(z, encoder_mid_f)
            return x, encoder_mid_f
        else:
            z = self.encoder(x)
            x = self.decoder(z)
            return x


class DirectTransform(torch.nn.Module):

    def __init__(self, arch='vgg16', device=torch.device('cpu')):
        super(DirectTransform, self).__init__()
        self.auto_encoder = AutoEncoder(arch=arch, mid_f=False, device=device)

    def forward(self, x):
        return self.auto_encoder(x)


class PretrainDirectLoss(torch.nn.Module):

    def __init__(self):
        super(PretrainDirectLoss, self).__init__()

    def forward(self, y_trs, y_ref, y_rec):
        # print(y_trs.shape, x_chunk.shape, y_rec.shape)
        # l1 = nn.MSELoss()(y_ref, y_rec)
        # l2 = torch.mean(y_rec ** 2 - 2 * y_trs)
        # return nn.L1Loss()(l1, l2)
        return nn.MSELoss()(y_trs, torch.abs(y_rec - y_ref))


class Criterion():
    def __init__(self):
        # self.quality_loss = nn.L1Loss()
        # self.quality_loss = nn.MSELoss()
        self.quality_loss = FidelityLoss()
        self.recover_quality_loss = RecoverQualityLoss(self.quality_loss, best_quality=1)
        self.recover_bls_loss = RecoverBlsLoss()


class PretrainCriterion():  # pretrain_recover_loss 采用有监督的方式，然后固定参数
    def __init__(self):
        self.pretrain_quality_loss = PretrainFidelityLoss()
        self.pretrain_recover_loss = nn.L1Loss()
        self.pretrain_direct_loss = PretrainDirectLoss()


def get_models(arch='vgg16', mid_f=False, is_pretrain=False, model_load_path='', device=torch.device('cpu')):
    auto_encoder = AutoEncoder(arch=arch, mid_f=mid_f, device=device)
    encoder = auto_encoder.encoder
    decoder = auto_encoder.decoder

    quality_predictor = QualityPredictor(encoder_arch=arch, mid_f=mid_f)
    # quality_predictor = QualityPredictor2()
    # quality_predictor = pyiqa.create_metric('dbcnn', as_loss=False, device=device)
    # quality_predictor.train()

    direct_transform = DirectTransform(arch='vgg16')

    encoder.to(device)
    decoder.to(device)
    quality_predictor.to(device)
    direct_transform.to(device)

    if not is_pretrain:
        criterion = Criterion()
    else:
        criterion = PretrainCriterion()

    if model_load_path:
        model = torch.load(model_load_path)
        encoder.load_state_dict(model['encoder'])
        decoder.load_state_dict(model['decoder'])
        # quality_predictor.load_state_dict(model['quality_predictor'])
        direct_transform.load_state_dict(model['direct_transform'])

    return encoder, decoder, quality_predictor, direct_transform, criterion


# ==================================================
#  Test Part

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from tools.utils import prepare_image

    # initial = '1600.BLUR.5'
    # initial = '1600.JPEG.5.png'
    # reference = '1600.png'

    # initial = 'img1.bmp'
    # reference = 'monarch.bmp'

    # initial = 'geckos.fnoise.5.png'
    # reference = 'geckos.png'

    initial = 'snow_leaves.AWGN.5.png'
    reference = 'snow_leaves.png'

    dist = '../images/{}'.format(initial)
    ref = '../images/{}'.format(reference)

    device = torch.device('cuda')

    # ref = prepare_image(Image.open(ref).convert("RGB")).to(device)
    # dist = prepare_image(Image.open(dist).convert("RGB")).to(device)

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dst = Image.open(dist).convert('RGB')
    dst = val_trans(dst).unsqueeze(0)
    dst = dst.to(device)

    model_load_path = 'D:\swh\workplace\\trying4-new\\model-save_pretrain\\KADIS_2024_03_30_15_28_03\\final_model.pth'
    encoder, decoder, quality_predictor, direct_transform, criterion = get_models(arch='resnet50', mid_f=True,
                                                                                  model_load_path=model_load_path,
                                                                                  device=device)
    encoder.eval()
    decoder.eval()
    quality_predictor.eval()
    direct_transform.eval()

    # from torchsummary import summary  # torch-summary torchinfo
    # summary(encoder, (3, 224, 224), depth=5)

    for name, module in encoder.named_children():
        print(name)

    if encoder.mid_f:
        z_dst, encoder_mid_f = encoder(dst)
        q_dst = quality_predictor(encoder_mid_f)  # l1(q_dst, label)
    else:
        z_dst = encoder(dst)
        q_dst = quality_predictor(z_dst)

    if encoder.mid_f:
        y_rec = decoder(z_dst, encoder_mid_f)
    else:
        y_rec = decoder(z_dst)

    if encoder.mid_f:
        z_rec, encoder_mid_f = encoder(y_rec)
        q_rec = quality_predictor(encoder_mid_f)  # l2(q_rec)
    else:
        z_rec = encoder(y_rec)
        q_rec = quality_predictor(z_rec)

    print(q_dst, q_rec)

    y_trs = direct_transform(dst)
    print(y_trs)

    ##########
    # model = AutoEncoder(device=device)
    # model.to(device)
    # model.eval()
    # output = model(img)
    ##########

    ori_img = transforms.ToPILImage()(dst.squeeze().cpu())
    pred_img = transforms.ToPILImage()(y_rec.squeeze().cpu())

    trans_img = transforms.ToPILImage()(y_trs.squeeze().cpu())

    fig = plt.figure(figsize=(4, 1.5), dpi=300)

    plt.subplot(131)
    plt.imshow(ori_img)
    plt.title('ori_img', fontsize=6)
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(pred_img)
    plt.title('pred_img', fontsize=6)
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(trans_img)
    plt.title('trans_img', fontsize=6)
    plt.axis('off')

    plt.show()
    plt.pause(3)

    # 保存 pred_img
    pred_img.save('pred_img.png')

    # from torchsummary import summary  # torch-summary torchinfo
    # summary(encoder, (3, 224, 224), depth=5)

    # from thop import profile
    # macs, params = profile(model, inputs=(frames.unsqueeze(0)), verbose=False)
    # print(f"macs = {macs / 1e9}G")
    # print(f"params = {params / 1e6}M")
