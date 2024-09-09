# coding:utf8

import csv
import random
import re

import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset
from tools.utils import MOS_label, read_split, write_split

# all_dataset_dir_path = 'D:/swh/dataset'
all_dataset_dir_path = 'E:\workplace\IVQAdataset\IQAdataset'

class Preprocesser():
    '''
    加载数据集
    划分数据集 根据参考图片划分
    '''

    def __init__(self, dataset, split_path='', preprocess=False, devide=3):

        print("\033[1;34m dataset: ", dataset, "\033[0m")

        assert dataset in ['KADID', 'CSIQ', 'LIVE', 'TID2008', 'TID2013', 'PIPAL'], 'unknown dataset'
        assert devide in [1, 2, 3]

        self.dataset_name = dataset

        self.dataset = None
        if dataset == 'KADID':
            self.dataset = KADID()
        elif dataset == 'CSIQ':
            self.dataset = CSIQ()
        elif dataset == 'LIVE':
            self.dataset = LIVE()
        elif dataset == 'TID2008':
            self.dataset = TID2008()
        elif dataset == 'TID2013':
            self.dataset = TID2013()
        elif dataset == 'PIPAL':
            self.dataset = PIPAL()

        if dataset == 'PIPAL':
            self.train_imgs = list(self.dataset.train_imgs)
            self.val_imgs = list(self.dataset.val_imgs)
            self.test_imgs = list(self.dataset.test_imgs)
        else:
            img_names = self.dataset.get_imgs()
            self.img_names = list(img_names)
            self.ref_names = self.dataset.get_refs()

            # print(len(self.img_names), self.img_names)
            # print(len(self.ref_names), self.ref_names)

            if preprocess:
                if devide == 1:
                    ratio = [sum(self.dataset.ratio), 0, 0]
                elif devide == 2:
                    ratio = [sum(self.dataset.ratio[0:2]), self.dataset.ratio[2], 0]
                else:
                    ratio = self.dataset.ratio
                train_refs, val_refs, test_refs = self.data_split(self.ref_names, ratio, shuffle=True)
                # print(len(train_refs), train_refs)
                # print(len(val_refs), val_refs)
                # print(len(test_refs), test_refs)

                import itertools
                self.train_imgs = list(
                    itertools.chain.from_iterable([self.dataset.get_dsts_by_ref(r) for r in train_refs]))
                self.val_imgs = list(itertools.chain.from_iterable([self.dataset.get_dsts_by_ref(r) for r in val_refs]))
                self.test_imgs = list(
                    itertools.chain.from_iterable([self.dataset.get_dsts_by_ref(r) for r in test_refs]))
                # print(len(self.train_imgs), self.train_imgs)
                # print(len(self.val_imgs), self.val_imgs)
                # print(len(self.test_imgs), self.test_imgs)

                self.write_infos(split_path)
                self.split_dict = read_split(split_path)
            else:
                self.split_dict = read_split(split_path)
                self.train_imgs, self.val_imgs, self.test_imgs = [], [], []
                for img in self.img_names:
                    if self.split_dict[img] == 'train':
                        self.train_imgs.append(img)
                    elif self.split_dict[img] == 'val':
                        self.val_imgs.append(img)
                    else:
                        self.test_imgs.append(img)

                if devide == 1:
                    self.train_imgs.extend(self.val_imgs)
                    self.train_imgs.extend(self.test_imgs)
                    self.val_imgs = []
                    self.test_imgs = []
                elif devide == 2:
                    self.train_imgs.extend(self.val_imgs)
                    self.val_imgs = self.test_imgs
                    self.test_imgs = []

    def data_split(self, full_list, ratio, shuffle=True):

        offset1 = ratio[0]
        offset2 = ratio[0] + ratio[1]

        if offset1 <= 1 and offset2 <= 1:
            if sum(ratio) != 1:
                raise Exception('ratio sum must be 1')
            offset1 = int(offset1 * len(full_list))
            offset2 = int(offset2 * len(full_list))
        else:
            if sum(ratio) != len(full_list):
                raise Exception('ratio sum must be len(full_list)')
            offset1 = int(offset1)
            offset2 = int(offset2)

        if shuffle:
            import random
            random.shuffle(full_list)

        l1 = full_list[:offset1]
        l2 = full_list[offset1:offset2]
        l3 = full_list[offset2:]
        return l1, l2, l3

    def write_infos(self, path):
        infos = []
        for img in self.train_imgs:
            infos.append([img, 'train'])
        for img in self.val_imgs:
            infos.append([img, 'val'])
        for img in self.test_imgs:
            infos.append([img, 'test'])
        write_split(path, infos, 'w')


class MyDataset(Dataset):

    def __init__(self, dataset, imgs, random_crop=False, flip=False):
        super(MyDataset, self).__init__()

        self.dataset = dataset
        self.img_names = imgs

        self.random = False
        if random_crop or flip:
            self.random = True

        forms = [
        ]
        if random_crop:
            forms.append(transforms.RandomCrop(224))
        else:
            forms.append(transforms.Resize([224, 224]))
            # transforms.Resize(224),
            # forms.append(transforms.CenterCrop(224))
        if flip:
            forms.append(transforms.RandomHorizontalFlip())

        forms.append(transforms.ToTensor())
        forms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(forms)

    def __getitem__(self, index):
        file_name = self.img_names[index]
        file_path = self.dataset.get_imgpath(file_name)
        ref_path = self.dataset.get_refpath(file_name)
        label = self.dataset.get_label(file_name)

        if self.random:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
        dst_image = Image.open(file_path).convert('RGB')
        file_data = self.transform(dst_image)
        if self.random:
            torch.random.manual_seed(seed)
        ref_image = Image.open(ref_path).convert('RGB')
        ref_data = self.transform(ref_image)

        sample = {
            'file_path': file_path,
            'file_name': file_name,
            'file_data': file_data,
            'ref_data': ref_data,
            'label': label
        }

        return sample

    def __len__(self):
        return len(self.img_names)


class CSIQ():
    '''
    CSIQ: 30  866  6  512x512  [0, 1]
    数据集，根据name获得path和label
    DMOS 越高越差  读取 MMOS
    '''

    def __init__(self, normalize=True):
        self.dataset_type = 'FR'
        dataset_dir_path = all_dataset_dir_path + '/FR/CSIQ'
        self.dst_img_dir = dataset_dir_path + '/dst_imgs'
        self.ref_img_dir = dataset_dir_path + '/src_imgs'
        self.label_path = dataset_dir_path + '/CSIQdmos.csv'
        self.mos_range = [0.0, 1.0]
        # self.mos_range = self.get_mos_range()
        self.ratio = [20, 5, 5]
        self.lower_better = False

        self.data = self.read_data()

    def get_mos_range(self):
        l = []
        with open(self.label_path, 'r') as f:
            reader = csv.reader(f)
            head_row = next(reader)
            for row in reader:
                dmos = float(row[1])
                l.append(dmos)
        print(min(l), max(l))
        return min(l), max(l)

    def read_data(self):
        data = {}
        with open(self.label_path, 'r') as f:
            reader = csv.reader(f)
            head_row = next(reader)
            for row in reader:
                img_name, mos = row[0], row[2]  # 1600.AWGN.1.png	0.062	0.938
                if self.lower_better:
                    label = 1 - MOS_label(float(mos), self.mos_range)
                else:
                    label = MOS_label(float(mos), self.mos_range)
                reference, noise = re.split('\.', img_name)[0:2]
                data[img_name] = [label, '{}.png'.format(reference), noise]
        return data

    def get_imgs(self):
        return self.data.keys()

    def get_imgpath(self, img_name):
        return '{}/{}'.format(self.dst_img_dir, img_name)

    def get_refs(self):
        return list(set([self.get_ref(img) for img in self.get_imgs()]))

    def get_refpath(self, img_name):
        ref_name = self.get_ref(img_name)
        return '{}/{}'.format(self.ref_img_dir, ref_name)

    def get_dsts_by_ref(self, ref_name):
        imgs = self.get_imgs()
        return list(filter(lambda k: self.get_ref(k) == ref_name, imgs))

    def get_label(self, img_name):
        return self.data[img_name][0]

    def get_ref(self, img_name):
        return self.data[img_name][1]

    def get_dst(self, img_name):
        return self.data[img_name][2]


class LIVE():
    '''
    数据集，根据name获得path和label
    LIVE: 29  779  5  -  [0, 100]
    DMOS 越高越差   在label处被1减
    '''

    def __init__(self, normalize=True):
        self.dataset_type = 'FR'
        dataset_dir_path = all_dataset_dir_path + '/FR/LIVE Image Quality Assessment Database'

        self.dst_img_dirs = [
            dataset_dir_path + '/jp2k',
            dataset_dir_path + '/jpeg',
            dataset_dir_path + '/wn',
            dataset_dir_path + '/gblur',
            dataset_dir_path + '/fastfading'
        ]

        self.ref_img_dir = dataset_dir_path + '/refimgs'
        self.label_path = dataset_dir_path + '/dmos.csv'
        self.org_path = dataset_dir_path + '/orgs.csv'
        self.refnames_path = dataset_dir_path + '/refnames_all.csv'
        self.mos_range = [0.0, 84.4889715362161]
        # self.mos_range = self.get_mos_range()
        self.ratio = [17, 6, 6]
        self.lower_better = True

        with open(self.label_path, 'r') as f:
            reader = csv.reader(f)
            dmos = next(reader)
            mos = [float(x) for x in dmos]
            # dmos=[dmos_jpeg2000(1:227) dmos_jpeg(1:233) white_noise(1:174) gaussian_blur(1:174) fast_fading(1:174)]
            # The values of dmos when corresponding orgs==1 are zero (they are reference images)

        with open(self.org_path, 'r') as f:
            reader = csv.reader(f)
            orgs = next(reader)
            # If orgs(i)==0, then this is a valid dmos entry. Else if orgs(i)==1 then image i denotes a copy of the reference image

        with open(self.refnames_path, 'r') as f:
            reader = csv.reader(f)
            refnames = next(reader)  # buildings.bmp

        all_img_names = []
        all_img_paths = []
        dst = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
        all_dsts = []
        for i, dst_dir in enumerate(self.dst_img_dirs):
            temp_list1 = []
            temp_list2 = []
            for x in os.scandir(dst_dir):
                if x.name.endswith('.bmp'):
                    all_dsts.append(dst[i])
                    temp_list1.append('{}_{}'.format(dst[i], x.name))
                    temp_list2.append(x.path)
            temp_list1.sort(key=lambda x: int(re.findall('\d+', x)[-1]))  # aaa_imgxxx.bmp
            temp_list2.sort(key=lambda x: int(re.findall('\d+', x)[-1]))  # imgxxx.bmp

            all_img_names.extend(temp_list1)
            all_img_paths.extend(temp_list2)

        # print(all_img_names)
        # print(all_img_paths)

        data = {}
        for i in range(len(mos)):
            if orgs[i] == '0':  # 是失真图像
                name = all_img_names[i]
                if self.lower_better:
                    label = 1 - MOS_label(mos[i], self.mos_range)
                else:
                    label = MOS_label(mos[i], self.mos_range)
                reference = refnames[i]
                noise = all_dsts[i]
                path = all_img_paths[i]
                data[name] = [label, reference, noise, path]
        # print(data)
        self.data = data

    def get_mos_range(self):
        with open(self.label_path, 'r') as f:
            reader = csv.reader(f)
            dmos = next(reader)
            mos = [float(x) for x in dmos]
        print(min(mos), max(mos))
        return min(mos), max(mos)

    def get_imgs(self):
        return self.data.keys()

    def get_imgpath(self, img_name):
        return self.data[img_name][3]

    def get_refs(self):
        return list(set([self.get_ref(img) for img in self.get_imgs()]))

    def get_refpath(self, img_name):
        ref_name = self.get_ref(img_name)
        return '{}/{}'.format(self.ref_img_dir, ref_name)

    def get_dsts_by_ref(self, ref_name):
        imgs = self.get_imgs()
        return list(filter(lambda k: self.get_ref(k) == ref_name, imgs))

    def get_label(self, img_name):
        return self.data[img_name][0]

    def get_ref(self, img_name):
        return self.data[img_name][1]

    def get_dst(self, img_name):
        return self.data[img_name][2]


class TID2008():
    '''
    数据集，根据name获得path和label
    TID2013: 25  1,700   17  512x384  [0, 9]
    MOS 越高越好
    '''

    def __init__(self, normalize=True):
        self.dataset_type = 'FR'
        dataset_dir_path = all_dataset_dir_path + '/FR/tid2008'

        self.dst_img_dir = dataset_dir_path + '/distorted_images'
        self.ref_img_dir = dataset_dir_path + '/reference_images'

        self.label_path = dataset_dir_path + '/mos_with_names.txt'

        self.ratio = [15, 5, 5]
        self.lower_better = False

        # self.mos_range = self.get_mos_range()
        self.mos_range = [0.0, 7.7143]

        self.data = self.read_data()

    def get_mos_range(self):
        l = []
        with open(self.label_path, encoding='utf-8') as f:
            content = f.read().splitlines()
            for line in content:
                mos = float(line.split(' ')[0])
                l.append(mos)
        print(min(l), max(l))
        return min(l), max(l)

    def read_data(self):
        data = {}
        with open(self.label_path, encoding='utf-8') as f:
            content = f.read().splitlines()
            for line in content:
                mos, img_name = line.split(' ')
                reference, noise = re.split('_', img_name)[0:2]
                # print(img_name, reference, noise)
                num = re.findall("\d+", reference)[-1]
                if num == '25':
                    reference = 'i{}.bmp'.format(num)
                else:
                    reference = 'I{}.bmp'.format(num)
                label = MOS_label(float(mos), self.mos_range)
                data[img_name] = [label, reference, noise]
        return data

    def get_imgs(self):
        return self.data.keys()

    def get_imgpath(self, img_name):
        return '{}/{}'.format(self.dst_img_dir, img_name)

    def get_refs(self):
        return list(set([self.get_ref(img) for img in self.get_imgs()]))

    def get_refpath(self, img_name):
        ref_name = self.get_ref(img_name)
        return '{}/{}'.format(self.ref_img_dir, ref_name)

    def get_dsts_by_ref(self, ref_name):
        imgs = self.get_imgs()
        return list(filter(lambda k: self.get_ref(k) == ref_name, imgs))

    def get_label(self, img_name):
        return self.data[img_name][0]

    def get_ref(self, img_name):
        return self.data[img_name][1]

    def get_dst(self, img_name):
        return self.data[img_name][2]


class TID2013():
    '''
    数据集，根据name获得path和label
    TID2013: 25  3,000   24  512x384  [0, 9]
    MOS 越高越好
    '''

    def __init__(self, normalize=True):
        self.dataset_type = 'FR'
        dataset_dir_path = all_dataset_dir_path + '/FR/tid2013'

        self.dst_img_dir = dataset_dir_path + '/distorted_images'
        self.ref_img_dir = dataset_dir_path + '/reference_images'

        self.label_path = dataset_dir_path + '/mos_with_names.txt'

        self.ratio = [15, 5, 5]
        self.lower_better = False

        # self.mos_range = self.get_mos_range()
        self.mos_range = [0.24242, 7.21429]

        self.data = self.read_data()

    def get_mos_range(self):
        l = []
        with open(self.label_path, encoding='utf-8') as f:
            content = f.read().splitlines()
            for line in content:
                mos = float(line.split(' ')[0])
                l.append(mos)
        print(min(l), max(l))
        return min(l), max(l)

    def read_data(self):
        data = {}
        with open(self.label_path, encoding='utf-8') as f:
            content = f.read().splitlines()
            for line in content:
                mos, img_name = line.split(' ')
                reference, noise = re.split('_', img_name)[0:2]
                # print(img_name, reference, noise)
                num = re.findall("\d+", reference)[-1]
                if num == '25':
                    reference = 'i{}.bmp'.format(num)
                else:
                    reference = 'I{}.bmp'.format(num)
                label = MOS_label(float(mos), self.mos_range)
                data[img_name] = [label, reference, noise]
        return data

    def get_imgs(self):
        return self.data.keys()

    def get_imgpath(self, img_name):
        return '{}/{}'.format(self.dst_img_dir, img_name)

    def get_refs(self):
        return list(set([self.get_ref(img) for img in self.get_imgs()]))

    def get_refpath(self, img_name):
        ref_name = self.get_ref(img_name)
        return '{}/{}'.format(self.ref_img_dir, ref_name)

    def get_dsts_by_ref(self, ref_name):
        imgs = self.get_imgs()
        return list(filter(lambda k: self.get_ref(k) == ref_name, imgs))

    def get_label(self, img_name):
        return self.data[img_name][0]

    def get_ref(self, img_name):
        return self.data[img_name][1]

    def get_dst(self, img_name):
        return self.data[img_name][2]


class PIPAL():
    '''
    数据集，根据name获得path和label
    PIPAL: 200  23,000   40  288x288  -
    EDL #
    '''

    def __init__(self):
        self.dataset_type = 'FR'
        dataset_dir_path = all_dataset_dir_path + '/FR/PIPAL'

        self.train_dst_img_dirs = [
            dataset_dir_path + '/Distortion_1',
            dataset_dir_path + '/Distortion_2',
            dataset_dir_path + '/Distortion_3',
            dataset_dir_path + '/Distortion_4',
        ]
        self.train_ref_img_dir = dataset_dir_path + '/Train_Ref'
        self.train_label_path = dataset_dir_path + '/Train_Label'

        self.val_dst_img_dir = dataset_dir_path + '/val/Dis'
        self.val_ref_img_dir = dataset_dir_path + '/val/Ref'
        self.val_label_path = dataset_dir_path + '/val/pipal21_val.txt'

        self.test_dst_img_dir = dataset_dir_path + '/test/Dis'
        self.test_ref_img_dir = dataset_dir_path + '/test/Ref'

        # self.mos_range = [916.7136, 1835.9856]  train
        self.mos_range = [868.2988, 1857.0]  # train+val
        # self.mos_range = self.get_mos_range()

        data_path = {}
        for i, dst_dir in enumerate(self.train_dst_img_dirs):
            for x in os.scandir(dst_dir):
                if x.name.endswith('.bmp'):
                    data_path[x.name] = x.path

        self.data = {}
        self.train_imgs = set()
        self.val_imgs = set()
        self.test_imgs = set()

        for x in os.scandir(self.train_label_path):
            with open(x.path, encoding='utf-8') as f:
                content = f.read().splitlines()
                for line in content:
                    img_name, mos = line.split(',')
                    reference = re.split('_', img_name)[0]
                    reference = '{}.bmp'.format(reference)
                    label = MOS_label(float(mos), self.mos_range)
                    self.data[img_name] = [label, reference, data_path[img_name]]
                    self.train_imgs.add(img_name)

        val_label = dict()
        with open(self.val_label_path, encoding='utf-8') as f:
            content = f.read().splitlines()
            for line in content:
                img_name, mos = line.split(',')
                label = MOS_label(float(mos), self.mos_range)
                val_label[img_name] = label

        for x in os.scandir(self.val_dst_img_dir):
            if x.name.endswith('.bmp'):
                reference = re.split('_', x.name)[0]
                reference = '{}.bmp'.format(reference)
                label = val_label[x.name]
                self.data[x.name] = [label, reference, x.path]
                self.val_imgs.add(x.name)

        for x in os.scandir(self.test_dst_img_dir):
            if x.name.endswith('.bmp'):
                reference = re.split('_', x.name)[0]
                reference = '{}.bmp'.format(reference)
                self.data[x.name] = [None, reference, x.path]
                self.test_imgs.add(x.name)

        self.train_ref_imgs = set()
        self.val_ref_imgs = set()
        self.test_ref_imgs = set()
        for x in os.scandir(self.train_ref_img_dir):
            if x.name.endswith('.bmp'):
                self.train_ref_imgs.add(x.name)
        for x in os.scandir(self.val_ref_img_dir):
            if x.name.endswith('.bmp'):
                self.val_ref_imgs.add(x.name)
        for x in os.scandir(self.test_ref_img_dir):
            if x.name.endswith('.bmp'):
                self.test_ref_imgs.add(x.name)

    def get_mos_range(self):
        l = []
        for x in os.scandir(self.train_label_path):
            with open(x.path, encoding='utf-8') as f:
                content = f.read().splitlines()
                for line in content:
                    mos = float(line.split(',')[1])
                    l.append(mos)

        with open(self.val_label_path, encoding='utf-8') as f:
            content = f.read().splitlines()
            for line in content:
                mos = float(line.split(',')[1])
                l.append(mos)

        print(min(l), max(l))
        return min(l), max(l)

    def get_imgs(self):
        return self.data.keys()

    def get_imgpath(self, img_name):
        return self.data[img_name][2]

    def get_refs(self):
        return list(set([self.get_ref(img) for img in self.get_imgs()]))

    def get_refpath(self, img_name):
        ref_name = self.get_ref(img_name)
        if ref_name in self.train_ref_imgs:
            ref_img_dir = self.train_ref_img_dir
        elif ref_name in self.val_ref_imgs:
            ref_img_dir = self.val_ref_img_dir
        elif ref_name in self.test_ref_imgs:
            ref_img_dir = self.test_ref_img_dir
        else:
            raise Exception('wrong img_name: {}'.format(ref_name))
        return '{}/{}'.format(ref_img_dir, ref_name)

    def get_label(self, img_name):
        return self.data[img_name][0]

    def get_ref(self, img_name):
        return self.data[img_name][1]

    def get_dst(self, img_name):
        return ''


class KADID():
    '''
    数据集，根据name获得path和label
    KADID: 81  10,125  25  512x384  [1, 5]
    dmos 越高越好
    '''

    def __init__(self, normalize=True):
        self.dataset_type = 'FR'

        dataset_dir_path = all_dataset_dir_path + '/FR/kadid10k'
        self.img_dir = dataset_dir_path + '/images'
        self.label_path = dataset_dir_path + '/image_labeled_by_per_noise.csv'
        self.mos_range = [1.0, 4.93]
        # self.mos_range = self.get_mos_range()
        self.ratio = [49, 16, 16]
        self.data = self.read_data()

    def get_mos_range(self):
        l = []
        with open(self.label_path, 'r') as f:
            reader = csv.reader(f)
            head_row = next(reader)
            for row in reader:
                dmos = float(row[1])
                l.append(dmos)
        print(min(l), max(l))
        return min(l), max(l)

    def read_data(self):
        data = {}
        with open(self.label_path, 'r') as f:
            reader = csv.reader(f)
            head_row = next(reader)
            for row in reader:
                img_name, dmos, reference, noise = row[0], row[1], row[2], row[3]  # I01_01_01.png  4.57  I01.png  1
                label = MOS_label(float(dmos), self.mos_range)
                data[img_name] = [label, reference, noise]
        return data

    def get_imgs(self):
        return self.data.keys()

    def get_imgpath(self, img_name):
        return '{}/{}'.format(self.img_dir, img_name)

    def get_refs(self):
        return list(set([self.get_ref(img) for img in self.get_imgs()]))

    def get_refpath(self, img_name):
        ref_name = self.get_ref(img_name)
        return '{}/{}'.format(self.img_dir, ref_name)

    def get_dsts_by_ref(self, ref_name):
        imgs = self.get_imgs()
        return list(filter(lambda k: self.get_ref(k) == ref_name, imgs))

    def get_label(self, img_name):
        return self.data[img_name][0]

    def get_ref(self, img_name):
        return self.data[img_name][1]

    def get_dst(self, img_name):
        return self.data[img_name][2]


class PretrainPreprocesser():
    '''
    加载数据集
    划分数据集 根据参考图片划分
    '''

    def __init__(self, dataset, split_path='', preprocess=False, devide=3):

        print("\033[1;34m dataset: ", dataset, "\033[0m")

        assert dataset in ['KADIS', 'AVA'], 'unknown dataset'
        assert devide in [1, 2, 3]

        self.dataset = None
        if dataset == 'KADIS':
            self.dataset = KADIS()
        elif dataset == 'AVA':
            self.dataset = AVA()

        self.img_names = self.dataset.get_imgs()

        if preprocess:
            if devide == 1:
                ratio = [sum(self.dataset.ratio), 0, 0]
            elif devide == 2:
                ratio = [sum(self.dataset.ratio[0:2]), self.dataset.ratio[2], 0]
            else:
                ratio = self.dataset.ratio
            self.train_imgs, self.val_imgs, self.test_imgs = self.data_split(self.img_names, ratio, shuffle=True)

            self.write_infos(split_path)
            self.split_dict = read_split(split_path)
        else:
            self.split_dict = read_split(split_path)
            self.train_imgs, self.val_imgs, self.test_imgs = [], [], []
            for img in self.img_names:
                if self.split_dict[img] == 'train':
                    self.train_imgs.append(img)
                elif self.split_dict[img] == 'val':
                    self.val_imgs.append(img)
                else:
                    self.test_imgs.append(img)

            if devide == 1:
                self.train_imgs.extend(self.val_imgs)
                self.train_imgs.extend(self.test_imgs)
                self.val_imgs = []
                self.test_imgs = []
            elif devide == 2:
                self.train_imgs.extend(self.val_imgs)
                self.val_imgs = self.test_imgs
                self.test_imgs = []

    def data_split(self, full_list, ratio, shuffle=True):

        offset1 = ratio[0]
        offset2 = ratio[0] + ratio[1]

        if offset1 <= 1 and offset2 <= 1:
            if sum(ratio) != 1:
                raise Exception('ratio sum must be 1')
            offset1 = int(offset1 * len(full_list))
            offset2 = int(offset2 * len(full_list))
        else:
            if sum(ratio) != len(full_list):
                raise Exception('ratio sum must be len(full_list)')
            offset1 = int(offset1)
            offset2 = int(offset2)

        if shuffle:
            import random
            random.shuffle(full_list)

        l1 = full_list[:offset1]
        l2 = full_list[offset1:offset2]
        l3 = full_list[offset2:]
        return l1, l2, l3

    def write_infos(self, path):
        infos = []
        for img in self.train_imgs:
            infos.append([img, 'train'])
        for img in self.val_imgs:
            infos.append([img, 'val'])
        for img in self.test_imgs:
            infos.append([img, 'test'])
        write_split(path, infos, 'w')

class MyPretrainDataset(Dataset):

    def __init__(self, dataset, imgs):
        super(MyPretrainDataset, self).__init__()

        self.dataset = dataset
        self.img_names = imgs

        forms = [
            transforms.Resize([224, 224]),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip()
        ]

        self.init_transform = transforms.ToTensor()
        self.transform = transforms.Compose(forms)

    def __getitem__(self, index):
        from modules.iqa_distortions import iqa_transformations

        file_name = self.img_names[index]
        file_path = self.dataset.get_imgpath(file_name)

        image = Image.open(file_path).convert('RGB')
        file_data = self.init_transform(image)

        chunk = file_data.unsqueeze(0)

        c = random.randint(1, 40)  # (27) more original img when pretrain
        l = random.randint(0, 4)
        # for i in range(0, 5):
        img_aug_i = self.init_transform(iqa_transformations(c, image, l))
        img_aug_i = img_aug_i.unsqueeze(0)
        chunk = torch.cat([chunk, img_aug_i], dim=0)

        chunk = self.transform(chunk)

        sample = {
            'file_path': file_path,
            'file_name': file_name,
            'file_data': chunk,
        }

        return sample

    def __len__(self):
        return len(self.img_names)


class KADIS():
    def __init__(self, normalize=True):
        self.dataset_type = 'Pretrain'
        dataset_dir_path = all_dataset_dir_path + '/Blind/kadis700k'
        self.ref_img_dir = dataset_dir_path + '/kadis700k/ref_imgs'
        self.dst_img_dir = dataset_dir_path + '/kadis700k/dist_imgs'
        self.ref_info_path = dataset_dir_path + '/kadis700k/kadis700k_ref_imgs.csv'
        self.ratio = [0.8, 0.1, 0.1]

        self.data = self.read_ref_data()

    def read_ref_data(self):
        data = {}
        with open(self.ref_info_path, 'r') as f:
            reader = csv.reader(f)
            head_row = next(reader)
            for row in reader:
                img_name, dist_type_1, dist_type_2 = row[0], row[1], row[2]  # a-fleshy-plant-1464418.png	7
                data[img_name] = [dist_type_1, dist_type_2]
        return data

    def get_imgs(self):
        return list(self.data.keys())

    def get_imgpath(self, img_name):
        return '{}/{}'.format(self.ref_img_dir, img_name)


class AVA():
    def __init__(self, normalize=True):
        self.dataset_type = 'Pretrain'
        dataset_dir_path = all_dataset_dir_path + '/Blind/AVA'
        self.img_dir = dataset_dir_path + '/images'
        self.info_path = dataset_dir_path + '/AVA.txt'
        self.ratio = [0.8, 0.1, 0.1]

        self.data = self.read_ref_data()

    def read_ref_data(self):
        dir_path = self.img_dir
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        files = set(files)

        data = {}
        with open(self.info_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_idx, img_name = line.split(' ')[:2]
                infos = line.split(' ')[2:]
                img_name = img_name + '.jpg'
                if img_name in files:
                    data[img_name] = infos
        return data

    def get_imgs(self):
        return list(self.data.keys())

    def get_imgpath(self, img_name):
        return '{}/{}'.format(self.img_dir, img_name)


if __name__ == '__main__':
    # dataset = 'KADIS'
    # prep = Preprocesser(dataset=dataset, split_path='{}_split.csv'.format(dataset), preprocess=False)
    dataset = AVA()
