import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img, resize_img
import json
import cv2
import natsort

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args

        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'derain': 3, 'dehaze': 4, 'desnow': 5, 'derain_heavy': 6, 'dehaze_heavy': 7, 'desnow_heavy': 8}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'desnow' in self.de_type:
            self._init_snow_ids()

        if 'derain_heavy' in self.de_type:
            self._init_heavyrain_ids()
        if 'dehaze_heavy' in self.de_type:
            self._init_heavyhazy_ids()
        if 'desnow_heavy' in self.de_type:
            self._init_heavysnow_ids()

        random.shuffle(self.de_type)

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.derain_dir + 'derain_train_low.json'
        temp_ids += [self.args.derain_dir + id_['image_path'].strip() for id_ in json.load(open(rs))]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]

        self.num_rl = len(self.rs_ids)
        print("Total low Rainy Ids : {}".format(self.num_rl))

    def _init_heavyrain_ids(self):
        temp_ids = []
        rs = self.args.derain_dir + 'derain_train_high.json'
        temp_ids += [self.args.derain_dir + id_['image_path'].strip() for id_ in json.load(open(rs))]
        self.heavy_rs_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids]

        self.num_heavy_rl = len(self.heavy_rs_ids)
        print("Total heavy Rainy Ids : {}".format(self.num_heavy_rl))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.dehaze_dir + 'dehaze_reside_train_low.json'
        temp_ids += [self.args.dehaze_dir + id_['image_path'].strip() for id_ in json.load(open(hazy))]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.num_hazy = len(self.hazy_ids)
        print("Total Low Hazy Ids : {}".format(self.num_hazy))

    def _init_heavyhazy_ids(self):
        temp_ids = []
        hazy = self.args.dehaze_dir + 'dehaze_reside_train_high.json'
        temp_ids += [self.args.dehaze_dir + id_['image_path'].strip() for id_ in json.load(open(hazy))]
        self.heavy_hazy_ids = [{"clean_id": x, "de_type": 7} for x in temp_ids]

        self.num_heavy_hazy = len(self.heavy_hazy_ids)
        print("Total Heavy Hazy Ids : {}".format(self.num_heavy_hazy))

    def _init_snow_ids(self):
        temp_ids = []
        snow = self.args.desnow_dir + 'desnow_snow100_train_low.json'
        temp_ids += [self.args.desnow_dir + id_['image_path'].strip() for id_ in json.load(open(snow))]
        self.snow_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]

        self.num_snow = len(self.snow_ids)
        print("Total low Snow Ids : {}".format(self.num_snow))

    def _init_heavysnow_ids(self):
        temp_ids = []
        snow = self.args.desnow_dir + 'desnow_snow100_train_high.json'
        temp_ids += [self.args.desnow_dir + id_['image_path'].strip() for id_ in json.load(open(snow))]
        self.heavy_snow_ids = [{"clean_id": x, "de_type": 8} for x in temp_ids]

        self.num_heavy_snow = len(self.heavy_snow_ids)
        print("Total Heavy Snow Ids : {}".format(self.num_heavy_snow))


    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        patch = self.args.patch_size

        if H < patch or W < patch:
            new_h = max(H, patch)
            new_w = max(W, patch)

            img_1 = cv2.resize(img_1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            H, W = img_1.shape[:2]

        ind_H = random.randint(0, H - patch)
        ind_W = random.randint(0, W - patch)

        patch_1 = img_1[ind_H:ind_H + patch, ind_W:ind_W + patch]
        patch_2 = img_2[ind_H:ind_H + patch, ind_W:ind_W + patch]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.replace('input', 'target')
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        nonhazy_name = hazy_name.replace('hazy', 'clear')
        x = nonhazy_name.split('/')[-1]
        y = x.split('_')[0]
        ext = '.' + x.split('.')[-1]
        nonhazy_name = nonhazy_name.replace(x, y + ext)
        return nonhazy_name

    def _get_nonsnow_name(self, snow_name):
        nonsnow_name = snow_name.replace('data2', 'gt')
        return nonsnow_name

    def _merge_ids(self):
        self.sample_ids = []

        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids

        if "derain_heavy" in self.de_type:
            self.sample_ids += self.heavy_rs_ids
        if "dehaze_heavy" in self.de_type:
            self.sample_ids += self.heavy_hazy_ids
        if "desnow_heavy" in self.de_type:
            self.sample_ids += self.heavy_snow_ids
    def get_context(self, sample, de_id):
        if de_id == 3:
            list_of_values = self.rs_ids
        elif de_id == 4:
            list_of_values = self.hazy_ids
        elif de_id == 5:
            list_of_values = self.snow_ids
        elif de_id == 6:
            list_of_values = self.heavy_rs_ids
        elif de_id == 7:
            list_of_values = self.heavy_hazy_ids
        elif de_id == 8:
            list_of_values = self.heavy_snow_ids
        else:
            raise ValueError("Invalid degradation type encountered")

        second_value = sample
        while second_value == sample:
            second_value = random.choice(list_of_values)

        return second_value

    def context_transform(self, n_px):
        return Compose([
            ToPILImage(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        context_id = self.get_context(sample, de_id)

        assert context_id["de_type"] == de_id

        if de_id == 3 or de_id == 6:
            # Rain Streak Removal
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_name = self._get_gt_name(context_id["clean_id"])
            clean_context_img = crop_img(np.array(Image.open(clean_context_name).convert('RGB')), base=16)

        elif de_id == 4 or de_id == 7:
            # Dehazing with SOTS outdoor training set
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_nonhazy_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_name = self._get_nonhazy_name(context_id["clean_id"])
            clean_context_img = crop_img(np.array(Image.open(clean_context_name).convert('RGB')), base=16)

        elif de_id == 5 or de_id == 8:
            # Desnowing with Snow100 training set
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_nonsnow_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_name = self._get_nonsnow_name(context_id["clean_id"])
            clean_context_img = crop_img(np.array(Image.open(clean_context_name).convert('RGB')), base=16)

        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clip_transform = self.context_transform(224)
        degrad_context_patch, clean_context_patch = (clip_transform(degrad_context_img),
                                                     clip_transform(clean_context_img))

        clean_patch = self.toTensor(clean_patch).float()
        degrad_patch = self.toTensor(degrad_patch).float()


        return [clean_name, de_id], degrad_patch, clean_patch, degrad_context_patch, clean_context_patch


    def __len__(self):
        return len(self.sample_ids)


class ValDataset(Dataset):
    def __init__(self, args):
        super(ValDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        print("Setting up Validation dataset")
        print(self.de_type)

        self.de_dict = {'derain': 3, 'dehaze': 4, 'desnow': 5, 'derain_heavy': 6, 'dehaze_heavy': 7, 'desnow_heavy': 8}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'desnow' in self.de_type:
            self._init_snow_ids()

        if 'derain_heavy' in self.de_type:
            self._init_heavyrain_ids()
        if 'dehaze_heavy' in self.de_type:
            self._init_heavyhazy_ids()
        if 'desnow_heavy' in self.de_type:
            self._init_heavysnow_ids()

        random.shuffle(self.de_type)

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.derain_dir + 'derain_val_low.json'
        temp_ids += [self.args.derain_dir + id_['image_path'].strip() for id_ in json.load(open(rs))]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total low validation Rainy Ids : {}".format(self.num_rl))

    def _init_heavyrain_ids(self):
        temp_ids = []
        rs = self.args.derain_dir + 'derain_val_high.json'
        temp_ids += [self.args.derain_dir + id_['image_path'].strip() for id_ in json.load(open(rs))]
        self.heavy_rs_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids]

        self.num_heavy_rl = len(self.heavy_rs_ids)
        print("Total heavy validation Rainy Ids : {}".format(self.num_heavy_rl))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.dehaze_dir + 'dehaze_reside_val_low.json'
        temp_ids += [self.args.dehaze_dir + id_['image_path'].strip() for id_ in json.load(open(hazy))]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]

        self.hazy_counter = 0

        self.num_hazy = len(self.hazy_ids)
        print("Total low validation Hazy Ids : {}".format(self.num_hazy))

    def _init_heavyhazy_ids(self):
        temp_ids = []
        hazy = self.args.dehaze_dir + 'dehaze_reside_val_high.json'
        temp_ids += [self.args.dehaze_dir + id_['image_path'].strip() for id_ in json.load(open(hazy))]
        self.heavy_hazy_ids = [{"clean_id": x, "de_type": 7} for x in temp_ids]

        self.num_heavy_hazy = len(self.heavy_hazy_ids)
        print("Total Heavy validation Hazy Ids : {}".format(self.num_heavy_hazy))

    def _init_snow_ids(self):
        temp_ids = []
        snow = self.args.desnow_dir + 'desnow_snow100_val_low.json'
        temp_ids += [self.args.desnow_dir + id_['image_path'].strip() for id_ in json.load(open(snow))]
        self.snow_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]

        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print("Total low validation Snow Ids : {}".format(self.num_snow))

    def _init_heavysnow_ids(self):
        temp_ids = []
        snow = self.args.desnow_dir + 'desnow_snow100_val_high.json'
        temp_ids += [self.args.desnow_dir + id_['image_path'].strip() for id_ in json.load(open(snow))]
        self.heavy_snow_ids = [{"clean_id": x, "de_type": 8} for x in temp_ids]

        self.num_heavy_snow = len(self.heavy_snow_ids)
        print("Total Heavy validation Snow Ids : {}".format(self.num_heavy_snow))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.replace('input', 'target')
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        nonhazy_name = hazy_name.replace('hazy', 'clear')
        x = nonhazy_name.split('/')[-1]
        y = x.split('_')[0]
        ext = '.' + x.split('.')[-1]
        nonhazy_name = nonhazy_name.replace(x, y + ext)
        return nonhazy_name

    def _get_nonsnow_name(self, snow_name):
        nonsnow_name = snow_name.replace('data2', 'gt')
        return nonsnow_name

    def _merge_ids(self):
        self.sample_ids = []

        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids

        if "derain_heavy" in self.de_type:
            self.sample_ids += self.heavy_rs_ids
        if "dehaze_heavy" in self.de_type:
            self.sample_ids += self.heavy_hazy_ids
        if "desnow_heavy" in self.de_type:
            self.sample_ids += self.heavy_snow_ids

    def get_context(self, sample, de_id):
        if de_id == 3:
            list_of_values = self.rs_ids
        elif de_id == 4:
            list_of_values = self.hazy_ids
        elif de_id == 5:
            list_of_values = self.snow_ids
        elif de_id == 6:
            list_of_values = self.heavy_rs_ids
        elif de_id == 7:
            list_of_values = self.heavy_hazy_ids
        elif de_id == 8:
            list_of_values = self.heavy_snow_ids
        else:
            raise ValueError("Invalid degradation type encountered")

        second_value = sample
        while second_value == sample:
            second_value = random.choice(list_of_values)

        return second_value

    def context_transform(self, n_px):
        return Compose([
            ToPILImage(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        context_id = self.get_context(sample, de_id)
        assert context_id["de_type"] == de_id

        if de_id == 3 or de_id == 6:
            # Rain Streak Removal
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_name = self._get_gt_name(context_id["clean_id"])
            clean_context_img = crop_img(np.array(Image.open(clean_context_name).convert('RGB')), base=16)

        elif de_id == 4 or de_id == 7:
            # Dehazing with SOTS outdoor training set
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_nonhazy_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_name = self._get_nonhazy_name(context_id["clean_id"])
            clean_context_img = crop_img(np.array(Image.open(clean_context_name).convert('RGB')), base=16)

        elif de_id == 5 or de_id == 8:
            # Desnowing with Snow100 training set
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_nonsnow_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_name = self._get_nonsnow_name(context_id["clean_id"])
            clean_context_img = crop_img(np.array(Image.open(clean_context_name).convert('RGB')), base=16)

        degrad_img = resize_img(degrad_img, idx)
        clean_img = resize_img(clean_img, idx)

        clip_transform = self.context_transform(224)
        degrad_context_patch, clean_context_patch = (clip_transform(degrad_context_img),
                                                     clip_transform(clean_context_img))

        clean_patch = self.toTensor(clean_img)
        degrad_patch = self.toTensor(degrad_img)

        return [clean_name, de_id], degrad_patch, clean_patch, degrad_context_patch, clean_context_patch

    def __len__(self):
        return len(self.sample_ids)


class TestDataset_IC(Dataset):
    def __init__(self, args, pair=None):
        super(TestDataset_IC, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []

        self.de_temp = 0
        print("Setting up In Context test dataset")

        self._init_ids()
        self._merge_ids()
        self.toTensor = ToTensor()
        self.pair = pair
        self.prev_context = None
        if pair is None:
            if args.in_context_file is not None:
                self._init_context_ids()
        else:
            self.deg_file, self.gt_file = pair

    def _init_context_ids(self):
        temp_ids = []
        temp_gt_ids = []
        rs = self.args.in_context_dir + self.args.in_context_file
        temp_ids += [os.path.join(self.args.in_context_dir, id_['image_path'].strip()) for id_ in json.load(open(rs))]
        temp_gt_ids += [os.path.join(self.args.in_context_dir, id_['target_path'].strip()) for id_ in json.load(open(rs))]
        self.ic_ids = [{"clean_id": x, "gt_id": y} for x, y in zip(temp_ids, temp_gt_ids)]

        self.ic_counter = 0
        self.num_ic = len(self.ic_ids)
        print("Total In-Context Ids : {}".format(self.num_ic))

    def _init_ids(self):
        temp_ids = []
        temp_gt_ids = []
        rs = self.args.test_dir + self.args.test_json
        temp_ids += [os.path.join(self.args.test_dir, id_['image_path']).strip() for id_ in json.load(open(rs))]
        temp_gt_ids += [os.path.join(self.args.test_dir, id_['target_path'].strip()) for id_ in json.load(open(rs))]
        self.rs_ids = [{"clean_id": x, "gt_id": y} for x, y in zip(temp_ids, temp_gt_ids)]

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Test Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _merge_ids(self):
        self.sample_ids = []
        self.sample_ids += self.rs_ids
        print(len(self.sample_ids))

    def get_context(self, sample):
        second_value = sample
        while second_value == sample:
            second_value = random.choice(self.ic_ids)

        return second_value

    def context_transform(self, n_px):
        return Compose([
            ToPILImage(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        if self.pair is None:
            context_id = self.get_context(sample)
            clean_name = sample["clean_id"]
            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_img = crop_img(np.array(Image.open(context_id["gt_id"]).convert('RGB')), base=16)
        else:
            degrad_context_img = crop_img(np.array(Image.open(self.deg_file).convert('RGB')), base=16)
            clean_context_img = crop_img(np.array(Image.open(self.gt_file).convert('RGB')), base=16)
            clean_name = sample["clean_id"]

        degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["gt_id"]).convert('RGB')), base=16)

        clip_transform = self.context_transform(224)
        degrad_context_patch, clean_context_patch = (clip_transform(degrad_context_img),
                                                     clip_transform(clean_context_img))

        clean_patch = self.toTensor(clean_img)
        degrad_patch = self.toTensor(degrad_img)

        return [clean_name], degrad_patch, clean_patch, degrad_context_patch, clean_context_patch

class TestDataset_Folder(Dataset):
    def __init__(self, args, pair=None):
        super(TestDataset_Folder, self).__init__()
        self.args = args

        print("Setting up Folder based test dataset")

        self._init_ids()
        self._merge_ids()
        self.toTensor = ToTensor()
        self.pair = pair
        self.prev_context = None
        if pair is None:
            if args.in_context_file is not None:
                self._init_context_ids()
        else:
            self.deg_file, self.gt_file = pair

    def _init_context_ids(self):
        temp_ids = []
        temp_gt_ids = []
        rs = self.args.test_dir + self.args.in_context_file
        temp_ids += [id_['image_path'].strip() for id_ in json.load(open(rs))]
        temp_gt_ids += [id_['target_path'].strip() for id_ in json.load(open(rs))]
        self.ic_ids = [{"clean_id": x, "gt_id": y} for x, y in zip(temp_ids, temp_gt_ids)]

        self.ic_counter = 0
        self.num_ic = len(self.ic_ids)
        print("Total In-Context Ids : {}".format(self.num_ic))

    def _init_ids(self):
        temp_ids = []
        temp_gt_ids = []
        dir_deg = self.args.test_dir + 'degraded'
        dir_gt = self.args.test_dir + 'GT'
        for img in natsort.natsorted(os.listdir(dir_deg)):
            temp_ids += [os.path.join(dir_deg, img)]

        for gt in natsort.natsorted(os.listdir(dir_gt)):
            temp_gt_ids += [os.path.join(dir_gt, gt)]

        self.rs_ids = [{"clean_id": x, "gt_id": y} for x, y in zip(temp_ids, temp_gt_ids)]

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Test Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _merge_ids(self):
        self.sample_ids = []
        self.sample_ids += self.rs_ids
        print(len(self.sample_ids))

    def get_context(self, sample):
        second_value = sample
        while second_value == sample:
            second_value = random.choice(self.ic_ids)

        return second_value

    def context_transform(self, n_px):
        return Compose([
            ToPILImage(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        if self.pair is None:
            context_id = self.get_context(sample)
            clean_name = sample["clean_id"]
            degrad_context_img = crop_img(np.array(Image.open(context_id["clean_id"]).convert('RGB')), base=16)
            clean_context_img = crop_img(np.array(Image.open(context_id["gt_id"]).convert('RGB')), base=16)
        else:
            degrad_context_img = crop_img(np.array(Image.open(self.deg_file).convert('RGB')), base=16)
            clean_context_img = crop_img(np.array(Image.open(self.gt_file).convert('RGB')), base=16)
            clean_name = sample["clean_id"]

        degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["gt_id"]).convert('RGB')), base=16)

        clip_transform = self.context_transform(224)
        degrad_context_patch, clean_context_patch = (clip_transform(degrad_context_img),
                                                     clip_transform(clean_context_img))

        clean_patch = self.toTensor(clean_img)
        degrad_patch = self.toTensor(degrad_img)

        return [clean_name], degrad_patch, clean_patch, degrad_context_patch, clean_context_patch

class RealDataset(Dataset):
    def __init__(self, args):
        super(RealDataset, self).__init__()
        self.args = args
        self.img_dir = args.test_dir
        self.toTensor = ToTensor()
        
        print("Setting up Real Dataset (No GT inference)")
        
        # 1. 获取目录下所有的待测试真实图片
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Directory {self.img_dir} does not exist!")
            
        self.img_paths = [os.path.join(self.img_dir, f) for f in natsort.natsorted(os.listdir(self.img_dir)) if f.lower().endswith(valid_ext)]
        print(f"Total Real Images to process: {len(self.img_paths)}")
        
        # 2. 解析 JSON 获取 In-Context 提示图对
        if args.in_context_file is not None and args.in_context_dir is not None:
            json_path = os.path.join(args.in_context_dir, args.in_context_file)
            with open(json_path, 'r') as f:
                context_data = json.load(f)
            
            self.ic_ids = []
            for id_ in context_data:
                # 拼接目录和 JSON 中的相对路径
                degrad_path = os.path.join(args.in_context_dir, id_['image_path'].strip())
                clean_path = os.path.join(args.in_context_dir, id_['target_path'].strip())
                self.ic_ids.append({"degrad_id": degrad_path, "clean_id": clean_path})
                
            print(f"Loaded {len(self.ic_ids)} In-Context pairs from JSON.")
        else:
            raise ValueError("Real dataset requires --in_context_dir and --in_context_file to load prompts from JSON!")

    def context_transform(self, n_px):
        return Compose([
            ToPILImage(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # 1. 加载待推理的真实退化图
        degrad_img = crop_img(np.array(Image.open(img_path).convert('RGB')), base=16)
        degrad_patch = self.toTensor(degrad_img)
        
        # 2. 从 JSON 解析出来的列表中，随机选取一对 In-Context 提示图
        context_pair = random.choice(self.ic_ids)
        degrad_context_img = crop_img(np.array(Image.open(context_pair["degrad_id"]).convert('RGB')), base=16)
        clean_context_img = crop_img(np.array(Image.open(context_pair["clean_id"]).convert('RGB')), base=16)
        
        # 3. 对提示图进行 CLIP 标准化预处理
        clip_transform = self.context_transform(224)
        degrad_context_patch = clip_transform(degrad_context_img)
        clean_context_patch = clip_transform(clean_context_img)
        
        return [img_path], degrad_patch, degrad_context_patch, clean_context_patch