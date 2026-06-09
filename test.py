import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pytorch_lightning as pl

from utils.dataset_utils import TestDataset_IC, TestDataset_Folder
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from net.model import SMILE


class SMILEModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SMILE(decoder=True)
        self.loss_fn = nn.L1Loss()

        # 与训练代码保持一致：冻结 CLIP 参数
        for name, param in self.named_parameters():
            if 'clip' in name:
                param.requires_grad = False

    def forward(self, x, context_embs):
        return self.net(x, context_embs)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def test_ds(net, dataset, args):
    output_path = args.output_path
    ensure_dir(output_path)

    testloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=0
    )

    psnr = AverageMeter()
    ssim = AverageMeter()
    lpips_scores = AverageMeter()
    fid_value = 0.0

    calc_lpips = args.lpips
    calc_fid = args.fid

    if calc_lpips:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

    if calc_fid:
        fid_metric = FrechetInceptionDistance(feature=2048).cuda()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch,
             degrad_context, clean_context) in tqdm(testloader):

            clean_name = clean_name[0].split('/')[-1]
            save_name = os.path.splitext(clean_name)[0]

            degrad_patch = degrad_patch.cuda(non_blocking=True)
            clean_patch = clean_patch.cuda(non_blocking=True)
            degrad_context = degrad_context.cuda(non_blocking=True).float()
            clean_context = clean_context.cuda(non_blocking=True).float()

            if degrad_context.ndim != 4 or clean_context.ndim != 4:
                raise ValueError(
                    f"Invalid context shape: degrad={degrad_context.shape}, clean={clean_context.shape}"
                )

            net.net.enable_fbsr = True

            restored, degradation_fields = net(degrad_patch, [degrad_context, clean_context])

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            restored = restored.clamp(0, 1)
            clean_patch = clean_patch.clamp(0, 1)

            # LPIPS 需要 [-1, 1]
            if calc_lpips:
                restored_lpips = 2 * restored - 1
                clean_patch_lpips = 2 * clean_patch - 1
                lpips_value = lpips_metric(restored_lpips, clean_patch_lpips)
                lpips_scores.update(lpips_value.item(), N)

            # FID 需要 uint8
            if calc_fid:
                restored_fid_uint8 = restored.mul(255).byte()
                clean_patch_fid_uint8 = clean_patch.mul(255).byte()
                fid_metric.update(restored_fid_uint8, real=False)
                fid_metric.update(clean_patch_fid_uint8, real=True)

            # 保存恢复图和 GT
            save_image_tensor(restored, os.path.join(output_path, save_name + ".png"))

    if calc_fid:
        fid_value = fid_metric.compute().item()

    print(
        "psnr: %.2f, ssim: %.4f, lpips: %.4f, fid: %.4f"
        % (psnr.avg, ssim.avg, lpips_scores.avg, fid_value)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        help='0: load from json, 1: load images from folder'
    )

    parser.add_argument('--test_dir', type=str, default="test/dehaze/")
    parser.add_argument('--test_json', type=str, default="test/dehaze/")
    parser.add_argument('--output_path', type=str, default="output/")
    parser.add_argument('--ckpt_name', type=str, default=".ckpt")
    parser.add_argument('--in_context_dir', type=str, default=None)
    parser.add_argument('--in_context_file', type=str, default=None)
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--fid', action='store_true')
    parser.add_argument('--degrad_context', type=str, default=None)
    parser.add_argument('--clean_context', type=str, default=None)

    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_name

    if testopt.degrad_context is not None and testopt.clean_context is not None:
        pair = (testopt.degrad_context, testopt.clean_context)
    else:
        pair = None

    if testopt.mode == 1:
        dset = TestDataset_Folder(testopt, pair=pair)
    else:
        dset = TestDataset_IC(testopt, pair=pair)

    print("CKPT name : {}".format(ckpt_path))

    net = SMILEModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    print("Loaded!")
    print("Start testing...")

    test_ds(net, dset, testopt)