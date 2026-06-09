import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from utils.dataset_utils import TrainDataset, ValDataset
from net.model import SMILE
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options.options import options as opt
from pytorch_lightning.strategies.ddp import DDPStrategy

class SMILEModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SMILE(decoder=True)
        self.loss_fn = nn.L1Loss()

        # 冻结 CLIP
        for name, param in self.named_parameters():
            if 'clip' in name:
                param.requires_grad = False

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x, context_embs):
        return self.net(x, context_embs)

    # =========================================================
    # Pseudo field
    # =========================================================
    def build_pseudo_field(self, degrad_patch, clean_patch):
        """
        基于退化图与清晰图差异构建 soft pseudo field.
        输出范围 [0,1]
        """
        pseudo = torch.mean(torch.abs(degrad_patch - clean_patch), dim=1, keepdim=True)

        # 轻微平滑，减少伪监督中的高频噪声
        pseudo = F.avg_pool2d(pseudo, kernel_size=5, stride=1, padding=2)

        pseudo = pseudo / (pseudo.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return pseudo.clamp(0.0, 1.0)

    # =========================================================
    # Multi-scale consistency
    # =========================================================
    def multi_scale_field_consistency_loss(self, degradation_fields):
        if degradation_fields is None or len(degradation_fields) <= 1:
            return torch.zeros((), device=self.device)

        losses = []
        for i in range(len(degradation_fields) - 1):
            coarse = degradation_fields[i]
            fine = degradation_fields[i + 1]
            coarse_up = F.interpolate(
                coarse, size=fine.shape[-2:], mode='bilinear', align_corners=False
            )
            losses.append(F.l1_loss(coarse_up, fine))

        return torch.stack(losses).mean()

    # =========================================================
    # Field supervision loss
    # =========================================================
    def multi_scale_field_supervision_loss(self, degradation_fields, pseudo_field):
        """
        对三层 field 做分层监督：
        - field3 更偏全局场，权重大
        - field1 更偏局部控制图，权重略低
        """
        if degradation_fields is None or len(degradation_fields) == 0:
            return torch.zeros((), device=self.device)

        level_weights = [0.5, 0.3, 0.2]
        losses = []

        for i, field in enumerate(degradation_fields):
            field_up = F.interpolate(
                field, size=pseudo_field.shape[-2:], mode='bilinear', align_corners=False
            )
            weight = level_weights[i] if i < len(level_weights) else 1.0 / len(degradation_fields)
            losses.append(weight * F.l1_loss(field_up, pseudo_field))

        return torch.stack(losses).sum()

    # =========================================================
    # Utility: field entropy
    # =========================================================
    def compute_field_entropy(self, field):
        eps = 1e-6
        entropy = -(
            field * torch.log(field + eps) +
            (1 - field) * torch.log(1 - field + eps)
        ).mean()
        return entropy

    def fft_loss(self, pred, target):
        """
        频域一致性约束：
        对 restored 与 clean_patch 的傅里叶表示做 L1 约束
        """
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        pred_fft = torch.view_as_real(pred_fft)      # [B, C, H, W, 2]
        target_fft = torch.view_as_real(target_fft)

        return F.l1_loss(pred_fft, target_fft)
    
    # =========================================================
    # Utility: save fields
    # =========================================================
    # def save_multi_level_fields(self, save_dir, img_names, fields, prefix=""):
    #     """
    #     保存三层 field 和 inverse field
    #     fields 顺序:
    #         fields[0] = field3
    #         fields[1] = field2
    #         fields[2] = field1
    #     """
    #     if fields is None or len(fields) == 0:
    #         return

    #     os.makedirs(save_dir, exist_ok=True)

    #     level_names = ["field_l3", "field_l2", "field_l1"]
    #     num_samples = min(4, fields[0].size(0))

    #     for i in range(num_samples):
    #         current_name = img_names[i] if isinstance(img_names, (list, tuple)) else img_names
    #         img_name_base = os.path.splitext(os.path.basename(current_name))[0]

    #         for lvl, field in enumerate(fields):
    #             lvl_name = level_names[lvl] if lvl < len(level_names) else f"field_l{lvl}"
    #             f = field[i]
    #             inv_f = 1.0 - f

    #             save_image(
    #                 f,
    #                 os.path.join(save_dir, f"{prefix}{img_name_base}_{lvl_name}.png"),
    #                 normalize=False,
    #                 value_range=(0, 1)
    #             )
    #             save_image(
    #                 inv_f,
    #                 os.path.join(save_dir, f"{prefix}{img_name_base}_{lvl_name}_inv.png"),
    #                 normalize=False,
    #                 value_range=(0, 1)
    #             )

    # =========================================================
    # Training
    # =========================================================
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch, degrad_context, clean_context) = batch

        degrad_context = degrad_context.float()
        clean_context = clean_context.float()

        if degrad_context.ndim != 4 or clean_context.ndim != 4:
            raise ValueError(f"Invalid context shape: degrad={degrad_context.shape}, clean={clean_context.shape}")

        # Curriculum
        if self.current_epoch < 15:
            self.net.enable_fbsr = False
        else:
            self.net.enable_fbsr = True

        restored, degradation_fields = self.net(degrad_patch, [degrad_context, clean_context])

        # =========================================================
        # 可视化多层 field
        # =========================================================
        # if batch_idx % 5000 == 0 and degradation_fields is not None and len(degradation_fields) > 0:
        #     save_dir = os.path.join("train_results", "fields", f"epoch_{self.current_epoch:03d}_{batch_idx}")
        #     self.save_multi_level_fields(save_dir, clean_name, degradation_fields)
        #     print(f"[Debug] 保存多尺度 field 至: {save_dir}")

        # =========================================================
        # 1. 主恢复损失
        # =========================================================
        loss_restore = self.loss_fn(restored, clean_patch)

        # =========================================================
        # 2. 伪退化场监督
        # =========================================================
        pseudo_field = self.build_pseudo_field(degrad_patch, clean_patch)
        loss_field = self.multi_scale_field_supervision_loss(degradation_fields, pseudo_field)

        # =========================================================
        # 3. 多尺度一致性约束
        # =========================================================
        loss_cons = self.multi_scale_field_consistency_loss(degradation_fields)

        # =========================================================
        # 4. 总损失
        # =========================================================
        loss_fft = self.fft_loss(restored, clean_patch)

        # =========================================================
        # 5. 总损失
        # =========================================================
        lambda_field = 0.05
        lambda_cons = 0.02
        lambda_fft = 0.01

        total_loss = (
            loss_restore
            + lambda_field * loss_field
            + lambda_cons * loss_cons
            + lambda_fft * loss_fft
        )

        # =========================================================
        # Logs
        # =========================================================
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("loss_restore", loss_restore)
        self.log("loss_field", loss_field)
        self.log("loss_cons", loss_cons)
        self.log("loss_fft", loss_fft)

        # if degradation_fields is not None and len(degradation_fields) > 0:
        #     field3 = degradation_fields[0].detach()
        #     self.log("field3_mean", field3.mean(), prog_bar=True)
        #     self.log("field3_min", field3.min())
        #     self.log("field3_max", field3.max())
        #     self.log("field3_var", field3.var())
        #     self.log("field3_entropy", self.compute_field_entropy(field3))

        #     if len(degradation_fields) > 1:
        #         field2 = degradation_fields[1].detach()
        #         self.log("field2_mean", field2.mean())
        #         self.log("field2_min", field2.min())
        #         self.log("field2_max", field2.max())
        #         self.log("field2_var", field2.var())
        #         self.log("field2_entropy", self.compute_field_entropy(field2))

        #     if len(degradation_fields) > 2:
        #         field1 = degradation_fields[2].detach()
        #         self.log("field1_mean", field1.mean())
        #         self.log("field1_min", field1.min())
        #         self.log("field1_max", field1.max())
        #         self.log("field1_var", field1.var())
        #         self.log("field1_entropy", self.compute_field_entropy(field1))

        #     # 监控温度参数
        #     if hasattr(self.net, 'prompt1') and hasattr(self.net.prompt1, 'sdfe'):
        #         self.log("temp_scale_l1", self.net.prompt1.sdfe.temperature.data.item(), prog_bar=True)
        #     if hasattr(self.net, 'prompt2') and hasattr(self.net.prompt2, 'sdfe'):
        #         self.log("temp_scale_l2", self.net.prompt2.sdfe.temperature.data.item())
        #     if hasattr(self.net, 'prompt3') and hasattr(self.net.prompt3, 'sdfe'):
        #         self.log("temp_scale_l3", self.net.prompt3.sdfe.temperature.data.item())

        return total_loss

    # =========================================================
    # Validation
    # =========================================================
    def validation_step(self, batch, batch_idx):
        self.net.enable_fbsr = (self.current_epoch >= 15)

        ([clean_name, de_id], degrad_patch, clean_patch, degrad_context, clean_context) = batch

        if degrad_context.ndim != 4 or clean_context.ndim != 4:
            raise ValueError(f"Invalid context shape: degrad={degrad_context.shape}, clean={clean_context.shape}")

        if isinstance(de_id, torch.Tensor):
            de_id = int(de_id.item())
        elif isinstance(de_id, (list, tuple)):
            de_id = int(de_id[0])
        else:
            de_id = int(de_id)

        degrad_context = degrad_context.float()
        clean_context = clean_context.float()

        restored, val_fields = self.net(degrad_patch, [degrad_context, clean_context])

        img_name = clean_name[0] if isinstance(clean_name, (list, tuple)) else clean_name
        img_name_base = os.path.splitext(os.path.basename(img_name))[0]

        # 保存恢复图
        # save_dir1 = os.path.join("val_results", "restored", f"epoch_{self.current_epoch:03d}")
        # os.makedirs(save_dir1, exist_ok=True)
        # save_image(restored, os.path.join(save_dir1, f"{img_name_base}_restored.png"))

        # 保存多层 field
        # if val_fields is not None and len(val_fields) > 0:
        #     val_field_dir = os.path.join("val_results", "fields", f"epoch_{self.current_epoch:03d}")
        #     self.save_multi_level_fields(val_field_dir, [img_name], val_fields)

        score_psnr = self.val_psnr(restored, clean_patch)
        score_ssim = self.val_ssim(restored, clean_patch)
        loss = self.loss_fn(restored, clean_patch)

        self.log_dict({"psnr": score_psnr, "ssim": score_ssim})

        if de_id == 3:
            self.log_dict({"valid_loss": loss, "psnr_rain": score_psnr, "ssim_rain": score_ssim})
        elif de_id == 4:
            self.log_dict({"valid_loss": loss, "psnr_haze": score_psnr, "ssim_haze": score_ssim})
        elif de_id == 5:
            self.log_dict({"valid_loss": loss, "psnr_snow": score_psnr, "ssim_snow": score_ssim})
        elif de_id == 6:
            self.log_dict({"valid_loss": loss, "psnr_heavyrain": score_psnr, "ssim_heavyrain": score_ssim})
        elif de_id == 7:
            self.log_dict({"valid_loss": loss, "psnr_heavyhaze": score_psnr, "ssim_heavyhaze": score_ssim})
        elif de_id == 8:
            self.log_dict({"valid_loss": loss, "psnr_heavysnow": score_psnr, "ssim_heavysnow": score_ssim})

        return loss

    def lr_scheduler_step(self, scheduler, metric, *args, **kwargs):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="SMILE_fft")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = TrainDataset(opt)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    valset = ValDataset(opt)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers
    )

    model = SMILEModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        monitor="psnr",
        mode="max",
        save_top_k=-1,
        every_n_epochs=1,
        filename="smile-epoch{epoch:03d}-psnr{psnr:.4f}",
        auto_insert_metric_name=False,
        save_last=True,
        save_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=2,
        strategy=DDPStrategy(find_unused_parameters=True), 
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0
    )

    if opt.resume:
        trainer.fit(
            model=model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader,
            ckpt_path=opt.ckpt_path
        )
    else:
        trainer.fit(
            model=model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader
        )


if __name__ == '__main__':
    main()