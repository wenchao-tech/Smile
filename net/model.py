import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
import clip


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class ContextAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, y):
        # x: B x C x H x W
        # y: B x N x C
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        y = self.norm(y)

        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H).contiguous()
        return out



class CWM(nn.Module):
    """Channel Weighting Module"""
    def __init__(self, channels, reduction=16):
        super(CWM, self).__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SAG(nn.Module):
    """Spatial Attention Generator"""
    def __init__(self):
        super(SAG, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class MiniDenseHourglass(nn.Module):
    """Lightweight dense hourglass"""
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1, groups=dim),
            nn.Conv2d(dim, dim * 2, 1)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.fuse = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        d = self.down(x)
        u = self.up(d)
        return self.fuse(torch.cat([x, u], dim=1))



class FieldGuidedSFT(nn.Module):
    """Field-Guided Spatial Feature Transform"""
    def __init__(self, channels):
        super(FieldGuidedSFT, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        nn.init.zeros_(self.conv_gamma[-1].weight)
        nn.init.zeros_(self.conv_gamma[-1].bias)
        nn.init.zeros_(self.conv_beta[-1].weight)
        nn.init.zeros_(self.conv_beta[-1].bias)

    def forward(self, x, field):
        gamma = 0.1 * torch.tanh(self.conv_gamma(field))
        beta = 0.1 * torch.tanh(self.conv_beta(field))
        return x * (1 + gamma) + beta


class FCR(nn.Module):
    """Field-Conditioned Feature Restoration"""
    def __init__(self, dim):
        super(FCR, self).__init__()
        self.proj_in = nn.Conv2d(dim + 3, dim, kernel_size=1)
        self.hourglass = MiniDenseHourglass(dim)
        self.sft = FieldGuidedSFT(dim)
        self.cwm = CWM(dim)
        self.sag = SAG()
        self.proj_out = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x, orig_img, field=None):
        b, _, h, w = x.shape
        down_img = F.interpolate(orig_img, size=(h, w), mode='bilinear', align_corners=False)

        feat = torch.cat([x, down_img], dim=1)
        feat = self.proj_in(feat)

        if field is not None:
            field_down = F.interpolate(field, size=(h, w), mode='bilinear', align_corners=False)
            field_down = 2.0 * field_down - 1.0  # center to [-1, 1]
            feat = self.sft(feat, field_down)

        hg_out = self.hourglass(feat)
        channel_weight = self.cwm(feat)
        spatial_weight = self.sag(feat)

        attended_feat = hg_out * channel_weight * spatial_weight
        out = self.proj_out(torch.cat([attended_feat, feat], dim=1))
        return out



class WeatherCueExtractor(nn.Module):
    """
    Weather-aware cue extractor:
    - directional energy: rain streaks
    - blob / contrast energy: snow / raindrop / occlusion-like patterns
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        k0 = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        k45 = torch.tensor([[0, 1, 2],
                            [-1, 0, 1],
                            [-2, -1, 0]], dtype=torch.float32)
        k90 = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32)
        k135 = torch.tensor([[-2, -1, 0],
                             [-1, 0, 1],
                             [0, 1, 2]], dtype=torch.float32)

        dir_base = torch.stack([k0, k45, k90, k135], dim=0).unsqueeze(1)
        dir_weight = dir_base.repeat(channels, 1, 1, 1)
        self.register_buffer("dir_filters", dir_weight)

        lap = torch.tensor([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]], dtype=torch.float32)
        lap_weight = lap.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("lap_filter", lap_weight)

        self.avg_pool_3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool_5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        # directional responses
        dir_responses = F.conv2d(x, self.dir_filters, padding=1, groups=self.channels)  # [B,4C,H,W]
        b, _, h, w = dir_responses.shape
        dir_responses = dir_responses.view(b, self.channels, 4, h, w)
        abs_dir = dir_responses.abs()

        dir_max = abs_dir.max(dim=2).values.mean(dim=1, keepdim=True)
        dir_mean = abs_dir.mean(dim=2).mean(dim=1, keepdim=True)

        # blob / local occlusion cues
        lap_resp = F.conv2d(x, self.lap_filter, padding=1, groups=self.channels).abs()
        lap_energy = lap_resp.mean(dim=1, keepdim=True)

        local_contrast = (x - self.avg_pool_3(x)).abs().mean(dim=1, keepdim=True)
        multi_scale_contrast = (self.avg_pool_3(x) - self.avg_pool_5(x)).abs().mean(dim=1, keepdim=True)

        return torch.cat([dir_max, dir_mean, lap_energy, multi_scale_contrast + local_contrast], dim=1)


class SDFE(nn.Module):
    """Spatial Degradation Field Estimator"""
    def __init__(self, in_channels):
        super(SDFE, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.weather_extractor = WeatherCueExtractor(in_channels)

        # feature + coord(2) + weather cues(4)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 2 + 4, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.pre(x)
        weather_cues = self.weather_extractor(feat)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

        feat_all = torch.cat([feat, grid, weather_cues], dim=1)
        logits = self.mlp(feat_all)

        safe_temp = torch.abs(self.temperature) + 1e-4
        degradation_field = torch.sigmoid(logits / safe_temp)
        return degradation_field



class FBSR(nn.Module):
    """Field-Based Semantic Routing"""
    def __init__(self, prompt_dim=128, lin_dim=512, context_dim=768, heads=4, bias=False):
        super(FBSR, self).__init__()
        self.linear_layer_ctx = nn.Linear(context_dim, prompt_dim)
        self.ctx_attn = ContextAttention(prompt_dim, heads=heads)
        self.proj_features = nn.Conv2d(lin_dim, prompt_dim, kernel_size=1, bias=False)

        self.fusion = CrossAttention(prompt_dim, heads=heads)

        self.field_fuser = nn.Sequential(
            nn.Conv2d(prompt_dim * 2, prompt_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU()
        )

        self.sdfe = SDFE(prompt_dim)

        # initialized to 0 => alpha = sigmoid(0) = 0.5
        self.route_scale = nn.Parameter(torch.tensor(0.0))

        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, context, enable_routing=True):
        context = context.permute(1, 0, 2)  # [B_ctx, L, C]
        context = self.linear_layer_ctx(context)
        context = F.gelu(context)

        Bc, _, _ = context.size()
        assert Bc % 2 == 0, "In-context samples not paired"

        degrad_context = context[: Bc // 2, :, :]
        clean_context = context[Bc // 2:, :, :]
        context_cat = torch.cat([degrad_context, clean_context], dim=1)
        merged_context = self.ctx_attn(context_cat)

        x_proj = F.gelu(self.proj_features(x))
        global_prompt_features = self.fusion(x_proj, merged_context)

        # semantic-aware field estimation
        field_feat = self.field_fuser(torch.cat([x_proj, global_prompt_features], dim=1))
        degradation_field = self.sdfe(field_feat)

        if enable_routing:
            gate = 2.0 * degradation_field - 1.0   # [0,1] -> [-1,1]
            alpha = torch.sigmoid(self.route_scale)
            routed_prompt_features = global_prompt_features * (1.0 + alpha * gate)
        else:
            routed_prompt_features = global_prompt_features

        prompt = F.gelu(self.conv3x3(routed_prompt_features))
        return prompt, degradation_field


class SMILE(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
        clip_path="./ViT-B-32.pt"
    ):
        super(SMILE, self).__init__()
        self.enable_fbsr = True
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        if self.decoder:
            self.prompt1 = FBSR(prompt_dim=64,  lin_dim=96,  context_dim=768, heads=heads[1])
            self.prompt2 = FBSR(prompt_dim=128, lin_dim=192, context_dim=768, heads=heads[2])
            self.prompt3 = FBSR(prompt_dim=320, lin_dim=384, context_dim=768, heads=heads[3])

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])

        # Level 3
        self.up4_3 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(
            dim=int(dim * 2 ** 2) + 512, num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type
        )
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 512, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.fcr_level3 = FCR(int(dim * 2 ** 2))
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        # Level 2
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(
            dim=int(dim * 2 ** 1) + 224, num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type
        )
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) + 224, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.fcr_level2 = FCR(int(dim * 2 ** 1))
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        # Level 1
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.noise_level1 = TransformerBlock(
            dim=int(dim * 2 ** 1) + 64, num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type
        )
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 64, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.fcr_level1 = FCR(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.clip, self.preprocess = clip.load(clip_path, device="cpu")
        self.clip.eval()
        self.target_layers = ['visual.transformer.resblocks.11.ln_2']
        self.intermediate_features = {}

        self.hooks = []
        for layer_name in self.target_layers:
            layer = self.clip
            for name in layer_name.split("."):
                layer = getattr(layer, name)

            hook = layer.register_forward_hook(
                lambda module, input, output, name=layer_name: self.hook_fn(module, input, output, name)
            )
            self.hooks.append(hook)

    def hook_fn(self, module: nn.Module, input, output, layer_name):
        self.intermediate_features[layer_name] = output

    def forward(self, inp_img, context_embs):
        self.intermediate_features.clear()

        context_embs = torch.cat(context_embs, dim=0)
        context_embs = self.clip.encode_image(context_embs)

        hook_key = self.target_layers[0]
        if hook_key not in self.intermediate_features:
            raise RuntimeError(f"Hook feature '{hook_key}' was not captured from CLIP.")

        context_embs = self.intermediate_features[hook_key]

        # Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        degradation_fields = []

        # ================== Level 3 ==================
        if self.decoder:
            dec3_param, field3 = self.prompt3(latent, context_embs, enable_routing=self.enable_fbsr)
            degradation_fields.append(field3)
            latent = torch.cat([latent, dec3_param], dim=1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        if self.decoder and len(degradation_fields) >= 1:
            inp_dec_level3 = self.fcr_level3(inp_dec_level3, inp_img, degradation_fields[0])
        else:
            inp_dec_level3 = self.fcr_level3(inp_dec_level3, inp_img, None)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # ================== Level 2 ==================
        if self.decoder:
            dec2_param, field2 = self.prompt2(out_dec_level3, context_embs, enable_routing=self.enable_fbsr)
            degradation_fields.append(field2)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], dim=1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        if self.decoder and len(degradation_fields) >= 2:
            inp_dec_level2 = self.fcr_level2(inp_dec_level2, inp_img, degradation_fields[1])
        else:
            inp_dec_level2 = self.fcr_level2(inp_dec_level2, inp_img, None)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # ================== Level 1 ==================
        if self.decoder:
            dec1_param, field1 = self.prompt1(out_dec_level2, context_embs, enable_routing=self.enable_fbsr)
            degradation_fields.append(field1)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], dim=1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)

        if self.decoder and len(degradation_fields) >= 3:
            inp_dec_level1 = self.fcr_level1(inp_dec_level1, inp_img, degradation_fields[2])
        else:
            inp_dec_level1 = self.fcr_level1(inp_dec_level1, inp_img, None)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, degradation_fields