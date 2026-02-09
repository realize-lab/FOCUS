

from functools import partial

import torch
import torch.nn as nn
import yaml
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple

import numpy as np

from einops import rearrange

import torch
import torch.nn as nn
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelAttention(nn.Module):
    def __init__(self, in_chans, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_chans, in_chans // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_chans // reduction_ratio, in_chans, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        
        # x: B, C, D, H, W
        attention = self.avg_pool(x) 
    
        # Apply the fully connected layers
        attention = self.fc(attention) 
       
        return x * attention


class Residual3DConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size, stride, padding),
            nn.BatchNorm3d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.residual_conv = nn.Conv3d(in_chans, out_chans, 1) if in_chans != out_chans else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual_conv(x)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """ Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=1,
            tubelet_size=1,
            in_chans=43,
            embed_dim=1024,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16,
                 num_frames=1, tubelet_size=1,
                 in_chans=43, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size,num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        
        # Channel attention
        self.channel_attention = ChannelAttention(in_chans)

        # Residual 3D convolution block
        self.residual_3d_conv = Residual3DConv(in_chans, in_chans)
        #self.multi_head_self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, tubelet_size * patch_size * patch_size * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: B, C, T, H, W
        x: B, L, D
        """
        p = self.patch_embed.patch_size[0]
        tub = self.patch_embed.tubelet_size
        x = rearrange(imgs, 'b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)', tub=tub, p=p, q=p)

        return x

    def unpatchify(self, x):
        """
        x: B, L, D
        imgs: B, C, T, H, W
        """
        p = self.patch_embed.patch_size[0]
        num_p = self.patch_embed.img_size[0] // p
        tub = self.patch_embed.tubelet_size
        imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)', h=num_p, w=num_p, tub=tub, p=p, q=p)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        
        #x = self.residual_3d_conv(x)
        #x = self.channel_attention(x)
        
        # embed patches
        x = self.patch_embed(x)
        #x = self.multi_head_self_attention(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
       
        x = self.decoder_embed(x)
        

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: B, C, T, H, W
        target: B, L, D
        pred: B, L, D
        mask: B, L. 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        #loss = (pred - target) ** 2
        loss = F.huber_loss(pred, target, reduction='none', delta=1.0)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
        
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import glob
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import yaml
from tqdm import tqdm
import logging
# Set up logging
logging.basicConfig(filename='pretrain.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import torch
import numpy as np




def preprocess_image(image, means, stds):
    means = np.array(means).reshape(-1, 1, 1)
    stds = np.array(stds).reshape(-1, 1, 1)
    # normalize image
    normalized = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized



import torchvision.transforms as transforms


class MultibandTiffDataset(Dataset):
    def __init__(self, directory_paths, bands, data_mean, data_std, random_cropping=False):
        self.file_paths = []
        for directory in directory_paths:
            self.file_paths.extend(glob.glob(os.path.join(directory, '*.tif')))
        self.bands = bands
        self.data_mean = torch.tensor(data_mean)
        self.data_std = torch.tensor(data_std)
        self.random_cropping = random_cropping
        if self.random_cropping:
            self.random_crop = transforms.RandomCrop(224)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as src:
            img = np.stack([src.read(band) for band in self.bands], axis=0)  # Shape: (num_bands, H, W)


        # Preprocess image
        img = preprocess_image(img, self.data_mean, self.data_std)
        
        # Random cropping if enabled
        if self.random_cropping:
            img = self.random_crop(img)
        
        
        return img


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm



def train_epoch(model, dataloader, optimizer, device, mask_ratio):
    model.train()
    running_loss = 0.0
    for imgs in tqdm(dataloader, desc="Training", unit="batch", leave=False):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss, pred, mask = model(imgs, mask_ratio=mask_ratio)  # Unpack the output tuple
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)
    

    

def validate_epoch(model, dataloader, device, mask_ratio=0.75):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs, mask_ratio=mask_ratio)  # Unpack the output tuple
            total_loss += loss.item()
    return total_loss / len(dataloader)



# read model config
model_cfg_path = "prithvi_nsra_all.yaml"   #for xin: need file needs editing based on your problem
with open(model_cfg_path) as f:
    model_config = yaml.safe_load(f)

model_args, train_params = model_config["model_args"], model_config["train_params"]

# Initialize model, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskedAutoencoderViT(**model_args).to(device)

# Set optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999))


# Data paths
train_directories = ['train_latest_water_elev_2022_new_new_filtered/train_latest_new_new/images_flowdir_arc/', 
'train_latest_water_elev_2019_new_filtered/train_latest_new/images_flowdir_arc/', 
'train_latest_water_elev_2008_new_filtered/train_latest_new/images_flowdir_arc/',
'train_latest_water_elev_2013_new_filtered/train_latest_new/images_flowdir_arc/']  # List of paths to train directories: diff for xin
val_directories = ['train_latest_water_elev_2013_new_filtered/train_latest_new/images_flowdir_arc/']    # List of paths to validation directories diff for xin

# Create datasets and dataloaders
train_dataset = MultibandTiffDataset(
    directory_paths=train_directories,
    bands=train_params['bands'],
    data_mean=train_params['data_mean'],
    data_std=train_params['data_std'],
    random_cropping=False
)

val_dataset = MultibandTiffDataset(
    directory_paths=val_directories,
    bands=train_params['bands'],
    data_mean=train_params['data_mean'],
    data_std=train_params['data_std'],
    random_cropping=False
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=7, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=7, pin_memory=True)

num_epochs = 150
# Scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    total_steps=len(train_loader) * num_epochs,
    pct_start=0.3,
    anneal_strategy='cos'
)



for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, mask_ratio=train_params['mask_ratio'])
    val_loss = validate_epoch(model, val_loader, device, mask_ratio=train_params['mask_ratio'])
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    scheduler.step()

# Save the trained weights
torch.save(model.state_dict(), 'masked_autoencoder.pth')