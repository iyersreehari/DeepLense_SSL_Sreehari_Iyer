# adapted from
#     https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
#     https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import torch
import torch.nn as nn
from .vit import VisionTransformer
from .swin import SwinTransformer
from .fan import FAN
from .convvit import ConvVisionTransformer
from timm.models.resnet import resnet50, resnet18
from functools import partial
from typing import Union, Tuple, List, Optional

def vit_tiny(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=192, 
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )

def vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        return_all_tokens: bool = True,
        masked_im_modeling: bool = True,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                attn_drop = 0.2,
                # attn_drop = 0.,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = return_all_tokens,
                masked_im_modeling = masked_im_modeling,
                **kwargs
            )

def vit_mlp_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                attn_drop = 0.2,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_fc_norm = True,
                head_drop_rate = 0.2,
                head = nn.Sequential(nn.LayerNorm(384), nn.Dropout(p=0.2), nn.ReLU(),\
                                     nn.Linear(384, 8*384), nn.LayerNorm(8*384), nn.Dropout(p=0.2),\
                                     nn.ReLU(),\
                                     # nn.Linear(8*384, 8*384), nn.LayerNorm(8*384), nn.Dropout(p=0.2),\
                                     # nn.ReLU(),\
                                     nn.Linear(8*384, 384)),
                **kwargs
            )
    
def vit_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        return_all_tokens: bool = True,
        masked_im_modeling: bool = True,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                attn_drop = 0.2,
                # attn_drop = 0.,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = return_all_tokens,
                masked_im_modeling = masked_im_modeling,
                **kwargs
            )

def vit_mlp_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                head = nn.Sequential(nn.Linear(768, 192), nn.GELU(),\
                                     nn.Linear(192, 768)),
                use_fc_norm = True,
                head_drop_rate = 0.,
                **kwargs
            )

def channel_vit_tiny(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                patch_embedding_type="channel_vit",
                embed_dim=192, 
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = False,
                masked_im_modeling = False,
                **kwargs
            )

def channel_vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                patch_embedding_type="channel_vit", 
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                attn_drop = 0.,
                # attn_drop = 0.,
                drop_path_rate = 0.,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = False,
                masked_im_modeling = False,
                **kwargs
            )

def channel_vit_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                patch_embedding_type="channel_vit", 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )
def swin_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        window_size: int = 7,
        use_dense_prediction: bool = True,
        ape: bool = False,
        **kwargs
    ):
    return SwinTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=96, 
                depths=[2, 2, 18, 2], 
                window_size=window_size,
                num_heads=[3 , 6, 12, 24], 
                mlp_ratio=4.,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                use_dense_prediction = use_dense_prediction,
                ape = ape,
                patch_norm = False,
                attn_drop = 0.,
                # attn_drop = 0.,
                drop_path_rate = 0.,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                **kwargs
            )
def swin_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        window_size: int = 7,
        use_dense_prediction: bool = True,
        ape: bool = False,
        **kwargs
    ):
    return SwinTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=128, 
                depths=[2, 2, 18, 2], 
                window_size=window_size,
                num_heads=[3,6,12,24], 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                use_dense_prediction = use_dense_prediction,
                ape = ape,
                attn_drop = 0.2,
                # attn_drop = 0.,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                **kwargs
            )

def fan_vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        **kwargs):
    return FAN(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                num_heads=8,
                eta=1.0,
                tokens_norm=True,
                sharpen_attn=False,
                use_se_mlp=False,
                sr_ratio=[1] * 12,
                mlp_ratio=4.,
                qkv_bias=False, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                attn_drop = 0.,
                # attn_drop = 0.,
                drop_path_rate = 0.,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                **kwargs
            )

def conv_vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool=False,
        **kwargs):
    return ConvVisionTransformer(
                img_size = image_size,
                in_chans = input_channels,
                patch_size=patch_size, 
                num_classes = 0,
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                attn_drop_rate = 0.,
                # attn_drop = 0.,
                drop_path_rate = 0.,
                drop_rate = 0.,
                local_up_to_layer = 10,
                locality_strength = 1.,
                use_dense_prediction = use_dense_prediction,
                **kwargs
            )

# def cvt_13(
#         image_size: Union[int, Tuple[int, int]],
#         input_channels: int,
#         patch_size: Union[int, Tuple[int, int]] = 16,
#         ):
#     return ConvolutionalVisionTransformer(
#             input_channels = 3,
#             patch_size: List[int],
#             num_classes: int = 0, 
#             num_stages = 3,
#             embed_dim: int = 768,
#             depth: int = 12,
#             num_heads: int = 12,
#             mlp_ratio: int = 4.,
#             qkv_bias: bool = False,
#             # qk_norm: bool = False,
#             attn_drop: float = 0.2,
#             drop_path_rate: float = 0.1,
#             drop_rate: float = 0.,
#             class_token: bool = True,
#             use_fc_norm: bool = False,
#             head: nn.Module = None,
#         )
# def fan_small_12_p16_224_se_attn(pretrained=False, **kwargs):
#     depth = 12
#     sr_ratio = [1] * (depth//2) + [1] * (depth//2)
#     model_kwargs = dict(
#         patch_size=16, embed_dim=384, depth=depth, num_heads=8, eta=1.0, tokens_norm=True, sharpen_attn=False,se_mlp=True, **kwargs)
#     model = _create_fan('fan_small_12_p16_224', pretrained=pretrained, sr_ratio=sr_ratio, **model_kwargs)
    # return model 

    
def Backbone(
        arch: str,
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: Optional[bool] = False,
        return_all_tokens: bool = False,
        masked_im_modeling: bool = False,
        window_size: Optional[int] = None,
        ape: Optional[bool] = False,
    ):
    if arch.lower() == "resnet50":
        net = resnet50(in_chans=input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.)
        net.fc = nn.Identity()
        net.embed_dim = net.num_features
        net.return_all_tokens = return_all_tokens
        return net
    if arch.lower() == "resnet18":
        net = resnet18(in_chans=input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.)
        net.fc = nn.Identity()
        net.embed_dim = net.num_features
        net.return_all_tokens = return_all_tokens
        return net
    if arch.lower() == "vit_tiny":
        return vit_tiny(image_size, input_channels, patch_size)
    elif arch.lower() == "vit_small":
        return vit_small(image_size, input_channels, patch_size, use_dense_prediction, return_all_tokens, masked_im_modeling)
    elif arch.lower() == "fan_vit_small":
        return fan_vit_small(image_size, input_channels, patch_size, use_dense_prediction)
    elif arch.lower() == "conv_vit_small":
        return conv_vit_small(image_size, input_channels, patch_size, use_dense_prediction)
    elif arch.lower() == "vit_mlp_small":
        return vit_mlp_small(image_size, input_channels, patch_size)
    elif arch.lower() == "swin_small":
        return swin_small(image_size, input_channels, patch_size, window_size, use_dense_prediction, ape)
    elif arch.lower() == "swin_base":
        return swin_base(image_size, input_channels, patch_size, window_size, use_dense_prediction, ape)
    elif arch.lower() == "vit_base":
        return vit_base(image_size, input_channels, patch_size, use_dense_prediction, return_all_tokens, masked_im_modeling)
    elif arch.lower() == "vit_mlp_base":
        return vit_mlp_base(image_size, input_channels, patch_size)
    if arch.lower() == "channel_vit_tiny":
        return channel_vit_tiny(image_size, input_channels, patch_size)
    elif arch.lower() == "channel_vit_small":
        return channel_vit_small(image_size, input_channels, patch_size, use_dense_prediction)
    elif arch.lower() == "channel_vit_base":
        return channel_vit_base(image_size, input_channels, patch_size)
    else:
        print(f"Backbone architecture specified as {arch} is not implemented. Exiting.")
        sys.exit(1)
