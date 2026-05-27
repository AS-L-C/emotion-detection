import shutil
import sys
from math import floor
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from huggingface_hub import hf_hub_download
from transformers import AutoModel

PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from CBraMod.models.cbramod import CBraMod
from UniShape.models.unishapemodel_finetune import UniShapeModel


def get_avail_models():
    return {"reve", "cbramod", "unishape"}


def get_model(
    model,
    channels,
    data_spec,
    n_classes,
    device,
):
    get_fcns = {"reve": get_reve, "cbramod": get_cbramod, "unishape": get_unishape}
    assert model in get_fcns, (
        f"Unrecognized model '{model}'. Available models: {get_fcns.keys()}"
    )
    Model = get_fcns[model](channels, data_spec, n_classes, device)
    return Model.to(device)


def get_reve(
    channels,
    data_spec,
    n_classes,
    device,
    base=True,
    freeze_backbone_flag=True,
    apply_input_linear=True,
):
    def get_npatches(n_times, sf):
        window_size = 1 * sf
        overlap = round(0.1 * window_size)
        stride = window_size - overlap
        return floor((n_times - window_size) / stride) + 1

    pretrained_model = "reve-base" if base else "reve-large"
    reve_model = AutoModel.from_pretrained(
        "brain-bzh/" + pretrained_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    position_model = AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
    )

    positions = position_model(channels)
    n_patches = get_npatches(data_spec["n_times"], data_spec["sf"])
    Model = ReveModel(
        base_model=reve_model,
        positions=positions,
        n_channels=data_spec["n_channels"],
        n_classes=n_classes,
        n_patches=n_patches,
        freeze_backbone_flag=freeze_backbone_flag,
        apply_input_linear=apply_input_linear,
    )
    return Model


def get_cbramod(
    channels,
    data_spec,
    n_classes,
    device,
    freeze_backbone_flag=True,
):
    def get_weights(CBRAMOD_DIR=PROJECT_DIR / "code" / "CBraMod"):
        weights_dir = CBRAMOD_DIR / "pretrained_weights"
        target_path = weights_dir / "pretrained_weights.pth"

        if not target_path.exists() or target_path.stat().st_size == 0:
            downloaded_path = hf_hub_download(
                repo_id="weighting666/CBraMod",
                filename="pretrained_weights.pth",
                repo_type="model",
            )

            shutil.copyfile(downloaded_path, target_path)
            print("Downloaded to:", target_path)
        else:
            print("Weights already exist:", target_path)
        return target_path

    weights_path = get_weights()
    n_patches = data_spec["n_times"] // data_spec["sf"]  # Number of 1-sec patches

    # Init base model
    base_model = CBraMod()
    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )
    base_model.load_state_dict(state_dict)

    # Init classification model
    Model = CbraModel(
        base_model=base_model,
        n_channels=data_spec["n_channels"],
        n_classes=n_classes,
        n_patches=n_patches,
        patch_size=data_spec["sf"],
        freeze_backbone_flag=freeze_backbone_flag,
    )
    return Model


def get_unishape(
    channels,
    data_spec,
    n_classes,
    device,
    base=True,
    freeze_backbone_flag=False,
    print_info=False,
):
    def load_pretrained_model(weights_path, model, device="cpu"):
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )

        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        cleaned_state_dict = {}

        for k, v in state_dict.items():
            # Remove wrapper prefix from pretrained backbone
            if k.startswith("backbone.") and not k.startswith("backbone.fc."):
                k = k[len("backbone.") :]

            # Skip classifier heads
            if k.startswith("fc.") or k.startswith("backbone.fc."):
                continue

            # Skip positional encoding if incompatible / regenerated
            if k == "transformer_enc.pos_encoder.pe":
                continue

            cleaned_state_dict[k] = v

        model_state_dict = model.state_dict()

        filtered_state_dict = {
            k: v
            for k, v in cleaned_state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        msg = model.load_state_dict(filtered_state_dict, strict=False)

        if print_info:
            print("Missing keys:", msg.missing_keys)
            print("Unexpected keys:", msg.unexpected_keys)
            print(f"Loaded {len(filtered_state_dict)} tensors.")

        return model

    # Weights of pretrained model
    checkpoint_path = (
        PROJECT_DIR
        / "code/UniShape/pretrained_model_ckpt/unishape_checkpoint_finetune.pth"
    )

    # Model hyperparameters
    config = SimpleNamespace(
        hidden_dim=128,
        window_emb_dim=128,
        window_size=16,
        stride=16,
        shape_ratio=0.6,
        scale_len=5,
    )

    # Backbone model
    backbone = UniShapeModel(
        config=config,
        series_size=data_spec["n_times"],
        in_channels=config.hidden_dim,
        window_emb_dim=config.window_emb_dim,
        out_channels=n_classes,
        window_size=config.window_size,
        stride=config.stride,
        pre_training=False,
        shape_alpha=0.01,
        shape_ratio=config.shape_ratio,
        scale_len=config.scale_len,
    )
    backbone = load_pretrained_model(checkpoint_path, backbone, device=device)

    # Classifier model
    Model = UniShapeLogitsOnly(
        backbone=backbone,
        n_classes=n_classes,
        freeze_backbone_flag=freeze_backbone_flag,
        series_size=data_spec["n_times"],
    )
    return Model


class ReveModel(nn.Module):
    def __init__(
        self,
        base_model,
        positions,
        n_channels,
        n_classes,
        n_patches,
        p_drop=0.1,
        freeze_backbone_flag=True,
        apply_input_linear=False,
    ):
        super().__init__()
        self.p_drop = p_drop
        self.positions = positions
        self.out_dim = n_channels * n_patches * base_model.embed_dim
        self.input_projection = (
            nn.Linear(n_channels, n_channels)
            if apply_input_linear
            else nn.Identity(n_channels, n_channels)
        )
        self.base_model = base_model
        self.output_projection = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(self.out_dim),
            torch.nn.Dropout(self.p_drop),
            torch.nn.Linear(self.out_dim, self.out_dim),
            nn.ELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ELU(),
            nn.Dropout(self.p_drop),
            torch.nn.Linear(self.out_dim, n_classes),
        )
        self.freeze_backbone(freeze_backbone_flag)

    def freeze_backbone(self, freeze_backbone_flag):
        if freeze_backbone_flag:
            for p in self.base_model.parameters():
                p.requires_grad = False

    def apply_linear(self, x):
        x = self.input_projection(rearrange(x, "b c t -> b t c"))
        return rearrange(x, "b t c -> b c t")

    def forward(self, x):
        pos = repeat(
            self.positions, "chs coords -> batch chs coords", batch=x.shape[0]
        ).to(x.device)
        x = self.base_model(self.apply_linear(x), self.apply_linear(pos))
        x = self.output_projection(x)
        return x


class CbraModel(nn.Module):
    def __init__(
        self,
        base_model,
        n_channels,
        n_classes,
        n_patches,
        patch_size,
        p_drop=0.1,
        freeze_backbone_flag=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.p_drop = p_drop
        self.base_model = base_model
        self.base_model.proj_out = nn.Identity()
        self.output_projection = self.init_output_proj()
        self.freeze_backbone(freeze_backbone_flag)

    def init_output_proj(self):
        return nn.Sequential(
            Rearrange("b c s p -> b (c s p)"),
            nn.Linear(
                self.n_channels * self.n_patches * self.patch_size,
                self.n_patches * self.patch_size,
            ),
            nn.ELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.n_patches * self.patch_size, self.patch_size),
            nn.ELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.patch_size, self.n_classes),
        )

    def freeze_backbone(self, freeze_backbone_flag):
        if freeze_backbone_flag:
            for p in self.base_model.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = rearrange(
            x,
            "b c (n_patches patch_size) -> b c n_patches patch_size ",
            n_patches=self.n_patches,
            patch_size=self.patch_size,
        )
        x = self.base_model(x)
        x = self.output_projection(x)
        return x


class UniShapeLogitsOnly(nn.Module):
    def __init__(
        self,
        backbone,
        n_classes,
        freeze_backbone_flag,
        series_size,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone(freeze_backbone_flag)

    def freeze_backbone(self, freeze_backbone_flag):
        if freeze_backbone_flag:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = x.float()
        B, C, T = x.shape

        # UniShape is univariate at the raw input level.
        # Treat each channel as an independent univariate time series.
        x_uni = x.reshape(B * C, 1, T)

        scale_idx = self.backbone.scale_len - 1

        if scale_idx < 0 or scale_idx >= 5:
            raise ValueError(
                f"Invalid scale_len={self.backbone.scale_len}. Expected one of 1, 2, 3, 4, 5."
            )

        if scale_idx == 4:
            x_embed = self.backbone.unit_scale_list[scale_idx](x_uni)
        else:
            x_embed = self.backbone.unit_scale_list_finetune[scale_idx](x_uni)

        cls_incep_token_list = self.backbone.inceptime_token(
            x_embed.permute(0, 2, 1)
        ).permute(0, 2, 1)

        cls_incep_token_list = self.backbone.drop_token(
            self.backbone.layer_norm_inc(cls_incep_token_list)
        )
        cls_incep_token_list = self.backbone.act_gelu_inc(cls_incep_token_list)

        attn_x_score = self.backbone.attention_head(cls_incep_token_list)
        attn_shape_embds = cls_incep_token_list * attn_x_score

        cls_tokens = torch.mean(attn_shape_embds, dim=1).unsqueeze(1)

        x_embed = x_embed.squeeze(1)

        trans_enc_class_token, _shape_tokens = self.backbone.transformer_enc(
            x_embed,
            cls_token_in=cls_tokens,
        )

        channel_logits = self.backbone.fc(trans_enc_class_token)

        # Average logits over channels.
        logits = channel_logits.reshape(B, C, -1).mean(dim=1)

        return logits
