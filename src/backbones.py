"""Frozen feature extractors.

Two pre-trained encoders are used with their weights frozen:
  * a Wide Residual Network (image-level global-average-pooled descriptor and
    a dense token grid from two intermediate stages), and
  * a self-supervised Vision Transformer (CLS descriptor and patch tokens).

No encoder is trained or fine-tuned. Weights are pulled from their public
model hubs on first use.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Backbones:
    """Loads both encoders once and exposes a batched embedding call."""

    def __init__(self, device=None):
        self.device = device or _device()
        self.vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(self.device).eval()
        self.cnn = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V2").to(self.device).eval()
        self._feat = {}
        self.cnn.layer2.register_forward_hook(lambda m, i, o: self._feat.__setitem__("l2", o))
        self.cnn.layer3.register_forward_hook(lambda m, i, o: self._feat.__setitem__("l3", o))

    def _to_batch(self, imgs):
        # imgs: list of HxW float arrays already resized to the model input size
        x = torch.from_numpy(np.stack(imgs)).float().unsqueeze(1).repeat(1, 3, 1, 1)
        return (x - _MEAN) / _STD

    @torch.no_grad()
    def embed(self, imgs, batch_size=32, want_patch=True):
        """Return a dict of arrays:
            vit_cls   [N, 768]            image-level ViT descriptor
            vit_patch [N, 256, 768]       dense ViT tokens (optional)
            cnn_img   [N, C]              image-level CNN descriptor (GAP of two stages)
        """
        out = {"vit_cls": [], "vit_patch": [], "cnn_img": []}
        for i in range(0, len(imgs), batch_size):
            xb = self._to_batch(imgs[i:i + batch_size]).to(self.device)
            f = self.vit.forward_features(xb)
            out["vit_cls"].append(f["x_norm_clstoken"].cpu().numpy())
            if want_patch:
                out["vit_patch"].append(f["x_norm_patchtokens"].cpu().numpy())
            _ = self.cnn(xb)
            l2, l3 = self._feat["l2"], self._feat["l3"]
            g = torch.cat([l2.mean((-2, -1)), l3.mean((-2, -1))], 1)
            out["cnn_img"].append(g.cpu().numpy())
        res = {k: np.concatenate(v, 0) for k, v in out.items() if v}
        return res
