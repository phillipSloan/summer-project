import numpy as np
import torch
import re

from monai.utils.enums import TransformBackends
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

class SplitDim(Transform):
    """
    Given an image of size X along a certain dimension, return a list of length X containing
    images. Useful for converting 3D images into a stack of 2D images, splitting multichannel inputs into
    single channels, for example.
    Note: `torch.split`/`np.split` is used, so the outputs are views of the input (shallow copy).
    Args:
        dim: dimension on which to split
        keepdim: if `True`, output will have singleton in the split dimension. If `False`, this
            dimension will be squeezed.
        update_meta: whether to update the MetaObj in each split result.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dim: int = -1, keepdim: bool = True, update_meta=True) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.update_meta = update_meta

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply the transform to `img`.
        """
        n_out = img.shape[self.dim]
        if n_out <= 1:
            raise RuntimeError(f"Input image is singleton along dimension to be split, got shape {img.shape}.")
        if isinstance(img, torch.Tensor):
            outputs = list(torch.split(img, 1, self.dim))
        else:
            outputs = np.split(img, n_out, self.dim)
        for idx, item in enumerate(outputs):
            if not self.keepdim:
                outputs[idx] = item.squeeze(self.dim)
            if self.update_meta and isinstance(img, MetaTensor):
                if not isinstance(item, MetaTensor):
                    item = MetaTensor(item, meta=img.meta)
                if self.dim == 0:  # don't update affine if channel dim
                    continue
                ndim = len(item.affine)
                shift = torch.eye(ndim, device=item.affine.device, dtype=item.affine.dtype)
                shift[self.dim - 1, -1] = idx
                item.affine = item.affine @ shift
        return outputs

class SplitDimd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Optional[Sequence[str]] = None,
        dim: int = 0,
        keepdim: bool = True,
        update_meta: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes: the postfixes to construct keys to store split data.
                for example: if the key of input data is `pred` and split 2 classes, the output
                data keys will be: pred_(output_postfixes[0]), pred_(output_postfixes[1])
                if None, using the index number: `pred_0`, `pred_1`, ... `pred_N`.
            dim: which dimension of input image is the channel, default to 0.
            keepdim: if `True`, output will have singleton in the split dimension. If `False`, this
                dimension will be squeezed.
            update_meta: if `True`, copy `[key]_meta_dict` for each output and update affine to
                reflect the cropped image
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.output_postfixes = output_postfixes
        self.splitter = SplitDim(dim, keepdim, update_meta)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            rets = self.splitter(d[key])
            postfixes: Sequence = list(range(len(rets))) if self.output_postfixes is None else self.output_postfixes
            if len(postfixes) != len(rets):
                raise ValueError(f"count of splits must match output_postfixes, {len(postfixes)} != {len(rets)}.")
            for i, r in enumerate(rets):
                split_key = f"{key}_{postfixes[i]}"
                if split_key in d:
                    raise RuntimeError(f"input data already contains key {split_key}.")
                d[split_key] = r
        return d