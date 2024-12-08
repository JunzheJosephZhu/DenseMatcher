# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import os.path as osp
from collections import defaultdict
from typing import List
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from fvcore.common.checkpoint import Checkpointer, _strip_prefix_if_present
from odise.utils.file_io import PathManager
import torch
import time
import os

def _longest_common_prefix(names: List[str]) -> str:
    """
    ["abc.zfg", "abc.zef"] -> "abc."
    """
    names = [n.split(".") for n in names]
    m1, m2 = min(names), max(names)
    ret = []
    for a, b in zip(m1, m2):
        if a == b:
            ret.append(a)
        else:
            # break for the first non-matching element
            # Fixing BUG in detectron2
            break
    ret = ".".join(ret) + "." if len(ret) else ""
    return ret


def group_by_prefix(names):
    grouped_names = defaultdict(list)

    for name in names:
        grouped_names[name.split(".")[0]].append(name)

    return grouped_names


class ODISECheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables
        )
        self.path_manager = PathManager

    def _load_model(self, checkpoint):

        if hasattr(self.model, "preprocess_state_dict"):
            self.logger.info("Preprocessing model state_dict")
            checkpoint["model"] = self.model.preprocess_state_dict(checkpoint["model"])
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )

        # for non-caffe2 models, use standard ways to load it
        incompatible = super(DetectionCheckpointer, self)._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        for k in incompatible.unexpected_keys[:]:
            # Ignore unexpected keys about cell anchors. They exist in old checkpoints
            # but now they are non-persistent buffers and will not be in new checkpoints.
            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)

        removed_keys = []
        self.logger.warn(f"ODISE: found {len(incompatible.missing_keys)} missing keys!")
        if hasattr(self.model, "ignored_state_dict"):
            ignored_keys = set(self.model.ignored_state_dict().keys())
        else:
            ignored_keys = set()
        for k in incompatible.missing_keys[:]:
            # Ignore clip.clip since it's fixed in DALLE2 decoder
            # Ignore text_encoder.encoder since it's fixed in Imagen
            if k in ignored_keys:
                incompatible.missing_keys.remove(k)
                removed_keys.append(k)
        if len(removed_keys) > 0:
            prefix_list = [
                _longest_common_prefix(grouped_names)
                for grouped_names in group_by_prefix(removed_keys).values()
            ]
            self.logger.warn(
                "Keys with prefix are removed from state_dict:\n" + ",".join(prefix_list)
            )

            self.logger.warn(
                f"Removed {len(removed_keys)} ignored_state_dict keys from missing_keys"
            )

        return incompatible

    @staticmethod
    def has_checkpoint_in_dir(save_dir) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = osp.join(save_dir, "last_checkpoint")
        return osp.exists(save_file)

class BackboneCheckpointer(Checkpointer):
    def _load_model(self, checkpoint):
        # the checkpoint is for ODISE, not the LDMImplicitCaptioner backbone only, so we need to remove the outer layer of prefix and only keep the backbone keys
        state_dict = checkpoint.pop("model")
        for key in list(state_dict.keys()):
            if not key.startswith("backbone."):
                state_dict.pop(key)
        _strip_prefix_if_present(state_dict, prefix="backbone.")
        incompatible = super()._load_model({"model": state_dict})
        # ignore keys, copied from ODISECheckpointer. Backbone also has an ignored_state_dict() method
        removed_keys = []
        self.logger.warning(f"ODISE: found {len(incompatible.missing_keys)} missing keys!")
        if hasattr(self.model, "ignored_state_dict"):
            ignored_keys = set(self.model.ignored_state_dict().keys())
        else:
            ignored_keys = set()
        for k in incompatible.missing_keys[:]:
            # Ignore clip.clip since it's fixed in DALLE2 decoder
            # Ignore text_encoder.encoder since it's fixed in Imagen
            if k in ignored_keys:
                incompatible.missing_keys.remove(k)
                removed_keys.append(k)
        if len(removed_keys) > 0:
            prefix_list = [
                _longest_common_prefix(grouped_names)
                for grouped_names in group_by_prefix(removed_keys).values()
            ]
            self.logger.warning(
                "Keys with prefix are removed from state_dict:\n" + ",".join(prefix_list)
            )

            self.logger.warning(
                f"Removed {len(removed_keys)} ignored_state_dict keys from missing_keys"
            )
        for k in incompatible.missing_keys[:]:
            if "feature_projections" in k:
                incompatible.missing_keys.remove(k)

        for item in incompatible.incorrect_shapes[:]:
            if "feature_projections" in item[0]:
                incompatible.incorrect_shapes.remove(item)
        return incompatible

class LdmCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables
        )
        self.path_manager = PathManager

    def _load_file(self, f: str):
        # not necessary, using cuda will slow down _load_file but speed up _load_model
        start = time.time()
        ckpt = torch.load(f, map_location=torch.device("cuda") if os.environ.get("INFERENCE", False) else torch.device("cpu")) 
        print(time.time() - start, "loaded ldm ckpt file")
        return ckpt
    
    def _load_model(self, checkpoint):
        # rename the keys in checkpoint
        start = time.time()
        checkpoint["model"] = checkpoint.pop("state_dict")
        incompatible = super()._load_model(checkpoint)
        for k in incompatible.unexpected_keys[:]:
            if "model_ema" in k or "cond_stage_model" in k:
                incompatible.unexpected_keys.remove(k)
        print(time.time() - start, "loaded state dict into ldm")
        return incompatible