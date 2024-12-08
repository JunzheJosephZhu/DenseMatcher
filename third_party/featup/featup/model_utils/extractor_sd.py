import itertools
from contextlib import ExitStack
import torch
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
import numpy as np
import torch.nn.functional as F
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.config import LazyCall as L
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.visualizer import ColorMode, random_color

from odise import model_zoo
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference
from odise.checkpoint.odise_checkpointer import ODISECheckpointer


def load_model(img_size, diffusion_ver, num_timesteps, config_path="Panoptic/odise_label_coco_50e.py", seed=42, block_indices=(2,5,8,11), decoder_only=True, encoder_only=False, resblock_only=False):
    cfg = model_zoo.get_config(config_path, trained=True)

    cfg.model.backbone.feature_extractor.init_checkpoint = "sd://"+diffusion_ver
    cfg.model.backbone.feature_extractor.steps = (num_timesteps,)
    cfg.model.backbone.feature_extractor.unet_block_indices = block_indices
    cfg.model.backbone.feature_extractor.encoder_only = encoder_only
    cfg.model.backbone.feature_extractor.decoder_only = decoder_only
    cfg.model.backbone.feature_extractor.resblock_only = resblock_only
    cfg.model.overlap_threshold = 0
    if img_size > 512:
        cfg.model.backbone.backbone_in_size = (512, 512) # single crop's size. If tuple use slide inference
        cfg.model.backbone.slide_training = True
    else:
        cfg.model.backbone.backbone_in_size = img_size # if int, don't use slide inference
        cfg.model.backbone.slide_training = False
    
    seed_all_rng(seed)

    model = instantiate_odise(cfg.model) # idk why, loading CLIP slows this the fuck down
    print('instantiated odise, start loading weights')
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    return model.backbone
