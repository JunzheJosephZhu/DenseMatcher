# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from detectron2.config import instantiate
import time

def instantiate_odise(cfg):
    start = time.time()
    backbone = instantiate(cfg.backbone)
    cfg.sem_seg_head.input_shape = backbone.output_shape()
    cfg.sem_seg_head.pixel_decoder.input_shape = backbone.output_shape()
    cfg.backbone = backbone
    print(time.time() - start, "instantiated backbone")
    start = time.time()
    model = instantiate(cfg)
    print(time.time() - start, "instantiated model")
    return model
