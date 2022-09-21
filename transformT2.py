# This file is a part of StarNet code.
# https://github.com/nekitmm/starnet
# 
# StarNet is a neural network that can remove stars from images leaving only background.
# 
# Throughout the code all input and output images are 8 bits per channel tif images.
# This code in original form will not read any images other than these (like jpeg, etc), but you can change that if you like.
# 
# Copyright (c) 2018 Nikita Misiura
# http://www.astrobin.com/users/nekitmm/
# 
# This code is distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, express or implied.
# Please review LICENSE file before use.

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.INFO)
from PIL import Image as img
from starnet_v1_TF2 import StarNet

import tifffile as tiff
                                       # and changing this will force you to train the net anew.
def transform(imageName, stride):
    starnet = StarNet(mode = 'RGB', window_size = 512, stride = 128)
    starnet.load_model('./weights', './history',inference=True)
    starnet.transform(imageName, "result.tif")