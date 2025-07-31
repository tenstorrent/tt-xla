# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
from torchvision import transforms
import requests
from PIL import Image
import PIL.Image as pil
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from ....tools.utils import get_file


class MonoDepth2(torch.nn.Module):
    def __init__(self, encoder, depth_decoder):
        super().__init__()
        self.encoder = encoder
        self.depth_decoder = depth_decoder

    def forward(self, input):
        features = self.encoder(input)
        outputs = self.depth_decoder(features)
        return outputs[("disp", 0)]


def load_model(variant):

    encoder_path = get_file("test_files/pytorch/monodepth/mono_640x192/encoder.pth")
    depth_decoder_path = get_file("test_files/pytorch/monodepth/mono_640x192/depth.pth")

    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
    depth_decoder.load_state_dict(loaded_dict)

    model = MonoDepth2(encoder, depth_decoder)
    model.eval()

    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]

    return model, feed_height, feed_width


def load_input(feed_height, feed_width):

    image_file = get_file(
        "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/assets/test_image.jpg"
    )
    input_image = Image.open(image_file).convert("RGB")
    original_width, original_height = input_image.size
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    return input_tensor, original_width, original_height
