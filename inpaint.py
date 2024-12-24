import argparse
import os
import warnings
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pickle
import PIL.Image
import torch
from PIL import Image
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGAN
from lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)

warnings.filterwarnings("ignore")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="migan-512", help="One of [migan-256, migan-512, comodgan-256, comodgan-512]")
    parser.add_argument("--model-path", type=str, default="models/migan_512_places2.pt", help="Saved model path.")
    # parser.add_argument("--images-dir", type=Path, help="Path to images directory.", required=True)
    # parser.add_argument("--masks-dir", type=Path, help="Path to masks directory.", required=True)
    # parser.add_argument("--invert-mask", action="store_true", help="Invert mask? (make 0-known, 1-hole)")
    # parser.add_argument("--output-dir", type=Path, help="Output directory.", required=True)
    #parser.add_argument("--device", type=str, help="Device.", default="cuda")
    return parser.parse_args()

class Inpainting:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model_name == "migan-256":
            resolution = 256
            model = MIGAN(resolution=256)
        elif args.model_name == "migan-512":
            resolution = 512
            model = MIGAN(resolution=512)
        elif args.model_name == "comodgan-256":
            resolution = 256
            comodgan_mapping = CoModGANMapping(num_ws=14)
            comodgan_encoder = CoModGANEncoder(resolution=resolution)
            comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
            model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
        elif args.model_name == "comodgan-512":
            resolution = 512
            comodgan_mapping = CoModGANMapping(num_ws=16)
            comodgan_encoder = CoModGANEncoder(resolution=resolution)
            comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
            model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
        else:
            raise Exception("Unsupported model name.")

        model.load_state_dict(torch.load(args.model_path))
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.resolution = resolution

    def resize(self, image, max_size, interpolation=Image.BICUBIC):
        w, h = image.size
        if w > max_size or h > max_size:
            resize_ratio = max_size / w if w > h else max_size / h
            image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
        return image
    def preprocess(self, img: Image, mask: Image, resolution: int) -> torch.Tensor:
        img = img.resize((resolution, resolution), Image.BICUBIC)
        mask = mask.resize((resolution, resolution), Image.NEAREST)
        img = np.array(img)
        mask = np.array(mask)[:, :, np.newaxis] // 255
        img = torch.Tensor(img).float() * 2 / 255 - 1
        mask = torch.Tensor(mask).float()
        img = img.permute(2, 0, 1).unsqueeze(0)
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        x = torch.cat([mask - 0.5, img * mask], dim=1)
        return x
    def inpaint(self, img, mask):
        img_resized = self.resize(img, max_size=self.resolution)
        mask_resized = self.resize(mask, max_size=self.resolution)
        x = self.preprocess(img_resized, mask_resized, self.resolution)
        x.to(self.device)
        with torch.no_grad():
            result_image = self.model(x)[0]
        result_image = (result_image * 0.5 + 0.5).clamp(0, 1) * 255
        result_image = result_image.to(torch.uint8).permute(1, 2, 0).detach().to("cpu").numpy()

        result_image = cv2.resize(result_image, dsize=img_resized.size, interpolation=cv2.INTER_CUBIC)
        mask_resized = np.array(mask_resized)[:, :, np.newaxis] // 255
        composed_img = img_resized * mask_resized + result_image * (1 - mask_resized)
        composed_img = Image.fromarray(composed_img)
        #composed_img.save(f"out.png")
        return composed_img
         
if __name__ == '__main__':
    args = get_args()
    print(args)
    inpaint = Inpainting(args)
    img = Image.open("d:\\work\\MI-GAN\\examples\\places2_512_object\\images\\1.png")
    mask = Image.open("mask.png")
    print(img.mode, mask.mode)
    inpaint.inpaint(img, mask)