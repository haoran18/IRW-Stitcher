import os
import argparse
import torch
import numpy as np
import cv2
import random

from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline

from procossing import preprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
    )
    parser.add_argument(
        "--loramodel_path",
        type=str,
        default="loraweight",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="<A>",
    )
    parser.add_argument(
        "--data_root",
        type=str,
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--num_seed",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    return args


def main(args):
    device = args.device

    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    loramodel_path = args.loramodel_path
    
    data_root = args.data_root
    save_root = args.save_root

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=torch.float16
    )

    pipe.load_lora_weights(loramodel_path)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    imagenames = sorted(os.listdir(os.path.join(data_root,"imgs")))
    
    for i in range(args.num_seed):
        seed = random.randint(0, 100000)
        for name in imagenames:
            generator = torch.manual_seed(seed)

            stitched_image = cv2.imread(os.path.join(data_root,"imgs", name))
            stitched_mask = cv2.imread(os.path.join(data_root,"masks", name), cv2.IMREAD_GRAYSCALE)
            h,w = stitched_image.shape[0], stitched_image.shape[1]

            image, mask_image = preprocessing(stitched_image,stitched_mask)

            text_prompt=args.test_prompt

            new_image = pipe(
                        prompt=text_prompt,
                        num_inference_steps=50,
                        generator=generator,
                        image=image,
                        mask_image=mask_image,
                        height = 512,
                        width =1024,
                    ).images[0]
            new_image = new_image.crop((512, 0, 1024, 512))
            new_image.resize((w,h))

            new_image.save(os.path.join(save_root, "seed"+str(seed) +"+"+name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
