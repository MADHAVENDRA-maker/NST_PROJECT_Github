import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from utils.models import VGGEncoder, Decoder
from utils.utils import (
    get_transform,
    adaptive_instance_normalization,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Content image"
    )

    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Style image"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Decoder checkpoint",
        default="experiment/checkpoint_epoch_10.pth"
    )

    parser.add_argument(
        "--vgg",
        type=str,
        default="vgg_normalised.pth"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="stylized.png"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = get_transform(
        args.size,
        crop=False,
        final_size=args.size,
    )

    content = Image.open(args.content).convert("RGB")
    style = Image.open(args.style).convert("RGB")

    content = transform(content).unsqueeze(0).to(device)
    style = transform(style).unsqueeze(0).to(device)

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    state_dict = checkpoint["decoder"]

    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k.replace("decoder.", "net.")
        new_state_dict[new_key] = v

    decoder.load_state_dict(new_state_dict)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        content_features = encoder(content)
        style_features = encoder(style)

        t = adaptive_instance_normalization(
            content_features[-1],
            style_features[-1],
        )

        output = decoder(t)

    save_image(output, args.output)

    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()