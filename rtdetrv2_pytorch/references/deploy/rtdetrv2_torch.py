"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from src.core import YAMLConfig

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def draw_and_save(im, labels, boxes, scores, save_path, thrh=0.6):
    d = ImageDraw.Draw(im)
    scr = scores[0]
    mask = scr > thrh
    lab = labels[0][mask]
    box = boxes[0][mask]
    scrs = scr[mask]
    for j, b in enumerate(box):
        d.rectangle(list(b), outline='red')
        d.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')
        print(f"  label={lab[j].item()}, score={round(scrs[j].item(), 4)}, box={[round(x.item(), 1) for x in b]}")
    im.save(save_path)
    print(f"  saved -> {save_path}")


def infer_single(model, im_pil, device, transforms):
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None].to(device)
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)
    return labels, boxes, scores


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    im_path = Path(args.im_file)

    if im_path.is_dir():
        image_files = sorted(p for p in im_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_files:
            print(f"[ERROR] 이미지 파일이 없습니다: {im_path}")
            return
        output_dir = Path('results') / im_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {len(image_files)}개 이미지 처리 시작 → 저장 위치: {output_dir}")
        for img_file in image_files:
            print(f"[{img_file.name}]")
            im_pil = Image.open(img_file).convert('RGB')
            labels, boxes, scores = infer_single(model, im_pil, args.device, transforms)
            save_path = output_dir / (img_file.stem + '.jpg')
            draw_and_save(im_pil, labels, boxes, scores, save_path)
    else:
        im_pil = Image.open(im_path).convert('RGB')
        labels, boxes, scores = infer_single(model, im_pil, args.device, transforms)
        draw_and_save(im_pil, labels, boxes, scores, 'results_0.jpg')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
