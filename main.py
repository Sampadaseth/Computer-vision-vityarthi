import argparse
import os
import sys
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Image Segmentation CLI')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    return parser.parse_args()

def get_color_map():
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ], dtype=np.uint8)
    return palette

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_image = Image.open(args.input).convert("RGB")
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()

    colors = get_color_map()
    mask = colors[output_predictions]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    out_name = os.path.splitext(os.path.basename(args.input))[0] + "_segmented.png"
    out_path = os.path.join(args.output, out_name)

    success = cv2.imwrite(out_path, mask_bgr)
    
    if success and os.path.exists(out_path):
        print(f"SUCCESS: Segmented image saved to -> {out_path}")
    else:
        print("ERROR: Image save failed.")

if __name__ == "__main__":
    main()