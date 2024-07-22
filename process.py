


from network import U2NET
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
import random
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class Normalize_image(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean
        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)
        else:
            assert "Please set proper channels! Normalization implemented only for 1, 3 and 18"

def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)

def average_color(colors):
    if len(colors) == 0:
        return [0, 0, 0]  # default to black if no colors found
    avg_color = np.mean(colors, axis=0).astype(int)
    return avg_color

def find_suitable_palette(target_colors, color_palettes):
    def color_similarity(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    suitable_palette = None
    min_difference = float('inf')

    for palette in color_palettes:
        hex_colors = [tuple(int(palette[i][j:j+2], 16) for j in (1, 3, 5)) for i in range(len(palette))]
        differences = [min(color_similarity(tc, pc) for pc in hex_colors) for tc in target_colors]

        total_difference = sum(differences)
        if total_difference < min_difference:
            min_difference = total_difference
            suitable_palette = hex_colors

    return suitable_palette

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

def generate_mask(input_image, net, palette, device='cpu'):
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    alpha_out_dir = os.path.join("output", 'alpha')
    cloth_seg_out_dir = os.path.join("output", 'cloth_seg')

    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    for cls in range(1, 4):  # Exclude background class (0)
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    color_regions = {}
    average_colors = {}

    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_mask_img.save(os.path.join(alpha_out_dir, f'{cls}.png'))

        # Extract colors from the original image
        mask = np.array(alpha_mask_img) / 255
        mask = mask.astype(bool)
        original_image_array = np.array(input_image)
        colors = original_image_array[mask]
        color_regions[cls] = colors

        # Calculate the average color
        avg_color = average_color(colors)
        average_colors[cls] = avg_color

    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    cloth_seg.save(os.path.join(cloth_seg_out_dir, 'final_seg.png'))

    return cloth_seg, output_arr, color_regions, average_colors

def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net

def generate_mask_unet(input_image_path, device='cpu'):
    # Load image
    image = load_rgb(input_image_path)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    model = create_model("Unet_2020-10-30")
    model.eval()
    model.to(device)

    with torch.no_grad():
        prediction = model(x.to(device))[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)

    return image, mask, dst

def apply_new_colors(image, output_arr, color_palettes):
    new_image = image.copy()
    for cls in range(1, 4):  # Iterate over each category (excluding background)
        mask = output_arr[0] == cls
        if np.any(mask):
            selected_palette = random.choice(color_palettes)
            selected_colors = [tuple(int(selected_palette[i][j:j+2], 16) for j in (1, 3, 5)) for i in range(len(selected_palette))]
            for i, color in enumerate(selected_colors):
                mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                new_image[mask_resized.astype(bool)] = color
    return new_image

def draw_color_regions(image, output_arr):
    def apply_random_colors(image, output_arr):
        image_with_boxes = image.copy()
        overlay = image_with_boxes.copy()
        alpha = 0.4  # Transparency factor

        for cls in range(1, 4):  # Iterate over each category (excluding background)
            mask = output_arr[0] == cls
            if np.any(mask):
                color = [random.randint(0, 255) for _ in range(3)]
                mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                color_mask = mask_resized.astype(bool)
                overlay[color_mask] = color

        cv2.addWeighted(overlay, alpha, image_with_boxes, 1 - alpha, 0, image_with_boxes)
        return image_with_boxes

    image1 = apply_random_colors(image, output_arr)
    image2 = apply_random_colors(image, output_arr)
    image3 = apply_random_colors(image, output_arr)

    return image1, image2, image3
