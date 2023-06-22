import numpy as np
import glob
import os
import shutil
from PIL import Image
from PIL import ImageDraw, ImageOps
from PIL import ImageFont
import random
import gzip
from nltk.corpus import wordnet as wn
import pdb
from tqdm import tqdm
import torch
import clip
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from compute_accuracy import ImageDataset, TextDataset, clip_image_encoder, clip_text_encoder
torch.multiprocessing.set_sharing_strategy('file_system')

FONT_TYPE = '/usr/share/fonts/truetype/msttcorefonts/verdanab.ttf'

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

ALPHABET = ['a', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
                'v', 'b', 'n', 'm']

class ImageDatasetAnnotate(Dataset):
    def __init__(self, image_files, random_position=False, random_font_size=False,
                 imagenet_class_dict_root='data/imagenet/'):
        self.image_files = image_files
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.curr_font = ImageFont.truetype(FONT_TYPE, 40)
        self.random_position = random_position
        self.random_font_size = random_font_size
        n_px = 224
        self.resize_transform = Compose([Resize(n_px, interpolation=BICUBIC), CenterCrop(n_px)])
        self.normalize_transform = Compose([lambda image: image.convert("RGB"),
                                            ToTensor(),
                                            Normalize((0.48145466, 0.4578275, 0.40821073),
                                                      (0.26862954, 0.26130258, 0.27577711))])

        self.class_idx = {v[0]: v[1] for v in json.load(open(imagenet_class_dict_root+"imagenet_class_index.json", 'r')).values()}
        self.choose_a = lambda x: 'an' if x[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        self.class_idx_number = {k: i for i,k  in enumerate(self.class_idx.keys())}

    def __len__(self):
        return len(self.image_files)

    def draw_text(self, pillowImage):
        # draw on imagenet
        d = ImageDraw.Draw(pillowImage)
        if self.random_font_size:
            curr_font = ImageFont.truetype(FONT_TYPE, np.random.randint(20, 40))
        else:
            curr_font = self.curr_font

        text = ('').join([np.random.choice(ALPHABET) for i in range(np.random.randint(3, 6))])
        offsetx, offsety = d.textsize(text, font=self.curr_font)
        h, w = pillowImage.height, pillowImage.width
        if self.random_position:
            position = (np.random.randint(0, w-offsetx), np.random.randint(0, h-offsety))
        else:
            position = ((w//2-offsetx//2, h//2-offsety//2))

        d.text(xy=position, text=text, font=curr_font, fill='black')
        # draw on empty image
        char_image = np.ones((224, 224, 3), np.uint8) * 255
        emptyPillowImage = Image.fromarray(char_image)
        d2 = ImageDraw.Draw(emptyPillowImage)
        d2.text(xy=position, text=text, font=curr_font, fill='black')
        return pillowImage, emptyPillowImage, text

    def __getitem__(self, idx):
        imagenet_identifier = self.image_files[idx].split('/')[-2].replace('.JPEG', '')
        imagenet_identifier_file = self.image_files[idx].split('/')[-1].replace('.JPEG', '')
        pillowImage = Image.open(self.image_files[idx])
        pillowImage = self.resize_transform(pillowImage)
        image, text_image, text = self.draw_text(pillowImage)
        image = self.normalize_transform(image)
        text_image = self.normalize_transform(text_image)
        cls = self.class_idx[imagenet_identifier].replace('_', ' ')
        class_text = 'an image of ' + self.choose_a(cls) +' '+ cls
        return image, text_image, text, imagenet_identifier, class_text, self.class_idx_number[imagenet_identifier]

def encode_images(paths, save_file):
    image_features = clip_image_encoder(paths, dataset=ImageDataset)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.float()
    np.save(f'{save_file}', image_features.data.cpu().numpy())

def encode_text(text, save_file):
    text_features = clip_text_encoder(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.float()
    np.save(f'{save_file}', text_features.data.cpu().numpy())


def compute_imagenet_features(train_img_folder, val_img_folder, output_train_features, output_val_features):
    imagenet_frames_train = glob.glob(f'{train_img_folder}/*/*')
    imagenet_frames_val = glob.glob(f'{val_img_folder}/*/*')
    encode_images(imagenet_frames_train, output_train_features)
    encode_images(imagenet_frames_val, output_val_features)


def write_on_images(image_paths, save_files,
                    random_position, random_font_size):
    files = glob.glob(image_paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    dataset = ImageDatasetAnnotate(files, random_position=random_position, random_font_size=random_font_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=30, collate_fn=None)
    image_with_text_features = []
    image_text_features = []

    start = time.time()
    all_text_written = []
    all_text_imagenet = []
    numerical_imagenet_classes = []


    for i, batch in enumerate(tqdm(dataloader)):
        image, text_image, text, imagenet_identifier, class_text, class_ids = batch
        with torch.no_grad():
            image_with_text_features.append(model.encode_image(image.to(device)))
            image_text_features.append(model.encode_image(text_image.to(device)))

        all_text_written.extend([i + '\n' for i in text])
        all_text_imagenet.extend([i + '\n' for i in class_text])
        numerical_imagenet_classes.extend(class_ids)

    end = time.time()
    print('Extracting features took: {} seconds'.format(end - start))
    # save image + text features
    image_features = torch.cat(image_with_text_features, axis=0).cpu()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.float()
    np.save(f'{save_files["image_with_text"]}', image_features.data.cpu().numpy())
    # save text image features
    image_features = torch.cat(image_text_features, axis=0).cpu()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.float()
    np.save(f'{save_files["text_image"]}', image_features.data.cpu().numpy())

    with open(save_files["text"], 'w') as f:
        f.writelines(all_text_written)
    with open(save_files["text_class"], 'w') as f:
        f.writelines(all_text_imagenet)

    np.save(save_files['class_id'], np.asarray(numerical_imagenet_classes))

    encode_text(all_text_written, save_files["text_features"])
    rev_dict = {v: k for k,v in dataset.class_idx_number.items()}
    class_labels_text_imagenet = ['an image of ' + dataset.choose_a(dataset.class_idx[rev_dict[i]]) + ' ' + dataset.class_idx[rev_dict[i]] for i in range(1000)]
    encode_text(class_labels_text_imagenet, save_files["text_class_features"])




if __name__ == "__main__":
    # compute_imagenet_features()

    image_paths = '/data/vision/torralba/datasets/imagenet_pytorch/train/*/*'
    save_files = {"image_with_text": 'data/imagenet/train_with_text_img.npy',
                  "text_image": 'data/imagenet/train_text_img.npy',
                  "text": 'data/imagenet/train_text.txt',
                  "text_class": 'data/imagenet/train_text_class.txt',
                  'text_features': 'data/imagenet/train_text.npy',
                  "text_class_features": 'data/imagenet/train_text_class.npy',
                  'class_id': 'data/imagenet/train_class_id.npy'}
    write_on_images(image_paths, save_files,
                    random_position=False, random_font_size=False)


