import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import time
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

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


class ImageDataset(Dataset):
    def __init__(self, image_files, preprocess=None):
        self.image_files = image_files
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        if preprocess is None:
            pass
        else:
            self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.preprocess(Image.open(self.image_files[idx]))

class TextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.tokenized_sentences = clip.tokenize(sentences)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.tokenized_sentences[idx]


def clip_image_encoder(files, dataset=ImageDataset, preprocess=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    dataset = dataset(files, preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=30, collate_fn=None)
    image_features = []
    start = time.time()

    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            image_features.append(model.encode_image(batch.to(device)))
    end = time.time()
    print('Extracting features took: {} seconds'.format(end - start))
    return torch.cat(image_features, axis=0).cpu()


def clip_text_encoder(sentences):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    dataset = TextDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=20, collate_fn=None)
    text_features = []
    start = time.time()
    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            text_features.append(model.encode_text(batch.to(device)))
    end = time.time()
    print('Extracting features took: {} seconds'.format(end - start))
    return torch.cat(text_features, axis=0).cpu()

def encode_images(paths, save_file, dataset=ImageDataset):
    image_features = clip_image_encoder(paths, dataset=ImageDataset)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.float()
    np.save(save_file, image_features.data.cpu().numpy())

def encode_text(text, save_file):
    text_features = clip_text_encoder(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.float()
    np.save(save_file, text_features.data.cpu().numpy())