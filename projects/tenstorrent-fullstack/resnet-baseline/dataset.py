import random
import tarfile
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

from dtype_config import MODEL_TORCH_DTYPE
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor


def download_oxford_pet(root):

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data"

    for name in ["images", "annotations"]:
        if (root / name).exists():
            continue
        archive = root / f"{name}.tar.gz"
        if not archive.exists():
            print(f"다운로드: {name}")
            urllib.request.urlretrieve(f"{base_url}/{name}.tar.gz", archive)
        with tarfile.open(archive) as file:
            file.extractall(root, filter="data")


class OxfordPetDataset(Dataset):
    def __init__(self, root, split="trainval", transform=None):
        self.root = Path(root).expanduser()
        self.transform = transform
        self.samples = []
        self.class_names = {}


        split_file = self.root / "annotations" / f"{split}.txt"
        for line in split_file.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.split()
            image_name = columns[0]
            label = int(columns[1]) - 1
            self.samples.append((image_name, label))
            self.class_names[label] = image_name.rsplit("_", 1)[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, label = self.samples[index]
        image_path = self.root / "images" / f"{image_name}.jpg"
        with Image.open(image_path) as file:
            image = file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = pil_to_tensor(image).float() / 255
        image = image.permute(1, 2, 0).contiguous()
        image = image.to(MODEL_TORCH_DTYPE)

        return image, label


def classification_loaders(root, image_size=112, batch_size=8, workers=0,
                           seed=42, val_fraction=0.2, download=False):

    if image_size < 32 or batch_size < 1 or workers < 0:
        raise ValueError("image_size >= 32, batch_size >= 1, workers >= 0 이어야 합니다.")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction은 0과 1 사이여야 합니다.")
    if download:
        download_oxford_pet(root)


    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = OxfordPetDataset(root, transform=train_transform)
    val_data = OxfordPetDataset(root, transform=val_transform)


    class_indices = {}
    for index, (_, label) in enumerate(train_data.samples):
        class_indices.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for indices in class_indices.values():
        if len(indices) < 2:
            raise ValueError("각 품종에 이미지가 최소 2장 필요합니다.")
        rng.shuffle(indices)
        val_count = round(len(indices) * val_fraction)
        val_count = max(1, min(len(indices) - 1, val_count))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    if not train_indices:
        raise ValueError("학습 이미지가 없습니다.")

    options = {"batch_size": batch_size, "num_workers": workers}
    if workers > 0:
        options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(
        Subset(train_data, train_indices), shuffle=True,
        generator=torch.Generator().manual_seed(seed), **options,
    )
    val_loader = DataLoader(Subset(val_data, val_indices), shuffle=False, **options)
    return train_loader, val_loader, train_data.class_names


if __name__ == "__main__":
    train_loader, val_loader, names = classification_loaders("~/datasets/oxford_pet")
    images, labels = next(iter(train_loader))
    print("이미지 크기:", images.shape)
    print("정답:", labels)
    print("학습 이미지:", len(train_loader.dataset))
    print("검증 이미지:", len(val_loader.dataset))
