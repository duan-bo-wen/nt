import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class CaptionDataset(Dataset):
    """
    使用预处理生成的 encoded_captions_*.json / image_paths_*.json / caplens_*.json
    """

    def __init__(
        self,
        split: str,
        data_root: str = "data",
        output_dir: str = os.path.join("data", "output"),
    ) -> None:
        assert split in {"train", "test"}
        self.split = split

        with open(
            os.path.join(output_dir, f"image_paths_{split}.json"), "r", encoding="utf-8"
        ) as f:
            self.image_paths: List[str] = json.load(f)

        with open(
            os.path.join(output_dir, f"encoded_captions_{split}.json"),
            "r",
            encoding="utf-8",
        ) as f:
            self.captions: List[List[int]] = json.load(f)

        with open(
            os.path.join(output_dir, f"caplens_{split}.json"), "r", encoding="utf-8"
        ) as f:
            self.caplens: List[int] = json.load(f)

        with open(os.path.join(output_dir, "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = torch.tensor(self.captions[idx], dtype=torch.long)
        caplen = self.caplens[idx]
        return image, caption, caplen


def get_loader(
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CaptionDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_vocab(output_dir: str = os.path.join("data", "output")):
    with open(os.path.join(output_dir, "vocab.json"), "r", encoding="utf-8") as f:
        return json.load(f)


