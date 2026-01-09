import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm


SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


def load_captions(caption_path: str) -> Dict[str, str]:
    with open(caption_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_vocab(captions: Dict[str, str], min_freq: int) -> Dict[str, int]:
    counter = Counter()
    for sent in captions.values():
        tokens = sent.strip().lower().split()
        counter.update(tokens)

    vocab = {}
    idx = 0
    for tok in SPECIAL_TOKENS:
        vocab[tok] = idx
        idx += 1

    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab


def encode_caption(
    sentence: str, vocab: Dict[str, int], max_len: int
) -> Tuple[List[int], int]:
    tokens = sentence.strip().lower().split()
    encoded = [vocab["<start>"]]
    for t in tokens:
        encoded.append(vocab.get(t, vocab["<unk>"]))
    encoded.append(vocab["<end>"])

    if len(encoded) > max_len:
        encoded = encoded[:max_len]
        encoded[-1] = vocab["<end>"]
    caplen = len(encoded)

    if len(encoded) < max_len:
        encoded.extend([vocab["<pad>"]] * (max_len - len(encoded)))

    return encoded, caplen


def split_train_test(
    items: List[Tuple[str, str]], train_ratio: float
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    n_train = int(len(items) * train_ratio)
    train = items[:n_train]
    test = items[n_train:]
    return train, test


def preprocess(
    data_root: str = "data",
    caption_file: str = "caption.json",
    images_dir: str = "images",
    output_dir: str = os.path.join("data", "output"),
    train_ratio: float = 0.8,
    min_freq: int = 5,
    max_len: int = 64,
) -> None:
    caption_path = os.path.join(data_root, caption_file)
    images_path = os.path.join(data_root, images_dir)
    os.makedirs(output_dir, exist_ok=True)

    captions = load_captions(caption_path)

    # 过滤不存在的图片
    items: List[Tuple[str, str]] = []
    for fname, sent in captions.items():
        img_path = os.path.join(images_path, fname)
        if os.path.isfile(img_path):
            items.append((fname, sent))

    # 构建词表
    all_captions = {k: v for k, v in items}
    vocab = build_vocab(all_captions, min_freq=min_freq)

    # 划分训练/测试
    train_items, test_items = split_train_test(items, train_ratio=train_ratio)

    def process_split(
        split_items: List[Tuple[str, str]]
    ) -> Tuple[List[List[int]], List[int], List[str]]:
        encoded_list: List[List[int]] = []
        caplens: List[int] = []
        img_paths: List[str] = []

        for fname, sent in tqdm(split_items, desc="Encoding captions"):
            enc, caplen = encode_caption(sent, vocab, max_len=max_len)
            encoded_list.append(enc)
            caplens.append(caplen)
            img_paths.append(os.path.join(data_root, images_dir, fname))
        return encoded_list, caplens, img_paths

    enc_train, caplens_train, paths_train = process_split(train_items)
    enc_test, caplens_test, paths_test = process_split(test_items)

    # 写入文件
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(output_dir, "encoded_captions_train.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(enc_train, f)
    with open(
        os.path.join(output_dir, "encoded_captions_test.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(enc_test, f)

    with open(
        os.path.join(output_dir, "image_paths_train.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(paths_train, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(output_dir, "image_paths_test.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(paths_test, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(output_dir, "caplens_train.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(caplens_train, f)
    with open(
        os.path.join(output_dir, "caplens_test.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(caplens_test, f)


if __name__ == "__main__":
    preprocess()


