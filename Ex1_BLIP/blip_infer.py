import os
from typing import Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip_model(
    ckpt_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return processor, model


def generate_caption_blip(
    image_path: str,
    ckpt_path: Optional[str] = None,
    max_length: int = 60,  # 与训练时的max_length保持一致
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    processor, model = load_blip_model(ckpt_path, device=device)

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        # 使用与评估时相同的生成参数，确保推理质量
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,  # beam search提升质量
            no_repeat_ngram_size=2,  # 防止重复短语
            length_penalty=1.0,  # 长度控制
            early_stopping=True,  # 遇到结束符即停止
            repetition_penalty=1.1,  # 重复惩罚
            do_sample=False,  # 确定性生成
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    text = generate_caption_blip(args.image, args.ckpt)
    print(text)


