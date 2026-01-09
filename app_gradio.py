import os
from typing import Dict, Tuple

import gradio as gr

from Model1_YellowOrange.train_eval import generate_caption_model1
from Model2_Transformer.train_eval import generate_caption_model2
from Original_Model.train_eval import generate_caption_original
from Model3_CNN_GRU.train_eval import generate_caption_cnn_gru
from Ex1_BLIP.blip_infer import generate_caption_blip


MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def generate_caption(
    image,
    model_name: str,
    checkpoint: str,
):
    if image is None:
        return "请先上传图片。"

    # 保存临时图片
    temp_path = "temp_input.jpg"
    image.save(temp_path)

    ckpt = checkpoint.strip() or None

    try:
        if model_name == "Model1":
            ckpt = ckpt or os.path.join(
                "data", "output", "weights", "model1_yelloworange.pth"
            )
            text = generate_caption_model1(temp_path, ckpt)
        elif model_name == "Model2":
            ckpt = ckpt or os.path.join("data", "output", "weights", "model2_transformer.pth")
            text = generate_caption_model2(temp_path, ckpt)
        elif model_name == "Original":
            ckpt = ckpt or os.path.join("data", "output", "weights", "original_model.pth")
            text = generate_caption_original(temp_path, ckpt)
        elif model_name == "CNN-GRU":
            ckpt = ckpt or os.path.join("data", "output", "weights", "cnn_gru.pth")
            text = generate_caption_cnn_gru(temp_path, ckpt)
        elif model_name == "BLIP":
            ckpt = ckpt or os.path.join("data", "output", "weights", "blip_finetuned.pth")
            text = generate_caption_blip(temp_path, ckpt)
        else:
            return "未知模型类型。"
    except Exception as e:
        return f"推理出错：{e}"

    return text


def launch():
    with gr.Blocks() as demo:
        # 标题和说明
        gr.Markdown(
            """
            # 🖼️ 图像描述生成系统
            
            上传图片，选择模型，即可自动生成图像描述。支持多种深度学习模型进行对比。
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # 图片上传区域，限制高度并保持比例
                img = gr.Image(
                    type="pil", 
                    label="📤 上传图片",
                    height=350,  # 限制图片显示高度（像素），保持宽高比
                    show_label=True,
                    container=True,
                )
                
                # 模型选择移到主区域，更容易访问
                model = gr.Dropdown(
                    ["Model1", "Model2", "Original", "CNN-GRU", "BLIP"],
                    value="Model1",
                    label="🤖 选择模型",
                    info="选择用于生成描述的模型",
                )
                
                with gr.Accordion("⚙️ 高级设置", open=False):
                    ckpt = gr.Textbox(
                        label="📁 Checkpoint 路径（可选）",
                        placeholder="留空使用默认模型权重",
                        value="",
                        info="如需使用自定义训练的模型权重，请输入完整路径",
                    )
                
                # 生成按钮，使用主要样式
                btn = gr.Button(
                    "🚀 生成描述", 
                    variant="primary",
                    size="lg",
                )
            
            with gr.Column(scale=1):
                # 输出区域
                output = gr.Textbox(
                    label="📝 生成的描述",
                    lines=10,
                    placeholder="生成的图像描述将显示在这里...",
                    max_lines=15,
                )
                
                # 模型信息展示区域
                with gr.Accordion("ℹ️ 模型说明", open=False):
                    gr.Markdown(
                        """
                        **Model1 (YellowOrange)**: CNN + 注意力机制 + GRU  
                        **Model2 (Transformer)**: Transformer 编码器-解码器架构  
                        **Original**: 原始 CNN + 简单注意力 + GRU  
                        **CNN-GRU**: ResNet + GRU 解码器  
                        **BLIP**: 预训练的视觉-语言模型
                        """
                    )

        # 绑定事件
        btn.click(
            fn=generate_caption,
            inputs=[img, model, ckpt],
            outputs=[output],
        )
        
        # 添加示例说明
        gr.Markdown(
            """
            ---
            ### 💡 使用提示
            - 支持常见图片格式（JPG、PNG、JPEG等）
            - 图片会自动调整到合适大小显示
            - 不同模型可能生成不同风格的描述，可以尝试多个模型对比
            """
        )

    demo.launch(share=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch()


