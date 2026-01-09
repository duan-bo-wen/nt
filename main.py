import argparse
import os

from preprocess import preprocess
from Model1_YellowOrange.train_eval import train_model1, eval_model1
from Model2_Transformer.train_eval import train_model2, eval_model2
from Original_Model.train_eval import train_original, eval_original
from Model3_CNN_GRU.train_eval import train_cnn_gru, eval_cnn_gru
from Ex1_BLIP.blip_infer import generate_caption_blip


def main():
    parser = argparse.ArgumentParser(description="Unified CLI for captioning project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess
    subparsers.add_parser("preprocess", help="run data preprocessing")

    # train
    train_p = subparsers.add_parser("train", help="train a model")
    train_p.add_argument(
        "--model",
        type=str,
        choices=["model1", "model2", "original", "cnn_gru", "blip"],
        required=True,
    )

    # eval
    eval_p = subparsers.add_parser("eval", help="evaluate a model")
    eval_p.add_argument(
        "--model",
        type=str,
        choices=["model1", "model2", "original", "cnn_gru", "blip"],
        required=True,
    )
    eval_p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="path to checkpoint (for BLIP optional)",
    )

    # ui
    subparsers.add_parser("ui", help="launch Gradio UI")

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess()
    elif args.command == "train":
        if args.model == "model1":
            train_model1()
        elif args.model == "model2":
            train_model2()
        elif args.model == "original":
            train_original()
        elif args.model == "cnn_gru":
            train_cnn_gru()
        elif args.model == "blip":
            from Ex1_BLIP.train_blip import train_blip

            train_blip()
    elif args.command == "eval":
        if args.model == "model1":
            ckpt = (
                args.checkpoint
                or os.path.join("data", "output", "weights", "model1_yelloworange.pth")
            )
            eval_model1(ckpt)
        elif args.model == "model2":
            ckpt = args.checkpoint or os.path.join(
                "data", "output", "weights", "model2_transformer.pth"
            )
            eval_model2(ckpt)
        elif args.model == "original":
            ckpt = args.checkpoint or os.path.join(
                "data", "output", "weights", "original_model.pth"
            )
            eval_original(ckpt)
        elif args.model == "cnn_gru":
            ckpt = args.checkpoint or os.path.join(
                "data", "output", "weights", "cnn_gru.pth"
            )
            eval_cnn_gru(ckpt)
        elif args.model == "blip":
            from Ex1_BLIP.train_blip import eval_blip

            ckpt = args.checkpoint or os.path.join("data", "output", "weights", "blip_finetuned.pth")
            eval_blip(ckpt_path=ckpt)
    elif args.command == "ui":
        import app_gradio

        app_gradio.launch()


if __name__ == "__main__":
    main()


