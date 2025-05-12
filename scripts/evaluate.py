#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script đánh giá mô hình nhận dạng tiếng nói tiếng Việt.
Sử dụng:
    python -m scripts.evaluate --model_path /đường/dẫn/đến/mô_hình --data_dir /đường/dẫn/đến/dữ_liệu --metadata_file /đường/dẫn/đến/metadata.csv
"""

import os
import sys
import torch
import argparse
from omegaconf import OmegaConf, DictConfig

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from vietnamese_asr.config import load_config
from vietnamese_asr.data import VietnameseASRDataModule
from vietnamese_asr.metrics import VietnameseASREvaluator
from vietnamese_asr.utils import set_seed, Timer


def parse_args():
    """Parse arguments từ command line."""
    parser = argparse.ArgumentParser(description='Đánh giá mô hình ASR tiếng Việt')

    # Tham số bắt buộc
    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn đến mô hình cần đánh giá')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Thư mục chứa dữ liệu âm thanh')
    parser.add_argument('--metadata_file', type=str, required=True,
                        help='File CSV chứa metadata')

    # Tham số tùy chọn
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Thư mục lưu kết quả đánh giá')
    parser.add_argument('--config', type=str, default=None,
                        help='File config.yaml từ thư mục mô hình (mặc định sẽ tìm trong model_path)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Kích thước batch cho đánh giá')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Số lượng mẫu tối đa để đánh giá (None = tất cả)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Hạt giống ngẫu nhiên')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Số worker cho DataLoader')
    parser.add_argument('--device', type=str, default=None,
                        help='Device để đánh giá (cuda, cpu, hoặc None để tự phát hiện)')
    parser.add_argument('--analyze_errors', action='store_true',
                        help='Phân tích các lỗi phổ biến')

    return parser.parse_args()


def create_config_from_args(args):
    """Tạo config từ arguments."""
    # Tạo config cơ bản
    config = OmegaConf.create({
        "data": {
            "data_dir": args.data_dir,
            "metadata_file": args.metadata_file,
            "audio_column": "audio_path",
            "text_column": "transcription",
            "val_split": 0.0,  # Không cần chia tập validation khi đánh giá
            "max_train_samples": None,
            "max_val_samples": args.max_samples,
            "sample_rate": 16000,
        },
        "dataloader": {
            "num_workers": args.num_workers,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True if args.num_workers > 0 else False,
        },
        "training": {
            "seed": args.seed,
            "per_device_eval_batch_size": args.batch_size,
        },
    })

    # Nếu có file config từ tham số hoặc trong thư mục mô hình
    config_path = args.config
    if config_path is None:
        # Tìm trong thư mục mô hình
        model_config_path = os.path.join(args.model_path, "config.json")
        if os.path.exists(model_config_path):
            print(f"Sử dụng config từ mô hình: {model_config_path}")
            try:
                model_config = OmegaConf.load(model_config_path)
                # Chỉ merge một số phần của config
                if "data" in model_config:
                    # Chỉ giữ lại sample_rate từ model config
                    if "sample_rate" in model_config.data:
                        config.data.sample_rate = model_config.data.sample_rate
                if "model" in model_config:
                    config.model = model_config.model
            except Exception as e:
                print(f"Không thể nạp config từ mô hình: {e}")
    else:
        # Sử dụng config từ tham số
        try:
            config_from_file = OmegaConf.load(config_path)
            config = OmegaConf.merge(config, config_from_file)
        except Exception as e:
            print(f"Không thể nạp config từ file: {e}")

    return config


def main():
    """Hàm main cho đánh giá mô hình."""
    # Parse arguments
    args = parse_args()

    # Tạo config từ arguments
    config = create_config_from_args(args)

    # Thiết lập output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "evaluation_results")
    os.makedirs(args.output_dir, exist_ok=True)

    # Đặt seed
    set_seed(config.training.seed)

    # Thiết lập device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    # Nạp mô hình và processor
    with Timer("Nạp mô hình"):
        try:
            processor = Wav2Vec2Processor.from_pretrained(args.model_path)
            model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
            model = model.to(device)
            model.eval()
            print(f"Đã nạp mô hình từ {args.model_path}")
        except Exception as e:
            print(f"Lỗi khi nạp mô hình: {e}")
            sys.exit(1)

    # Nạp dữ liệu
    with Timer("Nạp dữ liệu"):
        # Nạp và xử lý dữ liệu
        data_module = VietnameseASRDataModule(config, processor)

        # Nạp dữ liệu
        train_dataset, val_dataset = data_module.load_data()

        # Chỉ sử dụng tập train cho đánh giá nếu không có val split
        if config.data.val_split == 0:
            eval_dataset = train_dataset
        else:
            eval_dataset = val_dataset

        # Giới hạn số lượng mẫu nếu được chỉ định
        if args.max_samples and args.max_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(args.max_samples))

        # Xử lý dữ liệu
        eval_dataset = eval_dataset.map(data_module.remove_special_characters)
        eval_dataset = eval_dataset.map(
            data_module.prepare_dataset,
            remove_columns=eval_dataset.column_names,
            num_proc=config.dataloader.num_workers
        )

        print(f"Số lượng mẫu đánh giá: {len(eval_dataset)}")

    # Đánh giá mô hình
    with Timer("Đánh giá mô hình"):
        evaluator = VietnameseASREvaluator(
            model=model,
            processor=processor,
            output_dir=args.output_dir
        )

        # Đánh giá
        metrics = evaluator.evaluate_dataset(
            dataset=eval_dataset,
            batch_size=config.training.per_device_eval_batch_size
        )

        # In kết quả
        print("\n=== Kết quả đánh giá ===")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")

        # Phân tích lỗi nếu được yêu cầu
        if args.analyze_errors:
            print("\nPhân tích các lỗi phổ biến...")
            evaluator.analyze_errors(num_samples=20)

    print(f"\nĐánh giá hoàn tất! Kết quả được lưu tại {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐánh giá bị dừng bởi người dùng")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\nLỗi trong quá trình đánh giá: {e}")
        traceback.print_exc()
        sys.exit(1)