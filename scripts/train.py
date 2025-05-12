#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script huấn luyện mô hình nhận dạng tiếng nói tiếng Việt.
Sử dụng:
    python -m scripts.train --config configs/default_config.yaml
"""

import os
import sys
import torch
from omegaconf import DictConfig, OmegaConf

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vietnamese_asr.config import parse_args_with_config, update_config_for_distributed, print_config_summary, save_config_to_json
from vietnamese_asr.data import VietnameseASRDataModule
from vietnamese_asr.model import VietnameseASRModelHandler
from vietnamese_asr.trainer import VietnameseASRTrainer
from vietnamese_asr.utils import set_seed, setup_logging, Timer, summarize_model_info


def main():
    """Hàm main cho huấn luyện mô hình."""
    # Parse arguments và load config
    args, config = parse_args_with_config()

    # Cập nhật config cho môi trường phân tán
    config = update_config_for_distributed(config)

    # Thiết lập output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Lưu config
    save_config_to_json(config, os.path.join(config.training.output_dir, "config.json"))

    # Thiết lập logging
    logger = setup_logging(config.training.output_dir)

    # In thông tin cấu hình
    print_config_summary(config)

    # Đặt seed
    set_seed(config.training.seed)

    # Khởi tạo model handler
    with Timer("Khởi tạo mô hình"):
        model_handler = VietnameseASRModelHandler(config)
        device = model_handler.setup_device()
        processor = model_handler.load_processor()
        model = model_handler.load_model()

        # In thông tin mô hình
        print(summarize_model_info(model, processor))

    # Khởi tạo data module
    with Timer("Nạp và xử lý dữ liệu"):
        data_module = VietnameseASRDataModule(config, processor)
        train_dataset, val_dataset = data_module.load_data()

        # Hiển thị một số mẫu
        print("\nMẫu dữ liệu:")
        print(data_module.show_samples(3))

        # Xử lý dữ liệu
        train_dataset, val_dataset = data_module.process_data()

        # Tạo data collator
        data_collator = data_module.create_data_collator(processor)

    # Khởi tạo trainer
    trainer = VietnameseASRTrainer(
        config=config,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Khởi tạo HF Trainer
    hf_trainer = trainer.create_trainer()

    # Huấn luyện mô hình
    with Timer("Huấn luyện mô hình"):
        metrics, best_checkpoint = trainer.train()

    print(f"\nHuấn luyện hoàn tất!")
    print(f"Mô hình tốt nhất được lưu tại: {best_checkpoint}")
    print(f"WER cuối cùng: {metrics.get('eval_wer', 'N/A')}")

    # Gợi ý tiếp theo
    print("\nĐể đánh giá mô hình:")
    print(f"python -m scripts.evaluate --model_path {best_checkpoint} --data_dir {config.data.data_dir} --metadata_file {config.data.metadata_file}")

    print("\nĐể chuyển đổi giọng nói thành văn bản:")
    print(f"python -m scripts.transcribe --model_path {best_checkpoint} --audio_path /đường/dẫn/đến/file_audio.wav")


if __name__ == "__main__":
    # Xử lý lỗi Distributed
    try:
        main()
    except KeyboardInterrupt:
        print("\nHuấn luyện bị dừng bởi người dùng")
        sys.exit(0)
    except Exception as e:
        # Xử lý lỗi đặc biệt trong môi trường phân tán
        if "find_unused_parameters" in str(e):
            print("\nLỗi liên quan đến DDP parameters. Thử thêm --config_override configs/multi_gpu.yaml và đặt ddp_find_unused_parameters=True")
        else:
            import traceback
            print(f"\nLỗi trong quá trình huấn luyện: {e}")
            traceback.print_exc()
        sys.exit(1)