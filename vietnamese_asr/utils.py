#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module tiện ích cho dự án ASR tiếng Việt."""

import os
import random
import torch
import numpy as np
import logging
import json
import re
import time
from typing import Dict, List, Optional, Union, Any
from omegaconf import DictConfig, OmegaConf

def set_seed(seed: int):
    """
    Đặt hạt giống ngẫu nhiên cho khả năng tái tạo.

    Args:
        seed: Giá trị hạt giống.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(output_dir: str, log_level: str = "INFO"):
    """
    Thiết lập logging.

    Args:
        output_dir: Thư mục đầu ra.
        log_level: Mức độ log.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()

    # Thiết lập mức độ log
    root_logger.setLevel(getattr(logging, log_level))

    # Xóa các handler hiện có
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Handler cho console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Handler cho file
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    return root_logger

def save_config_to_json(config: DictConfig, output_file: str):
    """
    Lưu cấu hình vào file JSON.

    Args:
        config: Cấu hình cần lưu.
        output_file: Đường dẫn file đầu ra.
    """
    # Chuyển đổi OmegaConf sang Python dict
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Lưu vào file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

def format_time(seconds: float) -> str:
    """
    Định dạng thời gian.

    Args:
        seconds: Số giây.

    Returns:
        str: Chuỗi thời gian định dạng.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"

class Timer:
    """Đo thời gian thực thi."""

    def __init__(self, name: str = None):
        """
        Khởi tạo timer.

        Args:
            name: Tên của timer.
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Bắt đầu timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Kết thúc timer và in thời gian."""
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time

        if self.name:
            print(f"{self.name} hoàn thành trong {format_time(elapsed_time)}")
        else:
            print(f"Hoàn thành trong {format_time(elapsed_time)}")

def count_parameters(model) -> Dict[str, int]:
    """
    Đếm số lượng tham số trong mô hình.

    Args:
        model: Mô hình cần đếm tham số.

    Returns:
        Dict[str, int]: Dictionary chứa thông tin về số lượng tham số.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params
    }

def summarize_model_info(model, processor) -> str:
    """
    Tóm tắt thông tin về mô hình.

    Args:
        model: Mô hình.
        processor: Processor.

    Returns:
        str: Chuỗi tóm tắt thông tin.
    """
    param_info = count_parameters(model)

    vocab_size = len(processor.tokenizer)

    summary = [
        "=== Thông tin mô hình ===",
        f"Tổng số tham số: {param_info['total_params']:,}",
        f"Số tham số có thể huấn luyện: {param_info['trainable_params']:,} ({param_info['trainable_params']/param_info['total_params']*100:.2f}%)",
        f"Số tham số không huấn luyện: {param_info['non_trainable_params']:,} ({param_info['non_trainable_params']/param_info['total_params']*100:.2f}%)",
        f"Kích thước từ điển: {vocab_size} token",
    ]

    return "\n".join(summary)

def transcribe_audio_file(audio_path: str, model, processor, device=None) -> str:
    """
    Phiên âm một file audio.

    Args:
        audio_path: Đường dẫn đến file audio.
        model: Mô hình ASR.
        processor: Processor.
        device: Device để chạy mô hình.

    Returns:
        str: Văn bản được phiên âm.
    """
    import librosa

    # Nạp audio
    audio, sample_rate = librosa.load(audio_path, sr=16000)

    # Tiền xử lý audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Chuyển inputs sang device tương ứng
    if device is None:
        device = next(model.parameters()).device

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Dự đoán
    with torch.no_grad():
        logits = model(**inputs).logits

    # Lấy predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Giải mã
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

def parse_metadata_file(metadata_file: str, data_dir: str = None) -> List[Dict[str, str]]:
    """
    Phân tích file metadata.

    Args:
        metadata_file: Đường dẫn đến file metadata.
        data_dir: Thư mục dữ liệu.

    Returns:
        List[Dict[str, str]]: Danh sách các mục metadata.
    """
    import pandas as pd

    df = pd.read_csv(metadata_file)

    # Kiểm tra các cột bắt buộc
    required_cols = ['audio_path', 'transcription']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Metadata file phải chứa cột '{col}'")

    if data_dir:
        # Tạo đường dẫn đầy đủ cho các file audio
        df['audio_path'] = df['audio_path'].apply(
            lambda x: os.path.join(data_dir, x) if not os.path.isabs(x) else x
        )

    # Chuyển đổi sang danh sách các dict
    return df.to_dict(orient='records')

def check_audio_files(metadata: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Kiểm tra các file audio.

    Args:
        metadata: Danh sách các mục metadata.

    Returns:
        Dict[str, int]: Thống kê về các file audio.
    """
    stats = {
        "total": len(metadata),
        "found": 0,
        "missing": 0
    }

    for item in metadata:
        if os.path.exists(item['audio_path']):
            stats['found'] += 1
        else:
            stats['missing'] += 1

    return stats