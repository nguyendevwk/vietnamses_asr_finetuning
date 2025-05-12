#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module quản lý cấu hình cho dự án ASR tiếng Việt."""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig


def load_config(config_file: str) -> DictConfig:
    """
    Load cấu hình từ file YAML.

    Args:
        config_file: Đường dẫn đến file cấu hình.

    Returns:
        DictConfig: Cấu hình đã nạp.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File cấu hình không tồn tại: {config_file}")

    return OmegaConf.load(config_file)


def save_config(config: DictConfig, output_file: str) -> None:
    """
    Lưu cấu hình vào file YAML.

    Args:
        config: Cấu hình cần lưu.
        output_file: Đường dẫn file đầu ra.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge hai cấu hình, với override_config ghi đè lên base_config.

    Args:
        base_config: Cấu hình cơ sở.
        override_config: Cấu hình ghi đè.

    Returns:
        DictConfig: Cấu hình đã merge.
    """
    return OmegaConf.merge(base_config, override_config)


def parse_args_with_config() -> tuple:
    """
    Parse arguments từ command line và kết hợp với file cấu hình.

    Returns:
        tuple: (args, config) - args là các tham số từ command line,
               config là cấu hình đầy đủ đã merge.
    """
    parser = argparse.ArgumentParser(description='Fine-tune Vietnamese ASR model')

    # Tham số config
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--config_override', type=str, default=None,
                        help='Path to config override file')

    # Các tham số cơ bản để ghi đè cấu hình từ command line
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing audio files')
    parser.add_argument('--metadata_file', type=str,
                        help='Path to metadata CSV file')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save model checkpoints')
    parser.add_argument('--base_model', type=str,
                        help='Base model to fine-tune')

    # Các tham số training
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size per device')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int,
                        help='Number of dataloader workers')

    # Các tham số phân tán
    parser.add_argument('--n_gpu', type=int,
                        help='Number of GPUs to use')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')

    # Parse arguments
    args = parser.parse_args()

    # Load config từ file
    config = load_config(args.config)

    # Nếu có file config ghi đè
    if args.config_override:
        override_config = load_config(args.config_override)
        config = merge_configs(config, override_config)

    # Ghi đè config từ command line arguments
    cmd_config = OmegaConf.create({})

    # Data config
    if args.data_dir:
        cmd_config.data = cmd_config.get('data', {})
        cmd_config.data.data_dir = args.data_dir
    if args.metadata_file:
        cmd_config.data = cmd_config.get('data', {})
        cmd_config.data.metadata_file = args.metadata_file
    if args.base_model:
        cmd_config.model = cmd_config.get('model', {})
        cmd_config.model.base_model = args.base_model

    # Training config
    if args.batch_size:
        cmd_config.training = cmd_config.get('training', {})
        cmd_config.training.per_device_train_batch_size = args.batch_size
    if args.epochs:
        cmd_config.training = cmd_config.get('training', {})
        cmd_config.training.num_train_epochs = args.epochs
    if args.learning_rate:
        cmd_config.training = cmd_config.get('training', {})
        cmd_config.training.learning_rate = args.learning_rate
    if args.output_dir:
        cmd_config.training = cmd_config.get('training', {})
        cmd_config.training.output_dir = args.output_dir

    # Dataloader config
    if args.num_workers:
        cmd_config.dataloader = cmd_config.get('dataloader', {})
        cmd_config.dataloader.num_workers = args.num_workers

    # Distributed config
    if args.n_gpu:
        cmd_config.distributed = cmd_config.get('distributed', {})
        cmd_config.distributed.n_gpu = args.n_gpu
    if args.fp16:
        cmd_config.distributed = cmd_config.get('distributed', {})
        cmd_config.distributed.fp16 = args.fp16
    if args.local_rank != -1:
        cmd_config.distributed = cmd_config.get('distributed', {})
        cmd_config.distributed.local_rank = args.local_rank

    # Merge với config chính
    config = merge_configs(config, cmd_config)

    return args, config


def update_config_for_distributed(config: DictConfig) -> DictConfig:
    """
    Cập nhật cấu hình cho môi trường huấn luyện phân tán.

    Args:
        config: Cấu hình hiện tại

    Returns:
        DictConfig: Cấu hình đã cập nhật
    """
    # Lấy local_rank từ môi trường nếu khả dụng
    if "LOCAL_RANK" in os.environ:
        env_local_rank = int(os.environ["LOCAL_RANK"])
        if config.distributed.local_rank == -1:
            config.distributed.local_rank = env_local_rank

    # Xác định số lượng GPU nếu n_gpu = -1 (sử dụng tất cả GPU)
    if config.distributed.n_gpu == -1:
        import torch
        config.distributed.n_gpu = torch.cuda.device_count()

    # Điều chỉnh batch size và gradient accumulation steps nếu có nhiều GPU
    if config.distributed.n_gpu > 1:
        # Đảm bảo batch size toàn cục nhất quán
        original_global_batch_size = (
            config.training.per_device_train_batch_size *
            config.training.gradient_accumulation_steps
        )

        # Điều chỉnh để có cùng batch size toàn cục
        if config.training.gradient_accumulation_steps > 1:
            new_grad_acc_steps = max(
                config.training.gradient_accumulation_steps // config.distributed.n_gpu,
                1
            )
            new_per_device_batch_size = (
                original_global_batch_size //
                (new_grad_acc_steps * config.distributed.n_gpu)
            )

            config.training.gradient_accumulation_steps = new_grad_acc_steps
            config.training.per_device_train_batch_size = new_per_device_batch_size

            print(f"Điều chỉnh batch size cho {config.distributed.n_gpu} GPU:")
            print(f"  per_device_train_batch_size = {new_per_device_batch_size}")
            print(f"  gradient_accumulation_steps = {new_grad_acc_steps}")
            print(f"  global_batch_size = {new_per_device_batch_size * new_grad_acc_steps * config.distributed.n_gpu}")

    return config


def print_config_summary(config: DictConfig) -> None:
    """In tóm tắt cấu hình cho người dùng."""
    print("\n=== Cấu hình ASR tiếng Việt ===")
    print(f"Model cơ sở: {config.model.base_model}")
    print(f"Dữ liệu: {config.data.data_dir}")
    print(f"Metadata: {config.data.metadata_file}")
    print(f"Output: {config.training.output_dir}")
    print(f"Huấn luyện:")
    print(f"  - Epochs: {config.training.num_train_epochs}")
    print(f"  - Batch size: {config.training.per_device_train_batch_size}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Gradient accumulation: {config.training.gradient_accumulation_steps}")

    print(f"Tài nguyên:")
    print(f"  - Số GPU: {config.distributed.n_gpu}")
    print(f"  - Workers: {config.dataloader.num_workers}")
    print(f"  - FP16: {config.distributed.fp16}")

    global_batch_size = (
        config.training.per_device_train_batch_size *
        config.training.gradient_accumulation_steps *
        max(1, config.distributed.n_gpu)
    )
    print(f"  - Global batch size: {global_batch_size}")
    print("===============================")