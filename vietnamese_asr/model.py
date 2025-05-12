#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module quản lý mô hình cho dự án ASR tiếng Việt."""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from omegaconf import DictConfig

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config
)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator sẽ tự động padding các input nhận được.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Tách input và label vì chúng có độ dài khác nhau và cần padding khác nhau
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Padding inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Padding labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Thay thế padding bằng -100 để bỏ qua loss đúng cách
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


class VietnameseASRModelHandler:
    """
    Quản lý mô hình ASR tiếng Việt.
    """

    def __init__(self, config: DictConfig):
        """
        Khởi tạo handler mô hình.

        Args:
            config: Cấu hình cho mô hình.
        """
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        self.device = None

    def setup_device(self) -> torch.device:
        """
        Thiết lập device cho mô hình.

        Returns:
            torch.device: Device đã thiết lập.
        """
        if self.config.distributed.n_gpu > 0 and torch.cuda.is_available():
            if self.config.distributed.local_rank == -1:
                # Single GPU hoặc không sử dụng huấn luyện phân tán
                device = torch.device("cuda:0")
            else:
                # Huấn luyện phân tán
                torch.cuda.set_device(self.config.distributed.local_rank)
                device = torch.device("cuda", self.config.distributed.local_rank)
                # Khởi tạo process group
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(
                        backend=self.config.distributed.ddp_backend
                    )
        else:
            device = torch.device("cpu")

        self.device = device
        print(f"Sử dụng device: {device}")
        return device

    def load_processor(self) -> Wav2Vec2Processor:
        """
        Nạp processor từ mô hình cơ sở.

        Returns:
            Wav2Vec2Processor: Processor đã nạp.
        """
        print(f"Đang nạp processor từ {self.config.model.base_model}...")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.config.model.base_model)
            self.tokenizer = self.processor.tokenizer
            self.feature_extractor = self.processor.feature_extractor

            return self.processor
        except Exception as e:
            print(f"Lỗi khi nạp processor: {e}")
            raise

    def load_model(self) -> Wav2Vec2ForCTC:
        """
        Nạp mô hình Wav2Vec2 từ mô hình cơ sở hoặc từ checkpoint.

        Returns:
            Wav2Vec2ForCTC: Mô hình đã nạp.
        """
        if self.processor is None:
            self.load_processor()

        print(f"Đang nạp mô hình từ {self.config.model.base_model}...")

        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.config.model.base_model,
                attention_dropout=self.config.model.attention_dropout,
                hidden_dropout=self.config.model.hidden_dropout,
                feat_proj_dropout=self.config.model.feat_proj_dropout,
                mask_time_prob=self.config.model.mask_time_prob,
                layerdrop=self.config.model.layerdrop,
                ctc_loss_reduction="mean",
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=len(self.processor.tokenizer),
                ignore_mismatched_sizes=True
            )

            # Đóng băng feature encoder nếu được chỉ định
            if self.config.model.freeze_feature_encoder:
                self.model.freeze_feature_extractor()

            # Đóng băng toàn bộ mô hình cơ sở nếu được chỉ định
            if self.config.model.freeze_base_model:
                for param in self.model.wav2vec2.parameters():
                    param.requires_grad = False

            # Chuyển mô hình sang device tương ứng
            if self.device is None:
                self.setup_device()

            self.model = self.model.to(self.device)

            # Nếu sử dụng huấn luyện phân tán
            if self.config.distributed.local_rank != -1 and self.config.distributed.n_gpu > 0:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.config.distributed.local_rank],
                    output_device=self.config.distributed.local_rank,
                    find_unused_parameters=self.config.distributed.get("ddp_find_unused_parameters", False),
                    bucket_cap_mb=self.config.distributed.get("ddp_bucket_cap_mb", 25),
                )
            # Nếu có nhiều GPU nhưng không phải huấn luyện phân tán
            elif self.config.distributed.n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)

            return self.model

        except Exception as e:
            print(f"Lỗi khi nạp mô hình: {e}")
            raise

    def save_model(self, output_dir: str = None) -> None:
        """
        Lưu mô hình và processor.

        Args:
            output_dir: Thư mục đầu ra. Nếu None, sử dụng thư mục từ config.
        """
        if output_dir is None:
            output_dir = self.config.training.output_dir

        os.makedirs(output_dir, exist_ok=True)

        print(f"Đang lưu mô hình vào {output_dir}...")

        # Lưu processor
        self.processor.save_pretrained(output_dir)

        # Lưu model weights (xử lý DistributedDataParallel hoặc DataParallel)
        if hasattr(self.model, 'module'):
            self.model.module.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)

        print(f"Mô hình và processor đã được lưu vào {output_dir}")

    def transcribe(self, audio_path: str = None, audio_array: np.ndarray = None,
                  sampling_rate: int = 16000) -> str:
        """
        Phiên âm một file audio hoặc mảng audio.

        Args:
            audio_path: Đường dẫn đến file audio.
            audio_array: Mảng audio numpy.
            sampling_rate: Tần số lấy mẫu.

        Returns:
            str: Văn bản được phiên âm.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Mô hình hoặc processor chưa được nạp.")

        if audio_path is None and audio_array is None:
            raise ValueError("Phải cung cấp hoặc audio_path hoặc audio_array.")

        # Nạp audio từ file nếu được cung cấp
        if audio_path is not None:
            import librosa
            audio_array, sample_rate = librosa.load(audio_path, sr=sampling_rate)

        # Tiền xử lý audio
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Chuyển inputs sang device tương ứng
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Lấy logits
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Lấy predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Giải mã
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]