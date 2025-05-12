#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module xử lý dữ liệu cho dự án ASR tiếng Việt."""

import os
import re
import pandas as pd
import random
from typing import Dict, List, Optional, Union, Tuple
from omegaconf import DictConfig

from datasets import Dataset, Audio, load_dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

class VietnameseASRDataModule:
    """
    Module quản lý và xử lý dữ liệu cho ASR tiếng Việt.
    """

    def __init__(self, config: DictConfig, processor: Wav2Vec2Processor = None):
        """
        Khởi tạo module dữ liệu.

        Args:
            config: Cấu hình cho module dữ liệu.
            processor: Wav2Vec2Processor để xử lý audio và text.
        """
        self.config = config
        self.processor = processor
        self.train_dataset = None
        self.val_dataset = None
        self.data_collator = None

        # Regex để xóa ký tự đặc biệt
        self.chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\"\%\'\"\�\']'

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Nạp dữ liệu từ thư mục và file metadata.

        Returns:
            Tuple[Dataset, Dataset]: (train_dataset, val_dataset)
        """
        print(f"Đang nạp dữ liệu từ {self.config.data.metadata_file}...")

        try:
            # Kiểm tra xem file metadata có tồn tại
            if not os.path.exists(self.config.data.metadata_file):
                raise FileNotFoundError(f"File metadata không tồn tại: {self.config.data.metadata_file}")

            # Nạp metadata
            df = pd.read_csv(self.config.data.metadata_file)

            # Kiểm tra các cột bắt buộc
            required_cols = [self.config.data.audio_column, self.config.data.text_column]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Metadata file phải chứa cột '{col}'")

            # Tạo đường dẫn đầy đủ cho các file audio
            audio_col = self.config.data.audio_column
            df[audio_col] = df[audio_col].apply(
                lambda x: os.path.join(self.config.data.data_dir, x)
                if not os.path.isabs(x) else x
            )

            # Kiểm tra các file audio có tồn tại
            missing_files = [path for path in df[audio_col] if not os.path.exists(path)]
            if missing_files:
                print(f"Cảnh báo: {len(missing_files)} file audio không tìm thấy. Ví dụ: {missing_files[:5]}")
                # Lọc ra các file không tồn tại
                df = df[~df[audio_col].isin(missing_files)]

            # Tạo dictionary cho dataset
            dataset_dict = {
                'audio': df[audio_col].tolist(),
                'transcription': df[self.config.data.text_column].tolist()
            }

            # Tạo Hugging Face dataset
            dataset = Dataset.from_dict(dataset_dict)

            # Thêm cột audio
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.data.sample_rate))

            # Chia tập train/validation
            train_val_split = dataset.train_test_split(
                test_size=self.config.data.val_split,
                seed=self.config.training.seed
            )

            train_dataset = train_val_split["train"]
            val_dataset = train_val_split["test"]

            # Giới hạn số mẫu nếu được chỉ định
            if self.config.data.max_train_samples:
                train_dataset = train_dataset.select(
                    range(min(self.config.data.max_train_samples, len(train_dataset)))
                )

            if self.config.data.max_val_samples:
                val_dataset = val_dataset.select(
                    range(min(self.config.data.max_val_samples, len(val_dataset)))
                )

            print(f"Số lượng mẫu huấn luyện: {len(train_dataset)}")
            print(f"Số lượng mẫu kiểm tra: {len(val_dataset)}")

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

            return train_dataset, val_dataset

        except Exception as e:
            print(f"Lỗi khi nạp dữ liệu: {e}")
            raise

    def show_samples(self, num_examples: int = 5) -> pd.DataFrame:
        """
        Hiển thị một số mẫu ngẫu nhiên từ tập huấn luyện.

        Args:
            num_examples: Số mẫu cần hiển thị.

        Returns:
            pd.DataFrame: DataFrame chứa các mẫu.
        """
        if self.train_dataset is None:
            raise ValueError("Tập huấn luyện chưa được nạp. Hãy gọi load_data() trước.")

        assert num_examples <= len(self.train_dataset), "Không thể chọn nhiều mẫu hơn số mẫu có sẵn."
        picks = random.sample(range(len(self.train_dataset)), num_examples)
        df = pd.DataFrame([self.train_dataset[i] for i in picks])
        return df[['transcription']]

    def remove_special_characters(self, batch):
        """
        Làm sạch transcription bằng cách xóa ký tự đặc biệt.

        Args:
            batch: Batch dữ liệu cần xử lý.

        Returns:
            batch: Batch đã xử lý.
        """
        batch["transcription"] = re.sub(self.chars_to_remove_regex, '', batch["transcription"]).lower()
        return batch

    def extract_all_chars(self, batch):
        """
        Trích xuất tất cả các ký tự duy nhất từ transcription.

        Args:
            batch: Batch dữ liệu.

        Returns:
            Dict: Dictionary chứa vocabulary và toàn bộ văn bản.
        """
        all_text = " ".join(batch["transcription"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    def prepare_dataset(self, batch):
        """
        Chuẩn bị dataset cho việc huấn luyện.

        Args:
            batch: Batch dữ liệu cần xử lý.

        Returns:
            batch: Batch đã xử lý.
        """
        if self.processor is None:
            raise ValueError("Processor chưa được thiết lập. Hãy cung cấp processor trước khi gọi hàm này.")

        audio = batch["audio"]

        # Trích xuất input values
        batch["input_values"] = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_values[0]

        # Tính toán độ dài input
        batch["input_length"] = len(batch["input_values"])

        # Trích xuất labels
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["transcription"]).input_ids

        return batch

    def process_data(self) -> Tuple[Dataset, Dataset]:
        """
        Xử lý dữ liệu cho huấn luyện.

        Returns:
            Tuple[Dataset, Dataset]: (processed_train_dataset, processed_val_dataset)
        """
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets chưa được nạp. Hãy gọi load_data() trước.")

        if self.processor is None:
            raise ValueError("Processor chưa được thiết lập. Hãy cung cấp processor trước khi gọi hàm này.")

        print("Đang xử lý dữ liệu cho huấn luyện...")

        # Làm sạch transcriptions
        self.train_dataset = self.train_dataset.map(self.remove_special_characters)
        self.val_dataset = self.val_dataset.map(self.remove_special_characters)

        # Áp dụng tiền xử lý
        num_proc = min(self.config.dataloader.num_workers, 4)  # Sử dụng tối đa 4 workers cho xử lý

        self.train_dataset = self.train_dataset.map(
            self.prepare_dataset,
            remove_columns=self.train_dataset.column_names,
            num_proc=num_proc
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_dataset,
            remove_columns=self.val_dataset.column_names,
            num_proc=num_proc
        )

        print(f"Số lượng mẫu huấn luyện đã xử lý: {len(self.train_dataset)}")
        print(f"Số lượng mẫu kiểm tra đã xử lý: {len(self.val_dataset)}")

        return self.train_dataset, self.val_dataset

    def create_data_collator(self, processor: Wav2Vec2Processor = None):
        """
        Tạo data collator cho việc padding batch.

        Args:
            processor: Processor để sử dụng (nếu chưa được thiết lập).

        Returns:
            DataCollatorCTCWithPadding: Data collator.
        """
        if processor is not None:
            self.processor = processor

        if self.processor is None:
            raise ValueError("Processor chưa được thiết lập. Hãy cung cấp processor.")

        from vietnamese_asr.model import DataCollatorCTCWithPadding

        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        return self.data_collator

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Tạo DataLoader cho tập huấn luyện và kiểm tra.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_dataloader, val_dataloader)
        """
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets chưa được xử lý. Hãy gọi process_data() trước.")

        if self.data_collator is None:
            self.create_data_collator()

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=self.config.dataloader.persistent_workers if self.config.dataloader.num_workers > 0 else False,
        )

        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=self.config.dataloader.persistent_workers if self.config.dataloader.num_workers > 0 else False,
        )

        return train_dataloader, val_dataloader