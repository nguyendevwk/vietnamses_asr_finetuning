#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module đánh giá mô hình cho dự án ASR tiếng Việt."""

import os
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
import evaluate
from transformers import Wav2Vec2Processor
from datasets import Dataset

class VietnameseASREvaluator:
    """
    Đánh giá mô hình ASR tiếng Việt.
    """

    def __init__(self, model, processor: Wav2Vec2Processor, output_dir: str = "evaluation_results"):
        """
        Khởi tạo evaluator.

        Args:
            model: Mô hình ASR.
            processor: Processor để xử lý audio và text.
            output_dir: Thư mục đầu ra cho kết quả đánh giá.
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.device = next(model.parameters()).device

        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)

        # Tải metric
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer") if "cer" in evaluate.list_evaluation_modules() else None

    def transcribe_batch(self, batch) -> List[str]:
        """
        Phiên âm một batch dữ liệu.

        Args:
            batch: Batch dữ liệu.

        Returns:
            List[str]: Danh sách các văn bản phiên âm.
        """
        # Chuẩn bị inputs
        input_values = torch.tensor(batch["input_values"]).to(self.device)

        # Dự đoán
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Lấy predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Giải mã
        transcriptions = self.processor.batch_decode(predicted_ids)

        return transcriptions

    def evaluate_dataset(self, dataset: Dataset, batch_size: int = 8) -> Dict[str, float]:
        """
        Đánh giá trên một dataset.

        Args:
            dataset: Dataset cần đánh giá.
            batch_size: Kích thước batch.

        Returns:
            Dict[str, float]: Các metric đánh giá.
        """
        all_predictions = []
        all_references = []
        all_sample_ids = []
        all_audio_durations = []

        # Tính toán số lượng batch
        num_samples = len(dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size

        print(f"Đánh giá trên {num_samples} mẫu với batch size {batch_size}...")

        # Xử lý từng batch
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            # Lấy batch
            batch = dataset[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx

            # Dự đoán
            batch_predictions = self.transcribe_batch(batch)

            # Lấy references từ batch
            with self.processor.as_target_processor():
                batch_references = self.processor.batch_decode(
                    [batch["labels"][j] for j in range(batch_size_actual)],
                    group_tokens=False
                )

            # Thêm vào danh sách
            all_predictions.extend(batch_predictions)
            all_references.extend(batch_references)
            all_sample_ids.extend([f"sample_{start_idx + j}" for j in range(batch_size_actual)])

            # Tính độ dài audio
            if "input_length" in batch:
                all_audio_durations.extend([
                    batch["input_length"][j] / 16000 for j in range(batch_size_actual)
                ])

            # In tiến độ
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                print(f"  Đã xử lý {min((i + 1) * batch_size, num_samples)}/{num_samples} mẫu")

        # Tính metrics
        metrics = {}

        # WER - Word Error Rate
        wer = self.wer_metric.compute(predictions=all_predictions, references=all_references)
        metrics["wer"] = wer

        # CER - Character Error Rate (nếu có)
        if self.cer_metric:
            cer = self.cer_metric.compute(predictions=all_predictions, references=all_references)
            metrics["cer"] = cer

        # Tính RTF (Real-Time Factor) nếu có thông tin về độ dài audio
        if all_audio_durations:
            # TODO: Implement RTF calculation
            pass

        # Tính sample-level WER
        sample_metrics = []
        for i in range(len(all_predictions)):
            sample_wer = self.wer_metric.compute(
                predictions=[all_predictions[i]],
                references=[all_references[i]]
            )

            sample_entry = {
                "id": all_sample_ids[i],
                "reference": all_references[i],
                "prediction": all_predictions[i],
                "wer": sample_wer
            }

            if all_audio_durations and i < len(all_audio_durations):
                sample_entry["duration"] = all_audio_durations[i]

            sample_metrics.append(sample_entry)

        # Lưu kết quả chi tiết
        self.save_evaluation_results(metrics, sample_metrics)

        return metrics

    def save_evaluation_results(self, metrics: Dict[str, float], sample_metrics: List[Dict]):
        """
        Lưu kết quả đánh giá.

        Args:
            metrics: Các metric tổng thể.
            sample_metrics: Các metric cho từng mẫu.
        """
        # Lưu tóm tắt metrics
        with open(os.path.join(self.output_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Lưu metrics chi tiết theo mẫu
        df = pd.DataFrame(sample_metrics)
        df.to_csv(os.path.join(self.output_dir, "sample_metrics.csv"), index=False)

        # Tạo biểu đồ phân phối WER
        plt.figure(figsize=(10, 6))
        plt.hist(df['wer'], bins=20)
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Số lượng mẫu')
        plt.title('Phân phối WER trên tập dữ liệu')
        plt.savefig(os.path.join(self.output_dir, "wer_distribution.png"))
        plt.close()

        # Nếu có thông tin về độ dài
        if 'duration' in df.columns:
            # Tạo biểu đồ phân tán WER vs độ dài
            plt.figure(figsize=(10, 6))
            plt.scatter(df['duration'], df['wer'], alpha=0.5)
            plt.xlabel('Độ dài audio (giây)')
            plt.ylabel('Word Error Rate (WER)')
            plt.title('WER theo độ dài audio')
            plt.savefig(os.path.join(self.output_dir, "wer_vs_duration.png"))
            plt.close()

        print(f"Đã lưu kết quả đánh giá vào {self.output_dir}")

    def analyze_errors(self, num_samples: int = 20):
        """
        Phân tích các lỗi phổ biến.

        Args:
            num_samples: Số lượng mẫu để phân tích.
        """
        # Nạp kết quả mẫu
        sample_file = os.path.join(self.output_dir, "sample_metrics.csv")
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"File kết quả mẫu không tồn tại: {sample_file}")

        df = pd.read_csv(sample_file)

        # Sắp xếp theo WER giảm dần
        df = df.sort_values(by='wer', ascending=False)

        # Chọn top N mẫu có WER cao nhất
        worst_samples = df.head(num_samples)

        # Phân tích lỗi
        error_analysis = []

        for _, row in worst_samples.iterrows():
            ref = row['reference']
            pred = row['prediction']

            # Tính toán các lỗi
            import Levenshtein
            import difflib

            # Tách thành từ
            ref_words = ref.split()
            pred_words = pred.split()

            # Tìm sự khác biệt
            matcher = difflib.SequenceMatcher(None, ref_words, pred_words)
            diff = []

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    diff.append(f"Thay thế: '{' '.join(ref_words[i1:i2])}' -> '{' '.join(pred_words[j1:j2])}'")
                elif tag == 'delete':
                    diff.append(f"Xóa: '{' '.join(ref_words[i1:i2])}'")
                elif tag == 'insert':
                    diff.append(f"Chèn: '{' '.join(pred_words[j1:j2])}'")

            error_analysis.append({
                "id": row['id'],
                "reference": ref,
                "prediction": pred,
                "wer": row['wer'],
                "differences": diff
            })

        # Lưu phân tích lỗi
        with open(os.path.join(self.output_dir, "error_analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)

        # Tạo file báo cáo markdown
        with open(os.path.join(self.output_dir, "error_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Phân tích lỗi ASR tiếng Việt\n\n")

            for i, sample in enumerate(error_analysis):
                f.write(f"## Mẫu {i+1}: {sample['id']}\n\n")
                f.write(f"**WER:** {sample['wer']:.4f}\n\n")
                f.write(f"**Tham chiếu:** {sample['reference']}\n\n")
                f.write(f"**Dự đoán:** {sample['prediction']}\n\n")

                f.write("**Các lỗi:**\n\n")
                for diff in sample['differences']:
                    f.write(f"- {diff}\n")

                f.write("\n---\n\n")

        print(f"Đã lưu phân tích lỗi vào {self.output_dir}")