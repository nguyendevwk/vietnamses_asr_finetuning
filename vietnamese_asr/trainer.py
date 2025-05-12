#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module quản lý huấn luyện cho dự án ASR tiếng Việt."""

import os
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from omegaconf import DictConfig
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from datasets import Dataset
import evaluate


class WerCallback(TrainerCallback):
    """
    Callback để log Word Error Rate sau mỗi lần đánh giá.
    """

    def __init__(self, processor, eval_dataset, test_samples=5):
        """
        Khởi tạo callback.

        Args:
            processor: Processor để xử lý audio và text.
            eval_dataset: Tập đánh giá.
            test_samples: Số lượng mẫu để hiển thị kết quả.
        """
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.test_samples = min(test_samples, len(eval_dataset))
        self.wer_metric = evaluate.load("wer")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Được gọi sau mỗi lần đánh giá.

        Args:
            args: TrainingArguments.
            state: TrainerState.
            control: TrainerControl.
            metrics: Các metrics từ đánh giá.
            **kwargs: Các tham số khác.
        """
        if metrics is not None and "eval_wer" in metrics:
            wer = metrics["eval_wer"]
            step = state.global_step
            print(f"\n===== Đánh giá tại bước {step} =====")
            print(f"Word Error Rate (WER): {wer:.4f}")

            # Kiểm tra xem có metrics của epoch trước đó không
            history = state.log_history
            prev_wers = [log.get('eval_wer', None) for log in history if 'eval_wer' in log]
            if len(prev_wers) > 1:  # Nếu có ít nhất 2 WER (bao gồm WER hiện tại)
                best_wer = min(prev_wers)
                if wer == best_wer:
                    print(f"=> Đây là WER tốt nhất đến hiện tại!")
                else:
                    print(f"=> WER tốt nhất đến hiện tại: {best_wer:.4f}")

                # Tính toán cải thiện so với lần đánh giá trước
                prev_wer = prev_wers[-2]  # WER từ lần đánh giá trước
                improvement = prev_wer - wer
                improvement_percent = (improvement / prev_wer) * 100 if prev_wer > 0 else 0

                if improvement > 0:
                    print(f"=> Cải thiện: {improvement:.4f} ({improvement_percent:.2f}%)")
                else:
                    print(f"=> Thay đổi: {improvement:.4f} ({improvement_percent:.2f}%)")

            # Hiển thị một số mẫu ngẫu nhiên để so sánh
            print("\nSo sánh kết quả trên một số mẫu:")
            model = kwargs.get('model', None)
            if model is None:
                model = kwargs.get('trainer', None).model

            if model is not None:
                sample_indices = random.sample(range(len(self.eval_dataset)), min(self.test_samples, 3))

                for i, idx in enumerate(sample_indices):
                    test_audio = self.eval_dataset[idx]["input_values"]
                    test_label = self.eval_dataset[idx]["labels"]

                    # Giải mã nhãn tham chiếu
                    with self.processor.as_target_processor():
                        test_label_str = self.processor.batch_decode([test_label], group_tokens=False)[0]

                    # Lấy dự đoán
                    inputs = torch.tensor([test_audio]).to(model.device)
                    with torch.no_grad():
                        logits = model(inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    test_pred_str = self.processor.batch_decode(predicted_ids)[0]

                    # Tính WER cho mẫu này
                    sample_wer = self.wer_metric.compute(predictions=[test_pred_str], references=[test_label_str])

                    print(f"\nMẫu {i+1}:")
                    print(f"Tham chiếu: {test_label_str}")
                    print(f"Dự đoán   : {test_pred_str}")
                    print(f"WER       : {sample_wer:.4f}")

            print("====================================\n")


class SaveBestModelCallback(TrainerCallback):
    """
    Callback để lưu mô hình tốt nhất dựa trên một metric.
    """

    def __init__(self, metric_name="eval_wer", greater_is_better=False):
        """
        Khởi tạo callback.

        Args:
            metric_name: Tên của metric để theo dõi.
            greater_is_better: True nếu metric càng lớn càng tốt, False nếu ngược lại.
        """
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Được gọi sau mỗi lần đánh giá.

        Args:
            args: TrainingArguments.
            state: TrainerState.
            control: TrainerControl.
            metrics: Các metrics từ đánh giá.
            **kwargs: Các tham số khác.
        """
        if metrics is None or self.metric_name not in metrics:
            return

        metric_value = metrics[self.metric_name]

        # Kiểm tra xem đây có phải là metric tốt nhất không
        if self.best_metric is None:
            is_best = True
        elif self.greater_is_better:
            is_best = metric_value > self.best_metric
        else:
            is_best = metric_value < self.best_metric

        # Lưu mô hình tốt nhất
        if is_best:
            self.best_metric = metric_value

            # Tạo checkpoint mới cho mô hình tốt nhất
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-best"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)

            trainer = kwargs.get('trainer', None)
            if trainer is not None:
                if hasattr(trainer, 'model'):
                    # Lưu mô hình
                    trainer.save_model(output_dir)

                    # Lưu tokenizer và các cấu hình
                    if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                        trainer.tokenizer.save_pretrained(output_dir)

                    # Lưu processor nếu có
                    if hasattr(trainer, 'processor') and trainer.processor is not None:
                        trainer.processor.save_pretrained(output_dir)

                    # Lưu trạng thái huấn luyện
                    trainer.save_state()

                    # Ghi lại metric tốt nhất
                    with open(os.path.join(output_dir, "best_metric.txt"), "w") as f:
                        f.write(f"{self.metric_name}: {self.best_metric}")

                    print(f"\n=== Đã lưu mô hình tốt nhất với {self.metric_name} = {self.best_metric} vào {output_dir} ===\n")

                    self.best_checkpoint = output_dir


class VietnameseASRTrainer:
    """
    Quản lý quá trình huấn luyện mô hình ASR tiếng Việt.
    """

    def __init__(self, config: DictConfig, model=None, processor=None,
                 train_dataset=None, eval_dataset=None, data_collator=None):
        """
        Khởi tạo trainer.

        Args:
            config: Cấu hình cho trainer.
            model: Mô hình ASR.
            processor: Processor để xử lý audio và text.
            train_dataset: Tập huấn luyện.
            eval_dataset: Tập đánh giá.
            data_collator: Data collator.
        """
        self.config = config
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.trainer = None
        self.training_args = None
        self.callbacks = []

    def create_training_args(self) -> TrainingArguments:
        """
        Tạo training arguments từ cấu hình.

        Returns:
            TrainingArguments: Các tham số huấn luyện.
        """
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(self.config.training.output_dir, exist_ok=True)

        # Thiết lập các tham số huấn luyện
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            group_by_length=self.config.training.group_by_length,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            evaluation_strategy=self.config.evaluation.eval_strategy,
            num_train_epochs=self.config.training.num_train_epochs,
            max_steps=self.config.training.max_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            fp16=self.config.distributed.fp16,
            bf16=self.config.distributed.bf16,
            save_steps=self.config.evaluation.save_steps,
            eval_steps=self.config.evaluation.eval_steps,
            logging_steps=self.config.evaluation.logging_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            save_total_limit=self.config.evaluation.save_total_limit,
            push_to_hub=False,
            remove_unused_columns=False,
            report_to=self.config.reporting.report_to,
            load_best_model_at_end=self.config.evaluation.load_best_model_at_end,
            metric_for_best_model=self.config.evaluation.metric_for_best_model,
            greater_is_better=self.config.evaluation.greater_is_better,
            seed=self.config.training.seed,
            dataloader_num_workers=self.config.dataloader.num_workers,
            local_rank=self.config.distributed.local_rank,
            ddp_find_unused_parameters=self.config.distributed.get("ddp_find_unused_parameters", False),
            dataloader_pin_memory=self.config.dataloader.pin_memory,
            # Optimizer params
            weight_decay=self.config.optimizer.weight_decay,
            adam_beta1=self.config.optimizer.adam_beta1,
            adam_beta2=self.config.optimizer.adam_beta2,
            adam_epsilon=self.config.optimizer.adam_epsilon,
            max_grad_norm=self.config.optimizer.max_grad_norm,
            optim=self.config.optimizer.name,
            # Learning rate scheduler params
            # lr_scheduler_type=self.config.lr_scheduler.name,
            # num_cycles=self.config.lr_scheduler.num_cycles,
            # power=self.config.lr_scheduler.power,
        )

        # Thiết lập tên cho run
        if self.config.reporting.run_name:
            training_args.run_name = self.config.reporting.run_name

        self.training_args = training_args
        return training_args

    def compute_metrics(self, pred):
        """
        Tính toán metric Word Error Rate.

        Args:
            pred: Dự đoán từ mô hình.

        Returns:
            Dict: Dictionary chứa metric WER.
        """
        wer_metric = evaluate.load("wer")

        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Thay thế -100 với pad token ID
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Giải mã dự đoán và nhãn
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        # Tính WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def add_callbacks(self):
        """Thêm các callback cho trainer."""
        # Callback để log WER sau mỗi lần đánh giá
        wer_callback = WerCallback(
            processor=self.processor,
            eval_dataset=self.eval_dataset
        )
        self.callbacks.append(wer_callback)

        # Callback để lưu mô hình tốt nhất
        save_best_model_callback = SaveBestModelCallback(
            metric_name="eval_wer",
            greater_is_better=False
        )
        self.callbacks.append(save_best_model_callback)

        # Early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.001
        )
        self.callbacks.append(early_stopping_callback)

    def create_trainer(self) -> Trainer:
        """
        Tạo và trả về Trainer.

        Returns:
            Trainer: Trainer đã khởi tạo.
        """
        if self.training_args is None:
            self.create_training_args()

        # Thêm các callback
        self.add_callbacks()

        self.trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor,
            callbacks=self.callbacks
        )

        # Thêm processor vào trainer instance để có thể sử dụng trong callbacks
        self.trainer.processor = self.processor

        return self.trainer

    def train(self) -> Tuple[Dict[str, float], str]:
        """
        Huấn luyện mô hình.

        Returns:
            Tuple[Dict[str, float], str]: (metrics, best_checkpoint_path)
        """
        if self.trainer is None:
            self.create_trainer()

        print("\nBắt đầu huấn luyện...")
        result = self.trainer.train()

        # Lấy kết quả đánh giá cuối cùng
        print("\nĐánh giá cuối cùng...")
        metrics = self.trainer.evaluate()
        print(f"Kết quả đánh giá cuối cùng: {metrics}")

        # Lưu mô hình cuối cùng
        final_output_dir = os.path.join(self.config.training.output_dir, "final")
        self.trainer.save_model(final_output_dir)
        self.processor.save_pretrained(final_output_dir)
        print(f"Đã lưu mô hình cuối cùng vào {final_output_dir}")

        # Tìm mô hình tốt nhất
        best_checkpoint = None
        for callback in self.callbacks:
            if isinstance(callback, SaveBestModelCallback):
                best_checkpoint = callback.best_checkpoint
                break

        return metrics, best_checkpoint or final_output_dir