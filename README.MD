# Vietnamese ASR Fine-tuning

Dự án cung cấp bộ công cụ toàn diện để fine-tuning mô hình nhận dạng tiếng nói tiếng Việt dựa trên Wav2Vec2. Với thiết kế có tính mô-đun cao, dự án cho phép điều chỉnh chuyên sâu các tham số như số lượng worker và GPU, giúp tận dụng tối đa tài nguyên máy tính hiện có.

## Tính năng chính

- **Mô-đun hóa cao**: Thiết kế theo hướng đối tượng, dễ dàng mở rộng và tùy chỉnh.
- **Hỗ trợ huấn luyện đa GPU**: Tối ưu hóa cho môi trường đa GPU với các cấu hình phân tán.
- **Cấu hình linh hoạt**: Quản lý cấu hình thông qua file YAML, dễ dàng điều chỉnh.
- **Báo cáo tiến trình chi tiết**: Hiển thị WER (Word Error Rate) và so sánh kết quả sau mỗi đánh giá.
- **Phân tích lỗi**: Công cụ phân tích lỗi giúp xác định vấn đề và cải thiện mô hình.
- **Xử lý dữ liệu hiệu quả**: Tối ưu hóa việc xử lý dữ liệu với đa luồng.
- **Các script tiện ích**: Các script độc lập để huấn luyện, đánh giá và phiên âm.

## Cài đặt

### Yêu cầu

- Python 3.7+
- PyTorch 1.10+
- CUDA (khuyến nghị cho huấn luyện)

### Cài đặt từ source

```bash
# Clone repo
git clone https://github.com/your-username/vietnamese-asr-finetuning.git
cd vietnamese-asr-finetuning

# Cài đặt package và các dependency
pip install -e .
```

## Chuẩn bị dữ liệu

Dự án yêu cầu dữ liệu trong cấu trúc sau:

1. Thư mục chứa các file âm thanh
2. File CSV metadata chứa đường dẫn đến file âm thanh và văn bản tương ứng

Ví dụ file metadata:

```csv
audio_path,transcription
audio/sample1.wav,hôm nay trời đẹp quá
audio/sample2.wav,chúc bạn một ngày tốt lành
```

## Hướng dẫn sử dụng

### Fine-tuning mô hình

```bash
# Fine-tuning với cấu hình mặc định
python -m scripts.train \
    --config configs/default_config.yaml \
    --data_dir /đường/dẫn/đến/thư_mục_audio \
    --metadata_file /đường/dẫn/đến/metadata.csv \
    --output_dir ./vietnamese-asr-finetuned

# Fine-tuning với đa GPU
python -m scripts.train \
    --config configs/multi_gpu.yaml \
    --data_dir /đường/dẫn/đến/thư_mục_audio \
    --metadata_file /đường/dẫn/đến/metadata.csv \
    --n_gpu -1  # Sử dụng tất cả GPU
```

### Đánh giá mô hình

```bash
python -m scripts.evaluate \
    --model_path ./vietnamese-asr-finetuned \
    --data_dir /đường/dẫn/đến/thư_mục_audio \
    --metadata_file /đường/dẫn/đến/metadata.csv \
    --analyze_errors  # Thêm phân tích lỗi
```

### Chuyển đổi giọng nói thành văn bản

```bash
# Chuyển đổi một file
python -m scripts.transcribe \
    --model_path ./vietnamese-asr-finetuned \
    --audio_path /đường/dẫn/đến/audio.wav

# Chuyển đổi nhiều file trong thư mục
python -m scripts.transcribe \
    --model_path ./vietnamese-asr-finetuned \
    --audio_dir /đường/dẫn/đến/thư_mục_audio \
    --output_format json
```

## Điều chỉnh cấu hình

Dự án sử dụng file cấu hình YAML để quản lý các tham số. Bạn có thể tùy chỉnh các cấu hình trong thư mục `configs/`:

### Điều chỉnh số lượng worker và GPU

```yaml
dataloader:
  num_workers: 8  # Số lượng worker cho dataloader
  pin_memory: true
  prefetch_factor: 4

distributed:
  fp16: true
  n_gpu: 4  # Số lượng GPU sử dụng (-1 để sử dụng tất cả)
  ddp_backend: "nccl"  # nccl cho GPU, gloo cho CPU
```

### Điều chỉnh tham số huấn luyện

```yaml
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  num_train_epochs: 30
  learning_rate: 3e-4
  warmup_steps: 500
  gradient_checkpointing: true
```

## Cấu trúc dự án

```
vietnamese_asr_finetuning/
│
├── requirements.txt           # Các gói phụ thuộc
├── setup.py                   # Cài đặt package
├── README.md                  # Hướng dẫn sử dụng
│
├── vietnamese_asr/            # Module chính
│   ├── __init__.py
│   ├── config.py              # Quản lý cấu hình
│   ├── data.py                # Xử lý dữ liệu
│   ├── model.py               # Quản lý mô hình
│   ├── trainer.py             # Quản lý huấn luyện
│   ├── metrics.py             # Đo lường hiệu suất
│   └── utils.py               # Tiện ích
│
├── scripts/                   # Các script chạy
│   ├── train.py               # Script huấn luyện
│   ├── evaluate.py            # Script đánh giá
│   └── transcribe.py          # Script dự đoán
│
└── configs/                   # Cấu hình mẫu
    ├── default_config.yaml    # Cấu hình mặc định
    ├── large_dataset.yaml     # Cấu hình cho dataset lớn
    └── multi_gpu.yaml         # Cấu hình cho nhiều GPU
```

## Lưu ý quan trọng

1. Mô hình này fine-tune dựa trên `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, giữ nguyên cấu trúc các file cấu hình gốc.
2. Audio input nên có tần số lấy mẫu 16kHz để có kết quả tốt nhất.
3. Với dataset lớn, khuyến nghị sử dụng cấu hình `multi_gpu.yaml` và bật chế độ `gradient_checkpointing` để tối ưu bộ nhớ.
4. Quá trình huấn luyện sẽ tự động lưu mô hình tốt nhất dựa trên WER trên tập validation.

## Giấy phép

[MIT License](LICENSE)
