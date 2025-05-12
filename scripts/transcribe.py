#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chuyển đổi giọng nói tiếng Việt thành văn bản.
Sử dụng:
    python -m scripts.transcribe --model_path /đường/dẫn/đến/mô_hình --audio_path /đường/dẫn/đến/audio.wav

Hoặc để xử lý nhiều file:
    python -m scripts.transcribe --model_path /đường/dẫn/đến/mô_hình --audio_dir /đường/dẫn/đến/thư_mục_audio
"""

import os
import sys
import torch
import argparse
import time
import glob
import pandas as pd
import json
import librosa
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from vietnamese_asr.utils import format_time


def parse_args():
    """Parse arguments từ command line."""
    parser = argparse.ArgumentParser(description='Chuyển đổi giọng nói tiếng Việt thành văn bản')

    # Đường dẫn mô hình
    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn đến mô hình ASR')

    # Input audio (một trong hai tham số)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio_path', type=str,
                          help='Đường dẫn đến file âm thanh cần chuyển đổi')
    input_group.add_argument('--audio_dir', type=str,
                          help='Thư mục chứa các file âm thanh cần chuyển đổi')
    input_group.add_argument('--input_list', type=str,
                          help='File CSV chứa danh sách các file âm thanh cần chuyển đổi')

    # Tham số khác
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Thư mục lưu kết quả (mặc định: thư mục hiện tại)')
    parser.add_argument('--output_format', type=str, choices=['txt', 'json', 'csv'], default='txt',
                        help='Định dạng đầu ra (txt, json, hoặc csv)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Tần số lấy mẫu của audio')
    parser.add_argument('--device', type=str, default=None,
                        help='Device để chạy mô hình (cuda, cpu, hoặc None để tự phát hiện)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Kích thước batch cho xử lý hàng loạt')
    parser.add_argument('--file_ext', type=str, default='.wav',
                        help='Phần mở rộng của file âm thanh cần xử lý (mặc định: .wav)')

    return parser.parse_args()


def transcribe_audio(audio_path, model, processor, device, sample_rate=16000):
    """
    Chuyển đổi file âm thanh thành văn bản.

    Args:
        audio_path: Đường dẫn đến file âm thanh.
        model: Mô hình ASR.
        processor: Processor.
        device: Device để chạy mô hình.
        sample_rate: Tần số lấy mẫu.

    Returns:
        dict: Kết quả chuyển đổi với metadata.
    """
    start_time = time.time()

    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(audio_path):
            return {
                "audio_path": audio_path,
                "transcription": "",
                "error": "File không tồn tại",
                "processing_time": 0
            }

        # Nạp audio
        audio, sample_rate_orig = librosa.load(audio_path, sr=sample_rate)

        # Lấy thông tin về audio
        duration = len(audio) / sample_rate

        # Tiền xử lý audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # Chuyển inputs sang device tương ứng
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Dự đoán
        with torch.no_grad():
            logits = model(**inputs).logits

        # Lấy predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Giải mã
        transcription = processor.batch_decode(predicted_ids)[0]

        # Tính thời gian xử lý
        processing_time = time.time() - start_time
        rtf = processing_time / duration if duration > 0 else 0

        return {
            "audio_path": audio_path,
            "transcription": transcription,
            "duration": duration,
            "processing_time": processing_time,
            "rtf": rtf
        }

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "audio_path": audio_path,
            "transcription": "",
            "error": str(e),
            "processing_time": processing_time
        }


def batch_transcribe(audio_files, model, processor, device, batch_size=8, sample_rate=16000):
    """
    Chuyển đổi nhiều file âm thanh theo batch.

    Args:
        audio_files: Danh sách đường dẫn đến các file âm thanh.
        model: Mô hình ASR.
        processor: Processor.
        device: Device để chạy mô hình.
        batch_size: Kích thước batch.
        sample_rate: Tần số lấy mẫu.

    Returns:
        list: Danh sách kết quả chuyển đổi.
    """
    results = []

    # Xử lý từng file
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Đang chuyển đổi"):
        batch_files = audio_files[i:i+batch_size]

        # Hiện tại xử lý tuần tự từng file
        # TODO: Cải thiện để xử lý song song theo batch thực sự
        batch_results = []
        for audio_file in batch_files:
            result = transcribe_audio(audio_file, model, processor, device, sample_rate)
            batch_results.append(result)

        results.extend(batch_results)

    return results


def save_results(results, output_dir, output_format):
    """
    Lưu kết quả chuyển đổi.

    Args:
        results: Danh sách kết quả chuyển đổi.
        output_dir: Thư mục đầu ra.
        output_format: Định dạng đầu ra.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lưu kết quả tổng hợp
    if output_format == 'json':
        # Lưu vào file JSON
        output_file = os.path.join(output_dir, 'transcriptions.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elif output_format == 'csv':
        # Lưu vào file CSV
        output_file = os.path.join(output_dir, 'transcriptions.csv')
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)

    else:  # txt format
        # Lưu vào file TXT
        output_file = os.path.join(output_dir, 'transcriptions.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"File: {result['audio_path']}\n")
                f.write(f"Transcription: {result['transcription']}\n")
                if 'duration' in result:
                    f.write(f"Duration: {result['duration']:.2f}s\n")
                if 'processing_time' in result:
                    f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                if 'rtf' in result:
                    f.write(f"RTF: {result['rtf']:.2f}\n")
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                f.write("\n")

    # Lưu kết quả riêng lẻ cho từng file
    if len(results) > 1:
        for result in results:
            audio_path = result['audio_path']
            audio_filename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_filename)[0]

            # Tạo thư mục individual nếu chưa tồn tại
            individual_dir = os.path.join(output_dir, 'individual')
            os.makedirs(individual_dir, exist_ok=True)

            # Lưu văn bản
            if output_format == 'txt' or output_format == 'csv':
                output_file = os.path.join(individual_dir, f"{audio_name}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['transcription'])

            elif output_format == 'json':
                output_file = os.path.join(individual_dir, f"{audio_name}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả vào {output_file}")

    # Trả về file kết quả tổng hợp
    return output_file


def main():
    """Hàm main cho chuyển đổi giọng nói thành văn bản."""
    # Parse arguments
    args = parse_args()

    # Thiết lập output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "transcriptions")
    os.makedirs(args.output_dir, exist_ok=True)

    # Thiết lập device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    # Nạp mô hình và processor
    print(f"Đang nạp mô hình từ {args.model_path}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model_path)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
        model = model.to(device)
        model.eval()
        print(f"Đã nạp mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi nạp mô hình: {e}")
        sys.exit(1)

    # Lấy danh sách file audio
    audio_files = []

    if args.audio_path:
        # Xử lý một file
        if not os.path.exists(args.audio_path):
            print(f"Lỗi: File không tồn tại: {args.audio_path}")
            sys.exit(1)
        audio_files = [args.audio_path]

    elif args.audio_dir:
        # Xử lý thư mục
        if not os.path.exists(args.audio_dir):
            print(f"Lỗi: Thư mục không tồn tại: {args.audio_dir}")
            sys.exit(1)

        audio_files = glob.glob(os.path.join(args.audio_dir, f"*{args.file_ext}"))
        if not audio_files:
            print(f"Lỗi: Không tìm thấy file âm thanh nào trong thư mục {args.audio_dir}")
            sys.exit(1)

    elif args.input_list:
        # Xử lý danh sách từ file
        if not os.path.exists(args.input_list):
            print(f"Lỗi: File danh sách không tồn tại: {args.input_list}")
            sys.exit(1)

        # Xác định định dạng file
        file_ext = os.path.splitext(args.input_list)[1].lower()
        if file_ext == '.csv':
            # Nạp từ CSV
            try:
                df = pd.read_csv(args.input_list)
                # Tìm cột chứa đường dẫn audio
                audio_col = None
                for col in ['audio_path', 'audio', 'path', 'file']:
                    if col in df.columns:
                        audio_col = col
                        break

                if audio_col is None:
                    print(f"Lỗi: Không tìm thấy cột chứa đường dẫn audio trong file CSV")
                    sys.exit(1)

                audio_files = df[audio_col].tolist()
            except Exception as e:
                print(f"Lỗi khi nạp file CSV: {e}")
                sys.exit(1)

        elif file_ext == '.txt':
            # Nạp từ TXT
            try:
                with open(args.input_list, 'r', encoding='utf-8') as f:
                    audio_files = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Lỗi khi nạp file TXT: {e}")
                sys.exit(1)

        else:
            print(f"Lỗi: Định dạng file không được hỗ trợ: {file_ext}")
            sys.exit(1)

    # Kiểm tra số lượng file
    num_files = len(audio_files)
    print(f"Tìm thấy {num_files} file âm thanh cần xử lý")

    if num_files == 0:
        print("Không có file nào để xử lý")
        sys.exit(0)

    # Chuyển đổi
    start_time = time.time()

    if num_files == 1:
        # Xử lý một file
        print(f"Đang chuyển đổi file: {audio_files[0]}")
        result = transcribe_audio(audio_files[0], model, processor, device, args.sample_rate)
        results = [result]

        print("\nKết quả chuyển đổi:")
        print(f"Văn bản: {result['transcription']}")

        if 'duration' in result:
            print(f"Thời lượng: {result['duration']:.2f}s")
        if 'processing_time' in result:
            print(f"Thời gian xử lý: {result['processing_time']:.2f}s")
        if 'rtf' in result:
            print(f"Real-time factor: {result['rtf']:.2f}x")
        if 'error' in result:
            print(f"Lỗi: {result['error']}")
    else:
        # Xử lý nhiều file
        print(f"Đang chuyển đổi {num_files} file âm thanh...")
        results = batch_transcribe(audio_files, model, processor, device, args.batch_size, args.sample_rate)

    # Lưu kết quả
    output_file = save_results(results, args.output_dir, args.output_format)

    # Tính thời gian tổng
    total_time = time.time() - start_time
    total_duration = sum([r.get('duration', 0) for r in results])
    avg_rtf = total_time / total_duration if total_duration > 0 else 0

    print("\n=== Thống kê ===")
    print(f"Số lượng file: {num_files}")
    print(f"Tổng thời lượng: {total_duration:.2f}s")
    print(f"Tổng thời gian xử lý: {total_time:.2f}s")
    print(f"Real-time factor trung bình: {avg_rtf:.2f}x")
    print(f"Kết quả được lưu vào: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChuyển đổi bị dừng bởi người dùng")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\nLỗi trong quá trình chuyển đổi: {e}")
        traceback.print_exc()
        sys.exit(1)