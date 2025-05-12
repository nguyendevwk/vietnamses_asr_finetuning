from setuptools import setup, find_packages
import pathlib

# Đọc mô tả dài từ README nếu có
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="vietnamese-asr-finetuning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for fine-tuning Wav2Vec2 for Vietnamese automatic speech recognition (ASR)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vietnamese-asr-finetuning",
    project_urls={
        "Source": "https://github.com/yourusername/vietnamese-asr-finetuning",
        "Bug Tracker": "https://github.com/yourusername/vietnamese-asr-finetuning/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    python_requires=">=3.7, <4",
    install_requires=[
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0",
        "evaluate>=0.4.0",
        "jiwer>=2.3.0",
        "librosa>=0.9.1",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "tensorboard>=2.8.0",
        "fsspec==2023.6.0",
        "accelerate>=0.16.0",
        "omegaconf>=2.2.0",
        "soundfile>=0.10.3",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vnasr-train=scripts.train:main",
            "vnasr-evaluate=scripts.evaluate:main",
            "vnasr-transcribe=scripts.transcribe:main",
        ],
    },
)
