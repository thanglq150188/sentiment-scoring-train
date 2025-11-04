"""Setup script for LLM fine-tuning package"""

from setuptools import setup, find_packages

setup(
    name="llm-finetuning",
    version="1.0.0",
    description="Modular LLM fine-tuning pipeline with Unsloth",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "trl>=0.7.0",
        "unsloth",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm-train=src.cli:main",
        ],
    },
)
