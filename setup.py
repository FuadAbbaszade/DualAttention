"""Setup script for Dual-Attention Whisper."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Dual-Attention Mechanism for Noise-Robust Whisper-Based Speech Recognition"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="dual-attention-whisper",
    version="0.1.0",
    author="Fuad Abbaszade",
    description="Dual-Attention Mechanism for Noise-Robust Whisper-Based Speech Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dual-attention-whisper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dual-whisper-train=scripts.train:main",
            "dual-whisper-inference=scripts.inference:main",
            "dual-whisper-evaluate=scripts.evaluate:main",
        ],
    },
)
