"""
Setup configuration for Hybrid Semantic Search System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-search-system",
    version="1.0.0",
    author="Ekant Chandrakar",
    author_email="ekantchandrakar07@gmail.com",
    description="Production-grade hybrid semantic search combining BM25, Bi-Encoder, and Cross-Encoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ekantchandrakar/hybrid-search-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-search=src.cli:main",
        ],
    },
)
