#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ddbm",
    version="7.1.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Diophantine Dynamical Boundary Method for chaos detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DDBM",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/DDBM/issues",
        "Documentation": "https://github.com/yourusername/DDBM#documentation",
        "Source Code": "https://github.com/yourusername/DDBM",
        "arXiv": "https://arxiv.org/abs/XXXX.XXXXX",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "matplotlib>=3.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "ddbm-batch=ddbm.cli:batch_main",
        ],
    },
)
