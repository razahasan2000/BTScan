"""
BTScan: Spatially Preserved CNN Transformer Fusion for Multi-Method XAI Consensus
in Brain Tumor MRI Classification

Setup script for installing the BTScan package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="btscan",
    version="1.0.0",
    author="Raza Hasan, MD Aashif Ansari, Pratham Acharya, Salman Mahmood",
    author_email="raza.hasan@solent.ac.uk",
    description="BTScan: Spatially Preserved CNN Transformer Fusion for Multi-Method XAI Consensus in Brain Tumor MRI Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/BTScan",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "btscan-train=src.train_model:main",
            "btscan-eval=src.evaluate_model:main",
            "btscan-ablation=src.ablation_study:main",
            "btscan-xai=src.xai_consensus:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)