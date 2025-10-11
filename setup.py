"""Setup script for satellite_spectrum_emulator package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="satellite_downlink_simulator",
    version="0.1.0",
    author="Ben Mathews",
    author_email="Benjamin.L.Mathews",
    description="A tool for simulating satellite communication spectrum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ben-Mathews/satellite_downlink_simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Communications",
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
        "attrs>=21.0.0",
        "matplotlib>=3.3.0",
        "imageio>=2.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
)
