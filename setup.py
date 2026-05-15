# setup.py
from setuptools import setup, find_packages

setup(
    name="MLX",                     # Package name
    version="0.1.0",                   # Package version
    description="A collection of ML Tools",  # Brief description
    author="Daniel MacSwayne",
    author_email="dan.macswayne@gmail.com",
    url="https://github.com/Daniel-MacSwayne/MLX",  # Repo URL
    include_package_data=True,
    install_requires=[],  # Add dependencies here, if any
    packages=find_packages(),          # Automatically find packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
