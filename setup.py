#!/usr/bin/env python
# coding: utf-8

import setuptools
import os


with open("README.md", 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='unKR',
    version='1.0.3',
    author='SEUKGE',
    author_email='slchen@seu.edu.cn',
    url='https://github.com/seucoin/unKR',
    description='An Open Source Library for uncertain Knowledge Reasoning',
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pytorch_lightning==1.5.10',
        'PyYAML',
        'wandb',
        'IPython',
        'gensim'
    ],
    python_requires=">=3.6"
)
