import os.path
from setuptools import setup, find_packages

version = "0.0.1"

setup(
    name="common_libraries",
    version=version,
    url="https://github.com/arthurgrava/kaggle-playground/common_libraries",
    license="None",
    author="Arthur Grava",
    author_email="arthur.grava@gmail.com",
    description="Common packages for my DS stuff",
    packages=find_packages(),
)
