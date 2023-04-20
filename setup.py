import os
from setuptools import setup


def read(rel_path: str) -> str:
    """read the content of a file"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding='utf-8') as file:
        return file.read()


def get_version(rel_path: str) -> str:
    """read the version inside the __init__.py file"""
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")

install_requires = [
    "ultralytics",  # yolov8
]

setup(
    name="TACTUS - yolov8",
    version=get_version("tactus_yolov8/__init__.py"),
    description="Threatening activities classification toward users' security",
    long_description=long_description,
    packages=['tactus_yolov8'],
    classifiers=[
        "Development Status :: 5 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github/Cranfield-GDP3/TACTUS-yolov8",
    project_urls={
        "issues": "https://github/Cranfield-GDP3/TACTUS-yolov8/issues",
    },
    python_requires=">=3.9",
    install_requires=install_requires,
)
