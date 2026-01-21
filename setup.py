from setuptools import setup, find_packages

setup(
    name="center_surround",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy",
        "matplotlib",
        "optuna",
        "pyyaml",
        "pillow",
        "pandas",
        "scipy",
        "tqdm",
        "torchinfo",
    ],
    author="Ridvan Balamur",
    author_email="balamurridvan@gmail.com",
    description="A package for center-surround data processing and modeling",
)