from setuptools import setup, find_packages

setup(
    name="improved-diffusion",
    version="0.1",
    packages=find_packages(),  # Automatically finds all packages and subpackages
    install_requires=[
        "blobfile>=1.0.5",
        "torch",
        "tqdm",
        "glasses==0.0.6",
        "pillow", 
        "pandas",
        "matplotlib",  
        "torchmetrics",  
        "torchvision",  
        "torchsummary",  
        "torchinfo",  
        "requests",  
        "opencv-python",  
        "einops",  
        "scikit-learn",
        "tensorflow",
        "tensorboard"
    ],
)
