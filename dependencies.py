import torch
import os


# Retrieve the installed PyTorch version
torch_version = torch.__version__

# Define the packages to install
packages = [
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-geometric"
]

# Uninstall and then install the packages
for package in packages:
    install_command = f"pip install {package} -f https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
    os.system(install_command)    # Install the package