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
    uninstall_command = f"pip uninstall {package} -y"  # Automatically confirm the uninstallation
    install_command = f"pip install {package} -f https://data.pyg.org/whl/torch-{torch_version}+cpu.html"

    # os.system(uninstall_command)  # Uninstall the package
    os.system(install_command)    # Install the package