Step by step tutorial to install FastMoE on your local machine:

1. First of all you'll need to check your torch and nccl version, make sure to have a CUDA version compatible to the one torch was compiled (in general if you have the latest torch version it works also with the latest CUDA):
```
# go in terminal and use this command, the output should be something like this:

python -c  'import torch; print(torch.__version__); print(torch.cuda.nccl.version())'
>>> 2.0.1+cu117
>>> (2, 14, 3)  # -> this means version 2.14.3

# to check cuda version you can use one of this two options with a similar output,
# the binary path (second option) might be needed for troubleshooting:

nvcc --version
>>> Cuda compilation tools, release 11.7, V11.7.99
>>> Build cuda_11.7.r11.7/compiler.31442593_0

which nvcc
>>> /usr/local/cuda-11.7/bin/nvcc
```
2. An extra NCCL developer package is needed to enable the distributed features of FastMoE at the following link: https://developer.nvidia.com/nccl/nccl-legacy-downloads. Make sure to follow this steps:
```
# following the previous example I'll consider the version 2.14.3 with a CUDA version <= System CUDA version and Ubuntu 20.04
# the first command is different depending on the system and the version, just paste it from the site

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# don't forget to install the package, this command is difficult to see as it is written at the end
# outside the code block for each different installation

sudo apt install libnccl2=2.14.3-1+cuda11.7 libnccl-dev=2.14.3-1+cuda11.7 
```

3. Now you can clone the repository and enter the folder to launch the installation script as follows:
```
# clone repo and move into the folder

git clone https://github.com/laekov/fastmoe.git
cd fastmoe

# Option 1: disabling distributed features

USE_NCCL=0 python setup.py install

# Option 2: enabling distributed features

python setup.py install
```

#### Troubleshooting

If you have errors (warnings are OK) during the compilation make sure that the installer has the correct flags, this can be seen in the error as `-I/path/to/xxx/bin` and `-L/path/to/xxx/lib`. This flags should point to the correct CUDA for which all the other packages are compatible (torch and NCCL), if this paths are not correct you'll have to tell the system explicitly which CUDA version you want to use. Simple solutions could be this:
```
# (suggested) export the correct paths before compiling

export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib:$LD_LIBRARY_PATH"
python setup.py install

# eventually add these to your ~/.bashrc as an option to reduce future works

nano ~/.bashrc
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib:$LD_LIBRARY_PATH"
source ~/.bashrc
```
