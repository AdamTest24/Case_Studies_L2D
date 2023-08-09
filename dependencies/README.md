# Creating virtual environments

## Conda, virtual environment and its python-based dependencies 
You might need to install [conda](https://github.com/mxochicale/code/tree/main/conda) and create [ve.yml](ve.yml).

* Some commands to manage your conda environment.
See this [conda cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) for further commands.
``` 
conda update -n base -c defaults conda  ## UPDATE CONDA
conda list -n *VE # show list of installed packages
conda env create -f *ve.yml   		    ## INSTALL
conda env update -f *ve.yml --prune  	## UPDATE
conda activate *VE			    ## ACTIVATE
conda remove -n *VE --all   ## REMOVE
```

* Creating virtual env in the terminal for codespace
```
conda update -n base pip -c conda-forge # environment location: /opt/conda/envs/eVE
pip install ipykernel
pip install torch torchvision matplotlib
```

* Quick test for the availability of cuda in your machine.
```
conda activate l2dVE
python
import torch
torch.cuda.is_available()
```

## Launch jupyter notebook
``` 
cd $HOME/../notebooks
conda activate l2dVE && jupyter notebook --browser=firefox
```

## Our code have been tested in the following machines:


### Ubuntu 22.04.1 LTS with NVIDIA RTX A2000 8GB Laptop GPU
* OS
```
$ hostnamectl

 Static hostname: --
       Icon name: computer-laptop
         Chassis: laptop
      Machine ID: --
         Boot ID: --
Operating System: Ubuntu 22.04.1 LTS              
          Kernel: Linux 5.15.0-56-generic
    Architecture: x86-64
 Hardware Vendor: --

```

* GPU
```
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Sat Dec 17 13:27:52 2022
Driver Version                            : 520.61.05
CUDA Version                              : 11.8

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA RTX A2000 8GB Laptop GPU
    Product Brand                         : NVIDIA RTX
    Product Architecture                  : Ampere

```


