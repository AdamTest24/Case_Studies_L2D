# Creating virtual environments

## Conda, virtual environment and its python-based dependencies 
You might need to install [conda](https://github.com/mxochicale/code/tree/main/conda) and create [environment.yml](environment.yml).


### In your local machine 
* Some commands to manage your conda environment.
See this [conda cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) for further commands.
``` 
conda update -n base pip -c conda-forge  
conda env create -f environment.yml
conda env update -f environment.yml --prune
conda activate l2dVE
conda list -n l2dVE # show list of installed packages
conda remove -n l2dVE --all
```

* Quick test for the availability of cuda in your machine.
```
conda activate l2dVE
python -c 'import torch; torch.cuda.is_available()'
```

## Launch jupyter notebook
``` 
cd path of your notebooks
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


### In codespace 
[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=LearnToDiscover/Case_Studies_L2D)


#### Creation of new codespace 
1. Create a new  Codespace (click the button above!) using this repo. 
2. Select repository, branch, Region and machine type (8-core [32GBRAM 64GB] if it is available)
3. (Optional) Press `Ctrl+Shit+p` to select `Add Dev Container`> `Create a new env` > Anaconda (python3) > Conda, Mamba (Miniforge) > Keep Defaul
4. Open one of the notebooks `deep_leaarning_mab>1-Intro-dl_MLPs.ipynb`
5. Click `Select kernel` button in the upper-right and select `Install and Enable suggested extensions` > `Python Env` > `base Python (opt/conda/bin/python)`
6. Have fun! 🚀

* Add packages to `base`` virtual env in the terminal for codespace
```
conda update -n base pip -c conda-forge # environment location: /opt/conda/envs/base
pip install ipykernel torch torchvision matplotlib
python -c 'import torch; torch.cuda.is_available()'
```


