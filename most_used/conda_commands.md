## download conda
- [download](https://conda.io/miniconda.html)
	- mac or windows or linux

## Most used conda commands

1. check version: `conda --version`
1. update conda: `conda update conda`, install outside env

## Environment
1. create new env with a new python version: `conda create --name py3.6 python=3.6 anaconda`
	- `conda create -n py27 python=2.7 anaconda`
	- update python of the same version: `conda update python`
1. check env: `conda info --envs` or `conda env list`
1. clone or copy an env: `conda create --name new_env --clone existed_env`
1. remove an env: `conda remove --name old_env --all`

## share and create environment with a file
1. share your env: `conda env export > environment.yml`, given you are inside your env
2. create the env: `conda env create -f environment.yml`

## create an exact environment with a file
2. check libraries in env: `conda list`
2. create identical env list: `conda list --explicit > spec-file.txt`
2. create such an identical env: `conda create --name MyEnvironment --file spec-file.txt`
2. add new libraries to an existing env: `conda install --name MyEnvironment --file spec-file.txt`
2. how to copy an env?

### 下载方法
1. cmd
2. activate deep_trade
3. conda install numpy matplotlib bcolz
