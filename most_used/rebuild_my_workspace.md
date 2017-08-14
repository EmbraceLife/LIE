# how to rebuild my workspace

- install conda
- `conda create -n experiment3.5 python=3.5`
- `pip install pdbpp`
- `conda install scipy numpy matplotlib bcolz pandas`
- `conda install tensorflow keras`
- 如果反复遇到报错，可以试试将 `conda`换成`pip`
- `pip install tensorflow-1.3.0rc2-py3-none-any.whl`
- however, keras is still broken on preprocessing.text
- `brew install ta-lib`, `pip install TA-Lib`
- for windows, use `pip install ta-lib...whl`
- change bash_profile-pdbrc.md

## windows 安装
- conda, 下载地址https://conda.io/miniconda.html
- 打开cmd, 任意目录下安装应该都行，保险起见可以在根目录下安装
- `conda install scipy numpy matplotlib bcolz pandas`
- `conda install tensorflow keras`
- 如果反复遇到报错，可以试试将 `conda`换成`pip`
- 升级最新tensorflow，不安装这一步应该也没问题。whl文件下载地址https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows,PY=36/ 或者这里找https://github.com/tensorflow/tensorflow#installation
`pip install tensorflow-1.3.0rc2-cp36-cp36m-win_amd64.whl`
- 下载地址http://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 建议下载TA_Lib‑0.4.10‑cp36‑cp36m‑win_amd64.whl， 然后`pip install TA_Lib‑0.4.10‑cp36‑cp36m‑win_amd64.whl`
- 应该就这些了，如果有问题我们再沟通
