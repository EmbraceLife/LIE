# LIE: Learning Is Experimenting 

Do you believe this LIE?

Step by step, this repo will testify this LIE by experimenting Kur for deep learning...

## Get started 
Other Kur questions are found in [my stackoverflow](http://stackoverflow.com/users/4333609/daniel?tab=questions) 

How to use the cutting edge kur?
1. download kur from github, kur-master.zip
2. open it and go inside kur-master
3. `source activate your-env` and run `pip install -e .`
Now, you have the latest kur to use 

How to run an example?
1. cd kur/example
2. kur -vv data|build mnist.yml

How to create checksum for a dataset file?
1. `cksum filename/file_path` [check out](http://www.computerhope.com/unix/ucksum.htm)
2. or `sha256sum FILE` for linux
3. `shasum -a256 FILE` for macOS

## Progress
### Explore `kur -vv data -n 1 mnist.yml` 
- done 

### Explore `kur -vv build mnist.yml` 
- done 

### Explore `kur -vv train mnist.yml`
- [commit1](https://github.com/EmbraceLife/LIE/commit/a0b95951cab0dc98f5653f612589c8c2c9791e59): walk through logic of `kur train mnist.yml`
