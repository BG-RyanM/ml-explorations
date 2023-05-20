# ml-explorations
Machine Learning Explorations, 2023

## Basic Info

This repository contains Jupyter notebooks that accompany the video course for "Deep Learning with PyTorch", by Anand Saha. This course is from April 2018 and is somewhat out of date, in that the PyTorch API has deprecated or removed a few 2018-era functions.

Contents:
Folder    | Description
----------|----------------
`Deep-learning-with-PyTorch-video/` | Notebooks pertaining to the course. I've modified them somewhat from the original, fixing outdated code that no longer works and adding more comments.
`first-nn/` | An exercise I found online. It's [here](https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/).
`notes/` | [Notes](./notes) on the course lessons.

## Setup (As Needed for Deeping Learning With PyTorch)

### Basic Setup, Covering Part 1 of the Course
Things have changed a bit since the video was created, but it's worth following along with it anyway.

_(I did this all on host, not container)_

1. Download latest conda distribution.
2. Install via: `$ sh Anaconda3-2023.03-Linux-x86_64.sh`
3. Close terminal, reopen
4. `$ conda create -n ml-pytorch python=3.9 matplotlib numpy`
5. Wait while it slowly installs.
6. `$ conda activate ml-pytorch`
7. Turn on VPN if at home.
8. I assume you probably have a GPU that's compatible with the CUDA stuff, but feel free to look. Nvidia's site has instructions on finding out.
9. `~/anaconda3/envs/ml-pytorch/bin/pip install torch torchvision torchaudio`
10. Wait for slooooow install.
11. Do what he says from 5:40 to 6:00 in video 1-3, to verify that pytorch is installed.
12. `$ conda install jupyter`
13. `$ git clone https://github.com/PacktPublishing/Deep-learning-with-PyTorch-video/`

Note: my repository is missing the giant file `Deep-learning-with-PyTorch-video/data/cifar-10-python.tar.gz`. It was too big to include.

### CUDA: Video 1-4
This part's a bit irritating.

(Open terminal on host, but not inside new conda environment)

1. `$ wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run`
2. Long-ass download time.
3. `$ sudo apt-get install gcc build-essential`
4. `$ sudo sh cuda_12.1.1_530.30.02_linux.run`
5. Wait a while
6. "Continue" to leave the existing driver.
7. Do what this guy says: https://tabbas97.medium.com/get-cuda-the-right-way-c68d533bed3e. Don't forget the path stuff.
 -   PATH includes /usr/local/cuda-12.1/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.1/lib64, or, add /usr/local/cuda-12.1/lib64 to /etc/ld.so.conf and run ldconfig as root
8. Do what our instructor says from 2:45 to 3:15 in video 1-4

### Video 1-5
1. `$ cd Deep-learning-with-PyTorch-video`
2. `$ jupyter notebook`
3. For me, it got mad because it was missing character set libraries, fixed with `$ pip install chardet`
4. Once you have all the libraries Jupyter Notebook needs to be happy, open 1.5.tensors.ipynb. You should see the same notebook as shown in the video.