@echo off
call conda create --name dtc_training python=3.12.7 -y
call conda activate dtc_training

call conda install conda-forge::opencv==4.10.0 -y
call conda install pandas==2.2.2 -y
call conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
call conda install tqdm -y