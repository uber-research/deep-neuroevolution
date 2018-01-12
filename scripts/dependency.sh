#!/bin/bash

# from ami-d8bdebb8

set -x

sudo apt-get update
sudo apt-get install -y build-essential cmake git wget htop

# Build and install a new version of redis
# https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-redis-on-ubuntu-16-04
wget --quiet http://download.redis.io/releases/redis-3.2.7.tar.gz -O redis-3.2.7.tar.gz
tar -xvzf redis-3.2.7.tar.gz
cd redis-3.2.7
make
sudo make install
sudo mkdir /etc/redis
sudo cp redis.conf /etc/redis
cd ..
rm -rf redis-3.2.7 redis-3.2.7.tar.gz

# Set up redis working directory
sudo sed -ie 's/dir \.\//dir \/var\/lib\/redis/' /etc/redis/redis.conf
sudo mkdir /var/lib/redis
sudo chown ubuntu:ubuntu /var/lib/redis

# rely on firewall for security
sudo sed -ie "s/bind 127.0.0.1//" /etc/redis/redis.conf
sudo sed -ie "s/protected-mode yes/protected-mode no/" /etc/redis/redis.conf

# System settings for redis
echo "vm.overcommit_memory=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl vm.overcommit_memory=1
sudo apt-get install -y hugepages
echo "sudo hugeadm --thp-never" | sudo tee /etc/profile.d/disable_thp.sh > /dev/null
. /etc/profile.d/disable_thp.sh

# Start redis with systemctl
# sudo sed -ie "s/supervised no/supervised systemd/" /etc/redis/redis.conf
# ^ doesn't seem to matter; if it's enabled, the logs show "systemd supervision requested, but NOTIFY_SOCKET not found"
echo "
[Unit]
Description=Redis In-Memory Data Store
After=network.target

[Service]
User=ubuntu
Group=ubuntu
ExecStart=/usr/local/bin/redis-server /etc/redis/redis.conf
ExecStop=/usr/local/bin/redis-cli shutdown
Restart=always

[Install]
WantedBy=multi-user.target
" | sudo tee /etc/systemd/system/redis.service > /dev/null
sudo systemctl start redis

# anaconda
sudo echo 'export PATH=/opt/conda/bin:$PATH' | sudo tee /etc/profile.d/conda.sh > /dev/null
sudo wget --quiet https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh
sudo /bin/bash ~/anaconda.sh -b -p /opt/conda
sudo rm -f ~/anaconda.sh
. /etc/profile.d/conda.sh
sudo /opt/conda/bin/conda update -y --all

# additional python dependencies
sudo /opt/conda/bin/conda install -y numpy scipy opencv

# Mujoco
sudo mkdir -p /opt/mujoco

#######################################################
# WRITE CODE HERE TO PLACE MUJOCO 1.31 in /opt/mujoco #
# The key file should be in /opt/mujoco/mjkey.txt     #
# Mujoco should be installed in /opt/mujoco/mjpro131  #
#######################################################

sudo echo 'export MUJOCO_PY_MJKEY_PATH=/opt/mujoco/mjkey.txt' | sudo tee /etc/profile.d/mujoco.sh > /dev/null
sudo echo 'export MUJOCO_PY_MJPRO_PATH=/opt/mujoco/mjpro131' | sudo tee -a /etc/profile.d/mujoco.sh > /dev/null
. /etc/profile.d/mujoco.sh

# ALE
sudo /opt/conda/bin/conda install -y libgcc # ALE needs this for some reason
sudo apt-get install -y libsdl1.2-dev
git clone https://github.com/mgbellemare/Arcade-Learning-Environment
cd Arcade-Learning-Environment
mkdir build
cd build
cmake .. -DUSE_SDL=on
make
cd ..
sudo /opt/conda/bin/pip install .
cd ..
rm -rf Arcade-Learning-Environment

# Tensorflow 0.11.0
sudo /opt/conda/bin/conda install -c conda-forge -y tensorflow=0.11.0

# Gym
sudo apt-get install -y zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev libboost-all-dev libsdl2-dev swig freeglut3 libgl1 libglu1
sudo /opt/conda/bin/conda install -y pyopengl
sudo /opt/conda/bin/pip install \
    'gym[atari,classic_control,mujoco]' \
    PyYAML \
    six==1.10.0 \
    awscli

# pip redis
sudo /opt/conda/bin/pip install redis

# cleanup
sudo /opt/conda/bin/conda clean -y --all
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
# cleanup pip cache?

set +x
