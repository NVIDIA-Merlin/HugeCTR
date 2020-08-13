#!/bin/bash
apt purge -y --auto-remove cmake
python3 -m pip install -Iv -q cmake==3.14.3
apt-get install -y zlib1g-dev libboost-all-dev
apt update
apt install -y -V ca-certificates lsb-release wget
if [ $(lsb_release --codename --short) = "stretch" ]; then
  sudo tee /etc/apt/sources.list.d/backports.list <<APT_LINE
deb http://deb.debian.org/debian $(lsb_release --codename --short)-backports main
APT_LINE
fi
wget https://apache.bintray.com/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
apt install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
apt update
apt install -y libarrow-dev=0.17.1-1 libarrow-cuda-dev=0.17.1-1


