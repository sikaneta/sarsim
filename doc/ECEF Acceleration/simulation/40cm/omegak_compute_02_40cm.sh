#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 114688 122880 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 122880 131072 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 131072 139264 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 139264 147456 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 147456 155648 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 155648 163840 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 163840 172032 --rblock-size 400 &
wait
