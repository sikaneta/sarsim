#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 172032 180224 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 180224 188416 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 188416 196608 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 196608 204800 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 204800 212992 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 212992 221184 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 221184 229376 --rblock-size 400 &
wait
