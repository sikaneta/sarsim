#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 229376 237568 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 237568 245760 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 245760 246471 --rblock-size 400 &
wait
