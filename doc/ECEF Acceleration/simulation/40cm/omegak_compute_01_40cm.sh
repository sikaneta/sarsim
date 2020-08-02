#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 57344 65536 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 65536 73728 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 73728 81920 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 81920 90112 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 90112 98304 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 98304 106496 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/40cm.xml --xidx 106496 114688 --rblock-size 400 &
wait
