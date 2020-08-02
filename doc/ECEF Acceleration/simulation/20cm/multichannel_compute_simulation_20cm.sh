#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 0 256 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 256 512 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 512 768 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 768 1024 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1024 1280 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1280 1536 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1536 1792 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1792 2048 --xblock-size 16384 &
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2048 2304 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2304 2560 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2560 2816 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2816 3072 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3072 3328 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3328 3584 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3584 3840 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3840 4096 --xblock-size 16384 &
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 4096 4352 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 4352 4608 --xblock-size 16384 &
wait
