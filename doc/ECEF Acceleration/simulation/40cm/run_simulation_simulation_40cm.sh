#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 0 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 1 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 2 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 3 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 4 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 5 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 6 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 7 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 8 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 9 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 10 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 11 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 12 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 13 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 14 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 15 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 16 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 17 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 18 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 19 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 20 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 21 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 22 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 23 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs 24 --target-rangeidx 400 --rblock-size 400
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --ridx 0 400 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --ridx 400 800 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --ridx 800 1024 --xblock-size 16384 &
wait
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 0 16384 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 16384 32768 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 32768 49152 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 49152 65536 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 65536 81920 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 81920 98304 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 98304 114688 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 114688 131072 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 131072 147456 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 147456 163840 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 163840 180224 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 180224 196608 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 196608 212992 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 212992 229376 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 229376 245760 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 245760 246471 --rblock-size 400 &
wait
