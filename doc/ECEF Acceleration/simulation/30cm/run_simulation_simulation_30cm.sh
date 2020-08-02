#!/bin/bash
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 0 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 1 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 2 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 3 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 4 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 5 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 6 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 7 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 8 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 9 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 10 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 11 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 12 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 13 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 14 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 15 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 16 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 17 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 18 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 19 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 20 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 21 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 22 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 23 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 24 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 25 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 26 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 27 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 28 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 29 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 30 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 31 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 32 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 33 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 34 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --channel-idxs 35 --target-rangeidx 400 --rblock-size 400
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 0 400 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 400 800 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 800 1200 --xblock-size 16384 &
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 1200 1600 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 1600 2000 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 2000 2048 --xblock-size 16384 &
wait
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 0 16384 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 16384 32768 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 32768 49152 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 49152 65536 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 65536 81920 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 81920 98304 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 98304 114688 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 114688 131072 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 131072 147456 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 147456 163840 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 163840 180224 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 180224 196608 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 196608 212992 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 212992 229376 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 229376 245760 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 245760 262144 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 262144 278528 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 278528 294912 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 294912 311296 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 311296 327680 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 327680 344064 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 344064 360448 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 360448 376832 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 376832 393216 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 393216 409600 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --xidx 409600 420660 --rblock-size 400 &
wait
