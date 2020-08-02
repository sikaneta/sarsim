#!/bin/bash
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
