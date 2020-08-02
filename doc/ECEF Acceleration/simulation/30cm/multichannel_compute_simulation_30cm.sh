#!/bin/bash
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 0 400 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 400 800 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 800 1200 --xblock-size 16384 &
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 1200 1600 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 1600 2000 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/30cm/simulation_30cm.xml --ridx 2000 2048 --xblock-size 16384 &
wait
