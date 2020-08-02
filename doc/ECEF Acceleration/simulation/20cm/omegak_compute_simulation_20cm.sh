#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 0 16384 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 16384 32768 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 32768 49152 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 49152 65536 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 65536 81920 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 81920 98304 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 98304 114688 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 114688 131072 --rblock-size 256 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 131072 147456 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 147456 163840 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 163840 180224 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 180224 196608 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 196608 212992 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 212992 229376 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 229376 245760 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 245760 262144 --rblock-size 256 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 262144 278528 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 278528 294912 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 294912 311296 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 311296 327680 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 327680 344064 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 344064 360448 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 360448 376832 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 376832 393216 --rblock-size 256 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 393216 409600 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 409600 425984 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 425984 442368 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 442368 458752 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 458752 475136 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 475136 491520 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 491520 507904 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 507904 524288 --rblock-size 256 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 524288 540672 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 540672 557056 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 557056 573440 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 573440 589824 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 589824 606208 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 606208 622592 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 622592 638976 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 638976 655360 --rblock-size 256 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 655360 671744 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 671744 688128 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 688128 704512 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 704512 720896 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 720896 737280 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 737280 753664 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 753664 770048 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 770048 786432 --rblock-size 256 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 786432 802816 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 802816 819200 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 819200 835584 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 835584 851968 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 851968 868352 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 868352 884736 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 884736 886584 --rblock-size 256 &
wait
