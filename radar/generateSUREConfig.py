#!/usr/bin/env python2.7

#%%
""" This python program should generate the config files for burst mode multi or single channel data
    The program generates an xml file to be processed in MATLAB """

import os
import argparse
import generateXML.sure as s_config

#%% Use the argparse library
if "HOMEPATH" in os.environ.keys():
    default_simulation_file = os.path.sep.join([os.environ["HOMEPATH"],"simulation","1m","1m_simulation.xml"])
elif "HOME" in os.environ.keys():
    default_simulation_file = os.path.sep.join([os.environ["HOME"],"simulation","1m","1m_simulation.xml"])
    
parser = argparse.ArgumentParser(description="Generate multi-channel XML file")
parser.add_argument("--sim-folder",
                    help="Folder in which to place simulated data and files",
                    default= "./")
parser.add_argument("--config-xml", 
                    help="The config XML file", 
                    required=True)
parser.add_argument("--az-resolution",
                    help="The desired azimuth resolution (m)",
                    type = float,
                    default = 1.0)
parser.add_argument("--element-power",
                    help="The power transmitted by each array element (dB)",
                    type = float,
                    default = 10.0)
parser.add_argument("--rn-resolution",
                    help="The desired range resolution (m)",
                    type = float,
                    default = 1.0)
parser.add_argument("--swath-width",
                    help="The desired swath width (m)",
                    type = float,
                    default = 10000.0)
parser.add_argument("--range-samples",
                    help=""""Instead of simulating the entire swath, 
                             use fewer samples. Subset of the swath (int)""",
                    type = int,
                    default = 512)
parser.add_argument("--rn-oversample",
                    help="The desired range oversample factor",
                    type = float,
                    default = 1.2)
parser.add_argument("--az-oversample",
                    help="The desired azimuth oversample factor",
                    type = float,
                    default = 1.2)
parser.add_argument("--pulse-duration",
                    help="Desired pulse duration in us",
                    type = float,
                    default = 20.5)
parser.add_argument("--max-antenna-length",
                    help="The maximum total antenna length (m)",
                    type = float,
                    default = 25.0)
parser.add_argument("--file-data-domain",
                    help="""The domain of the data in the associated file. 
                    Frequency or space. Uppercase=frequency, 
                    lowercase=time/space""",
                    choices=["rx", "rX", "RX", "Rx"],
                    default="rX")
parser.add_argument("--keep-pickle",
                    help="""Keep any existing pickle file associated with the
                    computed radar object""",
                    action="store_true",
                    default=False)

vv = parser.parse_args()

#%% Create the XML object
if os.path.abspath(vv.config_xml) != vv.config_xml:
    vv.config_xml = os.path.join(os.path.abspath(vv.sim_folder), vv.config_xml)
myxml = s_config.generateXML(vv)

#%% Create the target directory if it does not exist
if vv.config_xml is not None:
    s_config.createXMLStructure(os.path.split(vv.config_xml)[0])
    
#%% Write the object to file
dummy = s_config.writeToXML(vv.config_xml, myxml)

#%% Delete any existing pickle radar file
if not vv.keep_pickle:
    pkl_file = ".".join(vv.config_xml.split(".")[0:-1]) + "_radar.pickle"
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
    
