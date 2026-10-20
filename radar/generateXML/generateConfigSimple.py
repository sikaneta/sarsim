#!/usr/bin/env python2.7

""" This python program should generate the config files for burst mode multi or single channel data
    The program generates an xml file to be processed in MATLAB """

import numpy as np
from copy import deepcopy
import lxml.etree as etree
import argparse
from generateXML.base_XML import modes
import json
from orbit.orientation import orbit
from orbit.geometry import perspective
from scipy.constants import c
import os

#%% Do the argument parsing stuff here
def get_parser():
    description = """ Generate an XML file for multi-channel signal
                      processing.
                      
                      This program sets up also for a burst mode."""
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument("--sim-folder",
                    help="Folder in which to place simulated data and files",
                    default= r"C:\Users\ishuwa.sikaneta\s1ng")
    
    parser.add_argument("--config-xml", 
                        required=True,
                        help="The name of the xml file to write")
    
    parser.add_argument("--instrument",
                        type=str,
                        default = "base_SentinelNG_2026")
    
    parser.add_argument("--beams",
                        nargs = "+",
                        default = [0],
                        type=int,
                        help="The list of beams to include")
    
    parser.add_argument("--channel-json",
                        default = r"C:\Users\Ishuwa.Sikaneta\OneDrive - ESA\Documents\ESTEC\Sentinel-NG\Tobias\TxTaperAz.json",
                        type=str,
                        help="Name of json file containing channel configurations")
    
    parser.add_argument("--keep-pickle",
                        help="""Keep any existing pickle file associated with the
                        computed radar object""",
                        action="store_true",
                        default=False)
    
    parser.add_argument("--bands",
                    help="The bands to process ...,-1,0,1,...",
                    type=int,
                    nargs="+",
                    default=None)
    
    parser.add_argument("--recompute-rsys",
                        help="Recompute the signal processing object",
                        action="store_true",
                        default=False)
    
    parser.add_argument("--recompute-Hfilter",
                        help="Recompute the multi channel filter",
                        action="store_true",
                        default=False)
    
    parser.add_argument("--ref-rangeidx",
                    help="""The reference range used for processing
                            If not specified, will use the center range""",
                    type = int,
                    default = None)
        
    parser.add_argument("--file-data-domain",
                    help="""The domain of the data in the associated file. 
                    Frequency or space. Uppercase=frequency, 
                    lowercase=time/space""",
                    choices=["rx", "rX", "RX", "Rx"],
                    default="rX")
    
    parser.add_argument("--channel-idxs",
                        help=""" The indexes of the channels to create. This 
                        allows the program to be run simulataneously on different 
                        CPUs. Each channel is of the form cXbY for channel X, 
                        beam Y""",
                        nargs = "*",
                        type=int,
                        default = None)
    
    parser.add_argument("--target-rangeidx",
                        help="The range index of the target to simulate",
                        type=int,
                        default=400)
    
    parser.add_argument("--rblock-size",
                        help="""The output files for each beam/channel will be
                                written to files with the range dimension of
                                size range block. 
                                i.e. with 
                                - a range block size of --range-block=512 
                                - a data file with name (...rX_c0b0.npy)
                                - a data size of 1200X16384 in range by
                                  azimuth respectively (rX) 
                                then will write several files of names 
                                ...r0X0_c0b0.npy    (has 512 rows)
                                ...r512X0_c0b0.npy  (has 512 rows)
                                ...r1024X0_c0b0.npy (has 1200-1024 rows)
                                If not specified, will set block to the total
                                number of range samples.""",
                        type=int,
                        default = None)
    
    parser.add_argument("--mchan-processed-file",
                        help="The name of the multi-channel processed output file",
                        default = None)
    
    parser.add_argument("--wk-processed-file",
                        help="The name of the w-k processed output file",
                        default = None)
    
    parser.add_argument("--vanilla-stolt",
                        help="Apply vanilla Stolt interpolation a3=a4=0",
                        action="store_true")
    
    parser.add_argument("--ridx",
                        help="""The range indeces to process. The process is
                                independent of the range variable so we can
                                easily split it into chunks""",
                        type=int,
                        nargs="+",
                        default=[0,-1])
    
    parser.add_argument("--xidx",
                        help="""The azimuth indeces to process. The process can
                                proceed with the azimuth direction split into 
                                tiles and processed independently. For a given
                                range of azimuth samples, all range samples need
                                to be loaded into memory""",
                        type=int,
                        nargs="+",
                        default=[0,None])
    
    parser.add_argument("--no-phase-correct",
                        help="""Do not Correct for difference between diffgeo and real""",
                        action="store_true")
    
    parser.add_argument("--p",
                        help="The value of p to use in the MCHAN filter",
                        type=float,
                        default = 0.9)
    
    parser.add_argument("--xblock-size",
                        help="Size of the output data block in the X direction",
                        type=int,
                        default=None)

    return parser

def createXMLStructure(folder): 
    for fd in ["simulation_data", 
               "simulation_plots",
               os.path.join("simulation_data", "raw"),
               os.path.join("simulation_data", "mchan_processed"),
               os.path.join("simulation_data", "wk_processed")]:
        tfd = os.path.join(folder, fd)
        if not os.path.exists(tfd):
            try:
                os.makedirs(tfd)
            except IOError as ie:
                print(ie)
                return False
    return True

def arr2str(el, pol='H'):
    p2d = (1.0/el['carrier'])/360.0
    mag = " ".join(["%0.4f" % x for x in el['amplitude']])
    phs = " ".join(["%0.4f" % x for x in el['phase']])
    dly = " ".join(["%0.6e" % (p2d*x) for x in el['phase']])
    return pol, mag, phs, dly

def genModeXML(vv):
    print("Wiper burst configuration generator")
    with open(vv.channel_json, 'r') as f:
        cfg = json.loads(f.read())
    
    brf = 1.0/sum([x['burst_len'] for x in cfg['tx']])
    
    TX = [x for k,x in enumerate(cfg['tx']) if k in vv.beams]
    o_j = cfg['orbit']
    orbit_tup = (o_j['inclination'],
                 o_j['e'],
                 o_j['a'],
                 o_j['arg_perigee'],
                 o_j['true_anomaly'])
        
    p = etree.XMLParser(remove_blank_text=True)
    base_XML = modes[vv.instrument] % orbit_tup

    # Generate the base XML object
    pq = etree.XMLParser(remove_blank_text=True)
    xml = etree.XML(base_XML,parser=pq)
    # azimuthPosNode = xml.findall('.//azimuthPositions')
    # xpos = [float(pos) for pos in azimuthPosNode[0].text.split()]
    
    myOrbit = orbit()
    myOrbit.setFromDict(o_j)
    
    sv, r, v = myOrbit.computeSV(o_j['true_anomaly'])
    
    persp = perspective(np.array(sv), np.datetime64(o_j['epoch']))
    off_nadir = np.radians(np.arange(5,55,0.01))
    geom = persp.computeGeometry(off_nadir)
    
    rngs = [[np.interp(x, geom['off_nadir'], geom['range']) 
             for x in y['look']] for y in TX]
    
    channel_xml = """
        <radarConfiguration channel="%d-%d">
            <transmitConfiguration>
                <polarization>%s</polarization>
                <magnitude unit="natural">%s</magnitude>
                <phase unit="degrees">%s</phase>
                <truedelay unit="s">%s</truedelay>
            </transmitConfiguration>
            <receiveConfiguration>
                <polarization>%s</polarization>
                <magnitude unit="natural">%s</magnitude>
                <phase unit="degrees">%s</phase>
                <truedelay unit="s">%s</truedelay>
            </receiveConfiguration>
        </radarConfiguration>
    """
    
    mode_template = """
    <mode>
        %s
    </mode>
    """
    # mytx = arr2str(channels['tx'][0])
    chans = [[channel_xml % ((iy,ix) + arr2str(y) + arr2str(x)) 
              for ix,x in enumerate(cfg['rx'])]
             for iy,y in enumerate(TX)]
        
    beams = [mode_template % "\n".join(y) for y in chans]
    

    modeXML = [etree.XML(beam,parser=p) for beam in beams]
    

    # Generate the reference time from which to compute the offsets. e.t.c.
    reference_TIME = np.datetime64("2026-09-21T15:45:34.123587782")
    dT = [np.timedelta64(int(x['burst_len']*1e9), 'ns') 
          for x in TX]
    N_az = [int(x['prf']/brf) for x in TX]
    B_az = [int(x['burst_len']*x['prf']) - 5 for x in TX]
    c_TIMES = [reference_TIME]
    for _dT in dT:
        c_TIMES.append(c_TIMES[-1] + _dT)
    prfs = [x['prf'] for x in TX]
    
    for prf, r, n_az, b_az, st, beam, tx, rx in zip(prfs, 
                                                    rngs, 
                                                    N_az,
                                                    B_az,
                                                    c_TIMES, 
                                                    modeXML,
                                                    TX,
                                                    cfg['rx']):
        print(st)
        for config in beam:
            # Add the signal elements
            signalData = etree.SubElement(xml,"signalData")
            timeNode = etree.SubElement(signalData,"azimuthStartTime")
            timeNode.set("iso", str(st))

            numRangeSamples = etree.SubElement(signalData,"numRangeSamples")
            numRangeSamples.set("unit","Pixels")
            numRangeSamples.text = "%d" % (int((r[1]-r[0])*2/c*rx['fs']))

            numAzimuthSamples = etree.SubElement(signalData,"numAzimuthSamples")
            numAzimuthSamples.set("unit","Pixels")
            numAzimuthSamples.text = "%d" % n_az
            
            numAzimuthSamples = etree.SubElement(signalData,"numBurstSamples")
            numAzimuthSamples.set("unit","Pixels")
            numAzimuthSamples.text = "%d" % b_az

            nearRangeTime = etree.SubElement(signalData,"nearRangeTime")
            nearRangeTime.set("unit","s")
            nearRangeTime.text = "%0.9e" % (r[0]*2/c)

            rangeSampleSpacing = etree.SubElement(signalData,"rangeSampleSpacing")
            rangeSampleSpacing.set("unit","s")
            fs = max(rx['fs'], 1.2*tx['waveform']['bandwidth'])
            rangeSampleSpacing.text = "%0.9e" % (1.0/fs)

            pulseRepetitionFrequency = etree.SubElement(signalData,"pulseRepetitionFrequency")
            pulseRepetitionFrequency.set("unit","Hz")
            pulseRepetitionFrequency.text = "%0.4f" % prf

            pulseBandwidth = etree.SubElement(signalData,"pulseBandwidth")
            pulseBandwidth.set("unit","Hz")
            pulseBandwidth.text = "%0.4e" % tx['waveform']['bandwidth']

            pulseDuration = etree.SubElement(signalData,"pulseDuration")
            pulseDuration.set("unit","s")
            pulseDuration.text = "%0.9e" % tx['waveform']['length']

            signalData.append(deepcopy(config))
            thisfilename = etree.SubElement(signalData,"filename")
            thisfilename_text = "%s_%s_c%s.npy" % (os.path.split(vv.config_xml)[-1].split(".")[0], 
                                                      vv.file_data_domain, 
                                                      config.get("channel"))
            thisfilename.text = os.path.sep.join([os.path.split(vv.config_xml)[0],
                                                  "simulation_data",
                                                  "raw",
                                                  thisfilename_text])

    return xml


#%%
# def_args = "--config-xml s1g.xml --beams 1".split()
# vv = get_parser().parse_args(def_args)

# if os.path.abspath(vv.config_xml) != vv.config_xml:
#     vv.config_xml = os.path.join(os.path.abspath(vv.sim_folder), vv.config_xml)

# if vv.ridx[-1] == -1:
#     vv.ridx[-1] = None
    
# #%%
# myxml = genModeXML(vv)

# #%%
# with open(vv.config_xml,'w') as outfile:
#     outfile.write(etree.tostring(myxml, pretty_print=True).decode())

# #%% Create the target directory if it does not exist
# if vv.config_xml is not None:
#     createXMLStructure(os.path.split(vv.config_xml)[0])

# #%% Delete any existing pickle radar file
# if not vv.keep_pickle:
#     pkl_file = ".".join(vv.config_xml.split(".")[0:-1]) + "_radar.pickle"
#     if os.path.exists(pkl_file):
#         os.remove(pkl_file)
