#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:33:36 2020

@author: ishuwa
"""
import numpy as np
import re
from glob import glob
import os

#%% Write simulated files to disk in chunks
def writeSimFiles(filename, data, rblock=None, xblock=None):
    # Split the file name into its parts
    fparts = filename.split("_")
    
    #domain = fparts[-2]
    chanID = fparts[-1]
    
    dims = data.shape
    
    # Split the domain string to se which is r which is x
    dom_split = re.findall("[rRxX][0-9]*", fparts[-2])
    
    # Calculate offsets
    d_offsets = [int(d[1:] or '0') for d in dom_split]
    
    fnames = []
    
    rdim = 0 if dom_split[0][0].lower() == 'r' else 1
    xdim = 1 - rdim
    
    rsig = dom_split[rdim][0]
    xsig = dom_split[xdim][0]
    
    rblock = rblock or dims[rdim]
    xblock = xblock or dims[xdim]
    
    rngs = np.arange(0, dims[rdim], rblock)
    xngs = np.arange(0, dims[xdim], xblock)
    
    fnames = []
    fcomp = ["",""]
    dtidx = np.array([[0,0],[0,0]])
    for rstart_idx in rngs:
        rend_idx = np.min([rstart_idx + rblock, dims[rdim]])
        for xstart_idx in xngs:
            xend_idx = np.min([xstart_idx + xblock, dims[xdim]])
            fcomp[rdim] = "%s%d" % (rsig, rstart_idx + d_offsets[rdim])
            fcomp[xdim] = "%s%d" % (xsig, xstart_idx + d_offsets[xdim])
            dtidx[rdim] = [rstart_idx, rend_idx]
            dtidx[xdim] = [xstart_idx, xend_idx]
            fsig = "%s%s" % (fcomp[0], fcomp[1])
            out_file = "_".join(fparts[0:-2] + [fsig, chanID])
            np.save(out_file, data[dtidx[0,0]:dtidx[0,1],
                                   dtidx[1,0]:dtidx[1,1]])
            fnames.append(out_file)
    
    return fnames

#%% Load the sim files written above. 
"""
If ridx is None or if xidx is None, then assume that the user wishes
to load all data that are available with that signature.
"""
def loadSimFiles(filename, ridx=[0, None], xidx=[0, None]):
    """Check if we have two entries for the indeces"""
    if len(ridx) == 1:
        ridx = [ridx[0], None]
    if len(xidx) == 1:
        xidx = [xidx[0], None]
        
    """Split the file name into its parts"""
    fparts = filename.split("_")
    
    chanID = fparts[-1]
    
    """Split the domain string to se which is r which is x"""
    dom_split = re.findall("[rRxX][0-9]*",fparts[-2])
    
    
    rdim = 0 if dom_split[0][0].lower() == 'r' else 1
    xdim = 1 - rdim
    
    rsig = dom_split[rdim][0]
    xsig = dom_split[xdim][0]
    
    fcomp = ["",""]
    fcomp[rdim] = "%s" % rsig
    fcomp[xdim] = "%s" % xsig
    glob_pattern = "_".join(fparts[0:-2] + 
                            ["%s*%s*" % (fcomp[0], fcomp[1]), 
                             chanID])
    flist = glob(glob_pattern)
    indexes = sorted([[int(d[1:] or '0') for d in 
                re.findall("[rRxX][0-9]*", f.split("_")[-2])] 
                for f in flist])
    indexes_flat = [np.array(sorted(list(set([x[k] for x in indexes])))) 
                    for k in range(2)]
    
    """Find what we want"""
    if len(indexes_flat[rdim]) == 1:
        rstart = 0
        rend = None
    else:
        rstart = np.argwhere(indexes_flat[rdim] <= ridx[0])[-1][0]
        if ridx[1] is None:
            rend = None
        else:
            rend = np.argwhere(indexes_flat[rdim] < ridx[1])[-1][0] + 1
    
    if len(indexes_flat[xdim]) == 1:
        xstart = 0
        xend = None
    else:
        xstart = np.argwhere(indexes_flat[xdim] <= xidx[0])[-1][0]
        if xidx[1] is None:
            xend = None
        else:
            xend = np.argwhere(indexes_flat[xdim] < xidx[1])[-1][0] + 1
        
    indexes_flat[xdim] = indexes_flat[xdim][xstart:xend]
    indexes_flat[rdim] = indexes_flat[rdim][rstart:rend]
    
    def getFileName(row, col):
        if xdim==0:
            fsig = "%s%d%s%d" % (xsig,row,rsig,col)
        else:
            fsig = "%s%d%s%d" % (rsig,row,xsig,col)
        return "_".join(fparts[0:-2] + [fsig, chanID])
        
    """Compute the file names"""
    rdata = np.vstack([np.hstack([np.load(getFileName(rw,cl)) 
                       for cl in indexes_flat[1]])
                       for rw in indexes_flat[0]])
    
    xlims = [[None, None], [None, None]]
    x_off = indexes_flat[xdim][0]
    r_off = indexes_flat[rdim][0]
    
    xidx = [x - x_off if x is not None else None for x in xidx]
    ridx = [r - r_off if r is not None else None for r in ridx]
    xlims[xdim] = xidx
    xlims[rdim] = ridx
    
    """Make sure there is data at the indeces"""
    drows, dcols = rdata.shape
    xlims[0][0] = np.max([0, xlims[0][0]])
    xlims[1][0] = np.max([0, xlims[1][0]])
    xlims[0][1] = np.min([drows, xlims[0][1]]) if xlims[0][1] is not None else None
    xlims[1][1] = np.min([dcols, xlims[1][1]]) if xlims[1][1] is not None else None
    
    return rdata[xlims[0][0]:xlims[0][1], xlims[1][0]:xlims[1][1]]


#%% Define a function to load the data with given file naming
# convention. This loads the raw data generated as measurements from
# the sensor. These data need to be multi-channel processed
def loadNumpy_raw_dataMem(radar, 
                          ridx=[0, None], 
                          xidx=[0, None], 
                          target_domain = "rX"):
    # Process the file names
    fls = [r['filename'] for r in radar]
    domain = [fl.split("_")[-2] for fl in fls]
    
    # Define the data loading function dictionary
    fn_dict = {'rxrX': lambda x: np.fft.fft(x, axis=1), 
               'RxrX': lambda x: np.fft.ifft(np.fft.fft(x, axis=1), axis=0),
               'rXrX': lambda x: x,
               'RXrX': lambda x: np.fft.ifft(x, axis=0),
               'rxrx': lambda x: x, 
               'Rxrx': lambda x: np.fft.ifft(x, axis=0),
               'rXrx': lambda x: np.fft.ifft(x, axis=1),
               'RXrx': lambda x: np.fft.ifft2(x),
               'rxRX': lambda x: np.fft.fft2(x), 
               'RxRX': lambda x: np.fft.fft(x, axis=1),
               'rXRX': lambda x: np.fft.fft(x, axis=0),
               'RXRX': lambda x: x,
               'rxRx': lambda x: np.fft.fft(x, axis=0), 
               'RxRx': lambda x: x,
               'rXRx': lambda x: np.fft.ifft(np.fft.fft(x, axis=0), axis=1),
               'RXRx': lambda x: np.fft.ifft(x, axis=1)}
    
    # Load the data
    #data = np.stack([np.fft.fft(np.load(fl), axis=1) for fl in fls], axis=0)
    return np.stack([fn_dict[dm + target_domain](loadSimFiles(fl, ridx, xidx)) 
                     for dm,fl in zip(domain,fls)], axis=0)
       
#%% Define a function to load the multichannel processed data. These
# data will need to be processed by a standard "stripmap" SAR processor
# like the Omega-K processor
def loadNumpy_mcp_data(data_file, 
                       ridx=[0, None], 
                       xidx=[0, None], 
                       target_domain = "rX"):
    fn_dict = {'rx': lambda x: np.fft.fft2(x, axis=1), 
               'Rx': lambda x: np.fft.fft(x, axis=1),
               'rX': lambda x: np.fft.fft(x, axis=0),
               'RX': lambda x: x,
               'xr': lambda x: np.fft.fft2(x).T, 
               'xR': lambda x: np.fft.fft(x, axis=0).T,
               'Xr': lambda x: np.fft.fft(x, axis=1).T,
               'XR': lambda x: x.T}
    domain = data_file.split("_")[-2]
    return fn_dict[domain](loadSimFiles(data_file, ridx, xidx))    

#%% Function to compute filenames
def fileStruct(filename, subfolder, domain, extension):
    path, tail = os.path.split(filename)
    path = os.path.join(os.path.split(path)[0], subfolder)
    tail = "_".join(tail.split("_")[0:-2] + [domain, extension])
    return os.path.join(path, tail)
    
    