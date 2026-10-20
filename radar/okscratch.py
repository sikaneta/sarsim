# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:26:18 2026

@author: Ishuwa.Sikaneta
"""

from generateXML.generateConfigSimple import get_parser 
import configuration.configuration as cfg
import numpy as np
import os
import utils.fileio as fio
from scipy.constants import c
import matplotlib.pyplot as plt
from common.utils import FFT_freq
import math
import omegak.nbomegak as nbwk
from omegak.stolz_cuda import getInterpolationPoints as gIP
import matplotlib.pyplot as plt
from numba import cuda
from common.cxinterp import dirichlet, fdiri, inner_fsum
from tqdm import tqdm
from omegak.cudawk import wkProcessNumbaCuda

#%% Generate a new simulation
def_args = "--config-xml s1g.xml --beams 1 --recompute-rsys".split()
vv = get_parser().parse_args(def_args)

if os.path.abspath(vv.config_xml) != vv.config_xml:
    vv.config_xml = os.path.join(os.path.abspath(vv.sim_folder), vv.config_xml)

if vv.ridx[-1] == -1:
    vv.ridx[-1] = None

radar = cfg.loadConfiguration(vv.config_xml)

bands = vv.bands or np.arange(-int(len(radar)/2),int(len(radar)/2)+1)

print("""Computing the radar system signal processing object""")
if vv.recompute_rsys:
    r_sys = cfg.computeStoreRsys(radar, bands, vv.ref_rangeidx)
else:
    r_sys = cfg.loadRsys(radar) or cfg.computeStoreRsys(radar, 
                                                        bands, 
                                                        vv.ref_rangeidx)
if vv.mchan_processed_file is None:
    vv.mchan_processed_file = fio.fileStruct(radar[0]['filename'],
                                             "mchan_processed",
                                             "Xr",
                                             "mchanprocessed.npy")

if vv.vanilla_stolt:
    print("Applying vanilla Stolt interpolation")
    r_sys.C.a3 = 0.0
    r_sys.C.a4 = 0.0
    

procData = fio.loadSimFiles(vv.mchan_processed_file, xidx=vv.xidx)
if not vv.no_phase_correct:
    print("Removing differential geo phase error")
    phsCorr = np.exp(-1j*r_sys.ks_phase_correction[vv.xidx[0]:vv.xidx[-1]])
    (rows,cols) = procData.shape
    for row in range(rows):
        procData[row,:] *= phsCorr[row]
        
#%% Do the SAR processing for this chunk
print("Processing")
print(vv.xidx)
mycols = [x for x in [cols, vv.rblock_size] if x is not None]

os_factor=16
mem_cols = min(mycols)
mem_rows = 8192

wkSignal = wkProcessNumbaCuda(procData, 
                              r_sys, 
                              ks = r_sys.ks_full[vv.xidx[0]:vv.xidx[1]],
                              os_factor=16, 
                              mem_cols = mem_cols,
                              mem_rows = mem_rows)

#%%
r = np.linalg.norm(r_sys.C.R)
ks = r_sys.ks_full
cols = len(r_sys.kr_sorted)
rows = len(ks)

""" Allocate memory and calculate indeces for the pulse workspace """
# Yos_idx = np.round(FFT_freq(cols,cols,0)).astype('int')

dkr = r_sys.kr[1]-r_sys.kr[0]

row_ticks = list(range(0,rows,mem_rows)) + [rows]
row_spans = [(row_ticks[k], row_ticks[k+1]) 
             for k in range(len(row_ticks)-1)]

wk_processed = np.zeros((rows,mem_cols), dtype=complex)
yupkr = 4.0*np.pi/c*(FFT_freq(cols*os_factor, r_sys.fs, 0) + r_sys.f0)
yupidx = np.argsort(yupkr)
span = row_spans[0]
print("Processing rows (azimuth) %d to %d..." % span)
n_rows = span[1]-span[0]
YY = np.zeros((n_rows, cols), dtype=complex)

iP = np.zeros((n_rows, r_sys.kr_sorted.shape[0]), 
              dtype=float)

""" Compute the points at which to interpolate """
print("Computing interpolation points")
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(iP.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(iP.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
KS = ks[span[0]:span[1]]
KR = r_sys.kr_sorted
a2 = float(r_sys.C.a2)
a3 = float(r_sys.C.a3)
a4 = float(r_sys.C.a4)
max_iter = 10
error_tol = 1e-10
KS_d = cuda.to_device(KS)
KR_d = cuda.to_device(KR)
iP_d = cuda.to_device(iP)

gIP[blockspergrid, threadsperblock](KS_d, 
                                    KR_d,
                                    iP_d,
                                    float(r),
                                    a2,
                                    a3,
                                    a4,
                                    max_iter,
                                    error_tol)

iP = iP_d.copy_to_host()
""" Perform the interpolation """
Xnew = (iP - r_sys.kr_sorted[0])/dkr
XX = Xnew
Y = np.fft.fft(procData[span[0]:span[1], :], axis=1)

#%%  
MY,NY = Y.shape 
H = fdiri(NY, os_factor)
MH, NH = H.shape

#%%
print("Interpolating the signal...")
Y_d = cuda.to_device(Y)
XX_d = cuda.to_device(XX)
YY_d = cuda.to_device(YY)
H_d = cuda.to_device(H)

inner_fsum[blockspergrid, threadsperblock](Y_d, 
                                           YY_d,
                                           XX_d,
                                           H_d,
                                           32)


YY = YY_d.copy_to_host()

#%% Test to see if the filters as defined in fdiri are working
m = 260
n = 8000
nrng = np.arange(NY)
nrng1 = np.arange(os_factor*NY)/os_factor
y = procData[span[0]:span[1], :][m,:]

yos = np.fft.fft(y, os_factor*NY)
  
myH = np.array([np.exp(-1j*2*np.pi*np.arange(NY)/NY*k/os_factor) 
                 for k in range(os_factor)])
yall = np.fft.fft(y*myH, axis=1)
 
plt.figure()
rfn = np.real
plt.plot(nrng1, rfn(yos))
for row in range(os_factor):
    plt.plot(nrng+row/os_factor, rfn(yall[row,:]),'.')
plt.grid()

#%% Test to see if the convolution is working as it should
y_f = np.tile(np.fft.fft(y), (os_factor,1))
y_int = np.tile(np.zeros_like(y), (os_factor, 1))
for k in tqdm(range(NY)):
    for kk in range(NY):
        y_int[:,k] += H[:,kk]*y_f[:, k-kk]
   
plt.figure()
rfn = np.real
plt.plot(nrng1, rfn(yos))
for row in range(os_factor):
    plt.plot(nrng+row/os_factor, rfn(y_int[row,:]),'.')
plt.grid()

#%% Test truncated convolution
Mc = 32
y_f = np.tile(np.fft.fft(y), (os_factor,1))
y_int = np.tile(np.zeros_like(y), (os_factor, 1))
for k in tqdm(range(NY)):
    for kk in range(Mc):
        y_int[:,k] += H[:,kk]*y_f[:, k-kk]
    for kk in range(NY-Mc, NY):
        y_int[:,k] += H[:,kk]*y_f[:, k-kk]
   
plt.figure()
rfn = np.real
plt.plot(nrng1, rfn(yos))
for row in range(os_factor):
    plt.plot(nrng+row/os_factor, rfn(y_int[row,:]),'.')
plt.grid()

#%% Test for the value at a particular point
xx = 8567.0
X_up = np.array([nrng+row/os_factor for row in range(os_factor)])
col = int(xx)
r_idx = 0
for r_idx in range(os_factor):
    if X_up[r_idx,col] > xx:
        break
if r_idx == os_factor-1:
    l_idx = r_idx
else:
    l_idx = r_idx - 1

""" Check if X_up[l_idx, col] == xx """
if X_up[l_idx, col] == xx:
    print("we're done")
    
""" Run the convolution """
y_f = np.fft.fft(y)
Mc = 32
l_total = 0.0
r_total = 0.0
for kk in range(Mc):
    l_total += H[l_idx,kk]*y_f[col-kk]
for kk in range(NY-Mc, NY):
    l_total += H[l_idx,kk]*y_f[col-kk]
    
if l_idx == r_idx:
    r_total = y_f[(col+1)%NY]
else:
    for kk in range(Mc):
        r_total += H[r_idx,kk]*y_f[col-kk]
    for kk in range(NY-Mc, NY):
        r_total += H[r_idx,kk]*y_f[col-kk]

weight = (xx-X_up[l_idx, col])*os_factor

total = (1.0-weight)*l_total + weight*r_total

#%%
def testOS(m,n,Y,XX,H):
    MY,NY = Y.shape 
    MXX, NXX = XX.shape
    MY, NY = Y.shape 
    os_factor, NH = H.shape
    
    xx = XX[m,n]
    col = int(xx)
    r_idx = 0
    for r_idx in range(os_factor):
        if col+r_idx/os_factor > xx:
            break
    if r_idx == os_factor-1:
        l_idx = r_idx
    else:
        l_idx = r_idx - 1

    """ Check if xx is an integer """
    if col == xx:
        return Y[m,col]
    
    """ Run the convolution """
    Mc = 32
    l_total = 0.0
    r_total = 0.0
    for kk in range(Mc):
        l_total += H[l_idx,kk]*Y[m, col-kk]
    for kk in range(NY-Mc, NY):
        l_total += H[l_idx,kk]*Y[m, col-kk]
        
    if l_idx == r_idx:
        r_total = Y[m, ((col+1)%NY)]
    else:
        for kk in range(Mc):
            r_total += H[r_idx,kk]*Y[m, col-kk]
        for kk in range(NY-Mc, NY):
            r_total += H[r_idx,kk]*Y[m, col-kk]

    weight = (xx- (col + l_idx/os_factor))*os_factor

    total = (1.0-weight)*l_total + weight*r_total
    
    return total

#%% Make a plot of the interpolated signal    
txidx = range(8200, 9200)
yy_int = [testOS(m, n, Y, XX, H) for n in txidx]  
 
plt.figure()
rfn = np.real
plt.plot(nrng1, rfn(yos))
plt.plot(XX[m,txidx], rfn(yy_int),'.')
plt.grid()
    

#%%
    # X_up = np.array([nrng+row/os_factor for row in range(os_factor)])

    # right_bookend = 0
    # while XX[m,n] > X[m, right_bookend] and right_bookend < NY:
    #     right_bookend += 1
        
    # idX = right_bookend - 1
        
    # delta = XX[m,n] - X[m, idX]
    
    # dx = X[m, 1] - X[m, 0]
    
    # idx_lft = int(delta/dx*oversample)
    # idx_rgt = idx_lft + 1
    
    # rel_lft = oversample*delta/dx - idx_lft
    
    # # t_lft = t_rgt = 0.0
    # # for k in range(NY):
    # #     if k < NY/2:
    # #         myidx = (k+idX)%NY
    # #     else:
    # #         #myidx = NY - (k + idX)
    # #         myidx = k + idX - NY
            
    # #     #total += (1.0 - rel_lft)*Y[myidx, n]*H[k, idx_lft]
    # #     t_lft += Y[m, myidx]*H[idx_lft, k]
    # #     t_rgt += Y[m, myidx]*H[idx_rgt, k]
      
    # kidx = np.argsort(FFT_freq(NY, NY, 0))
    # t_lft = t_rgt = 0.0
    # for k in range(NY):
    #     xidx = kidx[k]
    #     hidx = kidx[idX - k]
    #     # if k < NY/2:
    #     #     myidx = (k+idX)%NY
    #     # else:
    #     #     #myidx = NY - (k + idX)
    #     #     myidx = k + idX - NY
            
    #     #total += (1.0 - rel_lft)*Y[myidx, n]*H[k, idx_lft]
    #     t_lft += Y[m, xidx]*H[idx_lft, hidx]
    #     t_rgt += Y[m, xidx]*H[idx_rgt, hidx]
       
    # return t_lft*(1.0-rel_lft) + t_rgt*rel_lft

#%%    
plt.figure()
xidx = range(n-20, n+20)
yint = [testOS(m,x,X,Y,XX,H) for x in xidx]
rfn = np.abs
plt.plot(xidx, rfn(Y[m, xidx]))
plt.plot(xidx, rfn(Y[m, xidx]), '.')
plt.plot(XX[m,xidx], rfn(yint), 'x')
plt.grid()
    
#%%

# X_d = cuda.to_device(X)
# Y_d = cuda.to_device(Y)
# XX_d = cuda.to_device(XX)
# YY_d = cuda.to_device(YY)
# H_d = cuda.to_device(H)