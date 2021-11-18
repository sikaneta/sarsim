#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:26:55 2020

@author: ishuwa
"""


import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
from measurement.measurement import state_vector_ESAEOD
from measurement.measurement import state_vector_RSO
from measurement.measurement import state_vector
from measurement.arclength import slow
import argparse
<<<<<<< HEAD
from geoComputer.geoComputer import satGeometry as sG
=======
orbpath = "C:/Users/Ishuwa.Sikaneta/local/sarsim/radar/orbit"
orbfile = "S1B_OPER_AUX_POEORB_OPOD_20180825T110641_V20180804T225942_20180806T005942.EOF"
>>>>>>> 1998465c1355fd575e73239869f044f550e404d8

#%% Load the data
purpose = """
          Plot some figures showing the expansion of the orbit using the
          differential geometry approach. Also plots the numerically
          integrated (egm96) against the supplied ground truth data. The
          script will take a single state vector from the supplied file
          and propagate this vector backwards and forwards in time 
          (symetrically) using two different methods:
              a. By numerical integration of the equations of motion
              b. By using the differential geometry approach
          Plots are created that compare the difference between the 
          numerically integated values and values in the supplied file
          where the time of the state vector is identical.
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--orbit-file", 
                    help="The orbit file containing ground truth data", 
                    default = os.path.join(orbpath, orbfile),
                    )
parser.add_argument("--plot-name",
                    help="A name signature for the output plots",
                    default = "./orbitPropagation.pgf")
                    #required=True)
parser.add_argument("--index",
                    help="""Index into the state vector array to use as
                            a point of expansion""",
                    type=int,
<<<<<<< HEAD
                    default=20)
parser.add_argument("--range",
                    help="Range to a target on the ground (m)",
                    type=float,
                    default=800000.0)
parser.add_argument("--hae",
                    help="Height of the target oabove the ellipsoid (WGS84 in m)",
                    type=float,
                    default=0.0)
=======
                    default=100)
>>>>>>> 1998465c1355fd575e73239869f044f550e404d8
parser.add_argument("--state-fs",
                    help="Sampling frequency of the propagated state vectors (Hz)",
                    type=float,
                    default=2.0)
parser.add_argument("--period",
                    help="Time duration over which to propagate vectors (s)",
                    type=float,
                    default=6000.0)
parser.add_argument("--depression-angle",
                    help="""The depression angle to use for projection of 
                            error onto line-of-sight (in degrees)""",
                    type=float,
                    default=45.0)
parser.add_argument("--interactive",
                    help="Interactive mode. Program halts until figures are closed",
                    action="store_true",
                    default=True)
parser.add_argument("--validate-numerical",
                    help="Plot the difference between integration and the data on file",
                    action="store_true",
                    default=True)


#%% Parse the arguments           
vv = parser.parse_args()
fname_split = vv.plot_name.split(".")
fname_head = ".".join(fname_split[0:-1])
fname_tail = fname_split[-1]

#%% Load the state vectors
if "EOF" in vv.orbit_file:
    sv = state_vector_ESAEOD(vv.orbit_file)
elif ".dat" in vv.orbit_file:
    sv = state_vector_RSO(vv.orbit_file)
    

#%% Extract the desired vector around which to propagate
s_idx = vv.index
s_time = sv.measurementTime[s_idx]
s_vect = sv.measurementData[s_idx]

# Create a new state vector object to propagate
myrd = state_vector()
myrd.add(s_time, s_vect)

# def satellitePositionsArclength(myrd, prf = 10.0, nSamples=80):
nSamples = int(vv.period*vv.state_fs)
prf = vv.state_fs

deltaT = 1.0/prf

xState = myrd.expandedState(myrd.measurementData[0], 0.0)

# Compute the boradside target position
satG = sG()
groundXYZ, error = satG.computeECEF(s_vect, 0.0, vv.range, vv.hae)
print("Computed ground point:")
print("==========================================")
print(groundXYZ)
print("==========================================")

# reference_time = myrd.measurementTime[0]
reference_time = np.datetime64(datetime.datetime.strftime(myrd.measurementTime[0], "%Y-%m-%dT%H:%M:%S.%f"))

np_prf = np.timedelta64(int(np.round(1e9/prf)),'ns')
svTime = myrd.measurementTime[0]
print("The state vector around which to propagate")
print("==========================================")
print("Reference time:")
print(reference_time)
print("Pos, Vel, Acc, dAcc:")
print(xState)
format_string = "http://192.168.0.174:5000/item?timeUTC=%s&x=%0.6f&y=%0.6f&z=%0.6f&vx=%0.6f&vy=%0.6f&vz=%0.6f&dt=0"
print(format_string % (reference_time, 
                       xState[0,0],
                       xState[0,1],
                       xState[0,2],
                       xState[1,0],
                       xState[1,1],
                       xState[1,2]))

#%% Compute satellite geographic coordinates
llh = sv.xyz2polar(sv.measurementData[s_idx][0:3])
print("lat,lon,R: (%0.4f, %0.4f, %0.1f)" % llh)

#%% Compute the integration times and integrate
half_time = deltaT*nSamples/2
integration_times = np.arange(-half_time, half_time+deltaT, deltaT)
numerical_sv = myrd.estimateTimeRange(myrd.measurementTime, integrationTimes=integration_times)

#%% Extract what we'll need
npos = numerical_sv[:,0:3].T
nvel = numerical_sv[:,3:6].T

#%% Do the differential geometry calculation
C = slow([reference_time])
C.t = integration_times
dummy = C.diffG(xState)

# Convert time to arclength
C.t2s()

#%% Compute the integration times that have entries in the data file
mysv_secs = [(st - sv.measurementTime[s_idx]).total_seconds() for st in sv.measurementTime]
mysv_secs = np.array([(idx,st) for idx,st in enumerate(mysv_secs) if st>=-half_time and st <=half_time])
print("State_vec index, Time (s)")
print(mysv_secs)
s_idxs = [np.argmin(np.abs(integration_times - st[1])) for st in mysv_secs]
subset_integrated = npos[:,s_idxs]
subset_times = np.array([k[1] for k in mysv_secs])
from_file = np.array([sv.measurementData[int(k[0])][0:3] for k in mysv_secs]).T
from_file - subset_integrated

#%% Plot the data
if vv.validate_numerical:
    plt.figure()
    plt.plot(integration_times, (C.c - npos).T, 
             subset_times, (from_file[0,:]-subset_integrated[0,:]).T, 'x', 
             subset_times, (from_file[1,:]-subset_integrated[1,:]).T, '1', 
             subset_times, (from_file[2,:]-subset_integrated[2,:]).T, '+')
    print("Difference at extremes")
    print("="*80)
    print(from_file[:,-1]-subset_integrated[:,-1])
    plt.grid()
    plt.xlabel('time (s)')
    plt.ylabel('difference (m)')
    plt.legend(['ECEF-X','ECEF-Y','ECEF-Z', 'diffIntState-X', 'diffIntState-Y', 'diffIntState-Z'])
    plt.title('Difference between integration and expansion')
else:
    plt.figure()
    plt.plot(integration_times, (C.c - npos).T)
    plt.grid()
    plt.xlabel('time (s)')
    plt.ylabel('difference (m)')
    plt.legend(['ECEF-X','ECEF-Y','ECEF-Z'])
    plt.title('Difference between integration and expansion')

fname = ".".join([fname_head + "_%d_error" % vv.index,
                  fname_tail])
if vv.interactive:
    plt.show()
else:
    plt.savefig(fname)
    plt.close()

#%% Plot the projection onto a 45 degree depression angle
satRhat = npos/np.outer(np.ones((3,)), np.linalg.norm(npos, axis=0))
satVhat = nvel/np.outer(np.ones((3,)), np.linalg.norm(nvel, axis=0))
satBhat = np.cross(satRhat, satVhat, axis=0)
bNorm = 1 - np.sum(satRhat*satVhat, axis=0)
satBhat = satBhat/np.outer(np.ones((3,)), bNorm)

# Calculate the 30 depression angle curve
lookAngle = vv.depression_angle/180*np.pi
satLook = np.sin(lookAngle)*satBhat + np.cos(lookAngle)*satRhat

# Plot the projection of the difference vector in the 30 degree direction
projectionLook = np.sum((C.c-npos)*satLook, axis=0)
plt.figure()
plt.plot(integration_times, projectionLook)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('Projection (m)')
plt.title('Projection of error onto the look direction\n(%0.1f depression angle)' % vv.depression_angle)

fname = ".".join([fname_head + "_%d_errorProjection_%0.1f" % (vv.index, 
                                                              vv.depression_angle),
                  fname_tail])
if vv.interactive:
    plt.show()
else:
    plt.savefig(fname)
    plt.close()
<<<<<<< HEAD
    
#%% Plot the range as a function of sat position
range_history = np.linalg.norm(npos - np.outer(groundXYZ, np.ones((npos.shape[1],))), axis=0)
plt.figure()
plt.plot(integration_times, range_history)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('Range history (m)')
plt.title('Range history with near range: %0.6f)' % vv.range)

fname = ".".join([fname_head + "_%d_range_history_%0.6f" % (vv.index, 
                                                            vv.range),
                  fname_tail])
                  
if vv.interactive:
    plt.show()
else:
    plt.savefig(fname)
    plt.close()
=======
>>>>>>> 73ffde9e593c5c597417ab12fd2f8edc3e2488d9
