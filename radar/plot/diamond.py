import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.cm

from orbit.geometry import perspective

# Timing Plots
persp = perspective(sPosVel = svdata, planet = venus())

#%%
def computeForbiddenReturnIdxs(time_of_arrival, 
                               forbidden_times, 
                               prf_range,
                               rank = 1):
    """
    Find forbidden return events
    
    Find where forbidden returns appear in the diamond diagram. The position
    of these returns is PRF dependent (since they repeat at the PRF) as 
    illustrated below.
    
    Examples of forbidden times are: the transmit event of the first pulse 
    and the time of nadir return.
    
    This function works by finding intersections of the forbidden time(s) plus
    some multiple of the PRF (which is, of course, PRF dependent) and the 
    return from off-nadir. More specifically, it finds intersections of 
    forbidden_time + rank/prf_range, where the prf_range is an array of 
    different PRFs. This is computed for increasing values of rank until 
    there are no intersections found.
    
    The parameter rank is the starting point for the range of values for k. 
    
    
    Time           Time of arrival co-incident
    ^              with forbidden return (x)
    |                           |             - Return from
    |                           |         ---/  off-nadir
    |                           |   -----/
    |========================---x--/========== Forbidden time 
    |                 ------/          |       from next pulse
    |         -------/                 |       (PRF-dependent)
    |--------/                        PRI      
    |                                  |
    |                                  |
    |
    |
    |                        
    |-----------------------------------------
    |-------- Forbidden Times f(PRF) ---------
    |           
    |                  |
    |                  |
    |                 PRI
    |                  |
    |                  |
    |--------------------------------------- >
                  Time of arrival


    In the above diagram, Forbidden Time could be the nadir return time. The
    nadir return time is independent of the PRF.
    
    Instead, the Forbidden times could correspond to times when the radar is
    transmitting (since one cannot receive while transmitting). In this case,
    the duration of the pulse may be PRF-dependent, such as when it is tied to
    the duty cycle. So the end of pulse transmission could be slightly earlier
    for a higher PRF so that one can maintain the overall "on" or "active" 
    time.
    
    
    Note
    ----
    
    Computations are made using interpolation. Thus the time_of_arrival
    array and the prf_range arrays should be sorted
    
    Parameters
    ----------
    time_of_arrival : `numpy.ndarray(N,)`
        An N-d array with the time of arrival of reflected radar signals. This
        array typically spans a range of incidence angles. Sorted.
    forbidden_times :  `float` or `numpy.ndarray(N,)`
        The forbidden time value(s). If an array, then the array has to span
        prf_range.
    prf_range :  `numpy.ndarray(M,)`
        An array with a range of M different PRF values. Sorted.
    k0 : `int`
        The starting rank from which to iterate upwards

    Returns
    -------
    allIdxs : `dict`
        A dictionary with fields representing the forbidden return locations.

    """
    pri_range = 1/prf_range
    idxs = np.arange(len(time_of_arrival))
    allIdxs = []
    rankIdxs = np.array([None])
    while rankIdxs.size > 0:
        rankIdxs = np.interp(rank*pri_range + forbidden_times, 
                             time_of_arrival, 
                             idxs, 
                             left=-1, 
                             right=-1)
        forbiddenPRF = rankIdxs > 0
        rankIdxs = rankIdxs[forbiddenPRF]
        if rankIdxs.size > 0:
            
            allIdxs.append({"rank": rank,
                            "prf": prf_range[forbiddenPRF],
                            "idxs": rankIdxs})
        rank = rank + 1
    return allIdxs

def computeForbiddenLine(targetIdx, incidence):
    myinc = np.interp(targetIdx["idxs"], 
                      np.arange(len(incidence)), 
                      incidence,
                      left = 1000,
                      right = 1000)
    validInc = myinc < 1000
    return {"prf": targetIdx["prf"][validInc],
            "incidence": myinc[validInc]}

def computeForbiddenPolygons(time_of_arrival,
                             forbiddenTimeArrays,
                             prf_range,
                             incidence):
    fIdxs = [computeForbiddenReturnIdxs(time_of_arrival, 
                                        forbidden_times, 
                                        prf_range)
             for forbidden_times in forbiddenTimeArrays]
    
    fLines = [[computeForbiddenLine(Idx, incidence) for Idx in lIdxs]
               for lIdxs in fIdxs]
    
    polygons = [np.vstack((np.hstack((l1['incidence'], 
                                      l2['incidence'][::-1])), 
                           np.hstack((l1['prf'], 
                                      l2['prf'][::-1])))).T
                for l1, l2 in zip(fLines[0], fLines[1])]
    
    return polygons
    
class instrument:
    def __init__(self, duty_cycle = 0.05, antenna_az = 6.0):
        self.duty_cycle = duty_cycle
        self.antenna_az = 6.0
        
#%%
def plot_diamond(persp, 
                 off_nadir = None, 
                 prf_range = None, 
                 instr = instrument()):
    
    dopp_BW = 2*np.linalg.norm(persp.sPosVel[3:])/instr.antenna_az
    off_nadir = off_nadir or np.radians(np.arange(15, 40, 0.1))
    prf_range = prf_range or np.arange(1000, 5000, 10)
    nadir_time = 2*persp.satHAE/const.c

    """ Compute the geometry """
    range_slant, incidence, bearing, gSwath = persp.computeGeometry(off_nadir)
    
    time_of_arrival = range_slant * 2 / const.c
    min_rank = np.floor(time_of_arrival[0]*prf_range[0])
    
    
    """ Compute forbidden times from nadir returns """
    nadir_forbidden = [nadir_time*np.ones_like(prf_range),
                       nadir_time*np.ones_like(prf_range) + 
                       instr.duty_cycle/prf_range]
    
    nadir_polygons = computeForbiddenPolygons(time_of_arrival, 
                                              nadir_forbidden,
                                              prf_range,
                                              incidence)
    
    
                
    """ Compute forbidden times from transmit events """  
    tx_trigger = min_rank/prf_range
    forbidden = [tx_trigger - instr.duty_cycle/prf_range,
                 tx_trigger + instr.duty_cycle/prf_range]
    
    polygons = computeForbiddenPolygons(time_of_arrival, 
                                        forbidden,
                                        prf_range,
                                        incidence)

    """ Plot the diamond diagram """
    plot_name = 'Timing_Diagram'
    fig, axs = plt.subplots(num=plot_name, figsize=(8,6))
    # Design
    c_map_use = 'viridis'
    # Swath
    x_min = 20 #np.amin(mode.angle_incidence*180/np.pi)
    x_max = 40 #np.amax(mode.angle_incidence*180/np.pi)
    y_min = min(prf_range)
    y_max = max(prf_range)
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1, 
        edgecolor='r', 
        facecolor='lightblue', 
        alpha=0.4, 
        label='Total Access'
    )
    axs.add_patch(rect)
    # Nyquist PRF
    cmap = matplotlib.colormaps[c_map_use]
    nyquist = axs.plot(
        incidence, dopp_BW*np.ones(len(incidence),),
        label='Nyquist PRF = %2.3f kHz' % (dopp_BW),
        color='k', linestyle='dashed'
    )
    # Tx
    color_SF = int(cmap.N/len(polygons))
    mypatches = [patches.Polygon(p, 
                                 closed = True,
                                 facecolor = cmap(color_SF*k),                      
                                 ) for k,p in enumerate(polygons)]
    pCollection = PatchCollection(mypatches, 
                                  match_original=True, 
                                  cmap = cmap,
                                  array = min_rank + np.arange(len(polygons)))
    myp = axs.add_collection(pCollection)
    axcb = fig.colorbar(myp)
    axcb.set_label('Rank / 1')
        
    
    """ Plot Nadir return lines """
    nadpatches = [patches.Polygon(p, 
                                  closed = True,
                                  facecolor = 'darkorange',
                                  label = "Nadir Return"
                                  ) for k,p in enumerate(nadir_polygons)]
    nadCollection = PatchCollection(nadpatches, 
                                    match_original=True)
    _ = axs.add_collection(nadCollection)
    
        
    """ Axis labels """
    plt.grid(False)
    plt.xlabel(r'Angle incidence $\theta_i$ / deg')
    plt.ylabel('PRF / Hz')
    axs.legend(loc='upper center',
               handles = [nyquist[0], rect, nadpatches[0]],
               bbox_to_anchor=(0.5, -0.12), 
               fancybox=True, 
               shadow=False, 
               ncol=2)
    plt.tight_layout()
