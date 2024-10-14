import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.cm

    
class instrument:
    def __init__(self, 
                 duty_cycle = 0.05, 
                 antenna_az = 6.0,
                 antenna_el = 0.6,
                 carrier = 3.15e9):
        self.duty_cycle = duty_cycle
        self.antenna_az = antenna_az
        self.antenna_el = antenna_el
        self.carrier = carrier

class timingPlot:
    def __init__(self,
                 persp, 
                 off_nadir = None, 
                 prf_range = None, 
                 iminmax = [15,45],
                 field = {"name": "incidence",
                          "scaling_factor": 1,
                          "target_incidence_range": [25.5,34.5],
                          "label": r"Angle incidence $\theta_i$ / deg"},
                 instr = instrument()):
        self.iminmax = iminmax
        self.prf_range = prf_range or np.arange(1000, 5000, 10)
        self.instr = instr
        self.setPerspective(persp)
        off_nadir = off_nadir or np.radians(np.linspace(15, 40, 40))
        self.computeGeometry(off_nadir)
        
    def setPerspective(self, persp):
        self.persp = persp
        self.dopp_BW = 2*np.linalg.norm(persp.sPosVel[3:])/self.instr.antenna_az
        self.nadir_time = 2*persp.satHAE/const.c
        self.plot_title = "Time: %s, HAE: %0.1f km" % (persp.sTime, 
                                                       persp.satHAE/1e3)
        
    def computeGeometry(self, off_nadir):
        prf_range = self.prf_range
        instr = self.instr
        self.my_geometry = self.persp.computeGeometry(off_nadir)
        self.time_of_arrival = self.my_geometry["range"] * 2 / const.c
        self.min_rank = np.floor(self.time_of_arrival[0]*prf_range[0])
        
        """ Compute forbidden times from nadir returns """
        self.nadir_forbidden = [self.nadir_time*np.ones_like(prf_range),
                                self.nadir_time*np.ones_like(prf_range) + 
                                self.instr.duty_cycle/prf_range]
                    
        """ Compute forbidden times from transmit events """  
        tx_trigger = self.min_rank/prf_range
        self.tx_forbidden = [tx_trigger - instr.duty_cycle/prf_range,
                             tx_trigger + instr.duty_cycle/prf_range]
                
        """ Determine the minimum rank """
        rk = 1
        while (max(rk/prf_range + self.tx_forbidden[0]) 
               < min(self.time_of_arrival) 
               and rk < 100):
            rk = rk + 1
        self.min_rank = rk
        
    def computeForbidden(self, domain):
        nadir_polygons = self.computeForbiddenPolygons(self.nadir_forbidden,
                                                       domain)
        tx_polygons = self.computeForbiddenPolygons(self.tx_forbidden,
                                               domain)
        
        return nadir_polygons, tx_polygons
        
    def computeForbiddenReturnIdxs(self,
                                   #time_of_arrival, 
                                   forbidden_times, 
                                   #prf_range,
                                   rank = None):
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
        time_of_arrival = self.time_of_arrival
        prf_range = self.prf_range
        
        pri_range = 1/prf_range
        rank = rank or self.min_rank
        
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
    
    @staticmethod
    def computeForbiddenLine(targetIdx, domain):
        myinc = np.interp(targetIdx["idxs"], 
                          np.arange(len(domain)), 
                          domain,
                          left = np.nan,
                          right = np.nan)
        validInc = myinc != np.nan
        return {"prf": targetIdx["prf"][validInc],
                "domain": myinc[validInc],
                "rank": targetIdx["rank"]}
    
    def computeForbiddenPolygons(self,
                                 forbiddenTimeArrays,
                                 domain,
                                 rank=None):
            
        fIdxs = [self.computeForbiddenReturnIdxs(forbidden_times, rank=rank)
                 for forbidden_times in forbiddenTimeArrays]
        
        fLines = [[self.computeForbiddenLine(Idx, domain) for Idx in lIdxs]
                   for lIdxs in fIdxs]
        
        polygons = [np.vstack((np.hstack((l1['domain'], 
                                          l2['domain'][::-1])), 
                               np.hstack((l1['prf'], 
                                          l2['prf'][::-1])))).T
                    for l1, l2 in zip(fLines[0], fLines[1])]
        
        return polygons, [x['rank'] for x in fIdxs[0]]
      
    def field2incidence(self,
                        field,
                        fieldVals):
        
        return np.interp(fieldVals,
                         self.my_geometry[field],
                         self.my_geometry["incidence"])
    
    def incidencePatch(self, 
                       min_inc, 
                       max_inc, 
                       field, 
                       name = None, 
                       color = 'lightgreen',
                       prf_range = None,
                       fillpattern = None,
                       showlegend=True):
        prf_range = prf_range or self.prf_range
        domain = self.my_geometry[field]
        incidence = self.my_geometry["incidence"]
        x_min, x_max = np.interp([min_inc, max_inc], 
                                 incidence, 
                                 domain)
        y_min = min(prf_range)
        y_max = max(prf_range)
        return_dict = {'x': [x_min, x_min, x_max, x_max],
                'y': [y_min, y_max, y_max, y_min],
                'name': name,
                'fillpattern': fillpattern,
                'mode': 'lines',
                'fill': 'toself',
                'showlegend': showlegend,
                'line': {'color': color, 'width': 2}}
        
        return {k:v for k,v in return_dict.items() if v is not None}
        
    def iPatch(self, 
               min_inc, 
               max_inc, 
               domain,
               fillpattern = None,
               facecolor='lightblue'):
        prf_range = self.prf_range
        incidence = self.my_geometry["incidence"]
        x_min, x_max = np.interp([min_inc, max_inc], incidence, domain)
        y_min = min(prf_range)
        y_max = max(prf_range)
        return patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1, 
            edgecolor='r', 
            fillpattern=fillpattern,
            facecolor=facecolor, 
            alpha=0.4, 
            label= "Incidence (%0.1f to %0.1f deg)" % (min_inc, max_inc)
        )

    
    def populateSubplot(self, 
                        fig,
                        axs, 
                        #prf_range, 
                        field, 
                        scale = 1,
                        iminmax = [20,40],
                        tminmax = [25.5, 34.5],
                        c_map_use = 'viridis'):
        dopp_BW = self.dopp_BW
        min_rank = self.min_rank
        domain = self.my_geometry[field]*scale
        rect = self.iPatch(iminmax[0], iminmax[-1], domain)
        trect = self.iPatch(tminmax[0], 
                            tminmax[-1], 
                            domain,
                            fillpattern = {'shape': '-'},
                            facecolor = 'lightgreen')
        axs.add_patch(rect)
        axs.add_patch(trect)
    
        # Nyquist PRF
        cmap = matplotlib.colormaps[c_map_use]
        nyquist = axs.plot(
            domain, dopp_BW*np.ones(len(domain),),
            label='Nyquist PRF = %0.2f kHz' % (dopp_BW*1e-3),
            color='k', linestyle='dashed'
        )
    
        # Tx
        tx_polygons = self.computeForbiddenPolygons(self.tx_forbidden,
                                               domain)
        polyLen = len(tx_polygons)
        color_SF = int(cmap.N/polyLen)
        mypatches = [patches.Polygon(p, 
                                     closed = True,
                                     facecolor = cmap(color_SF*k),                      
                                     ) for k,p in enumerate(tx_polygons)]
        pCollection = PatchCollection(mypatches, 
                                      match_original=True, 
                                      cmap = cmap,
                                      array = min_rank + np.arange(polyLen))
        myp = axs.add_collection(pCollection)
        
    
        """ Plot Nadir return lines """
        nadir_polygons = self.computeForbiddenPolygons(self.nadir_forbidden,
                                                       domain,
                                                       rank=1)
        nadpatches = [patches.Polygon(p, 
                                      closed = True,
                                      facecolor = 'darkorange',
                                      label = "Nadir Return"
                                      ) for k,p in enumerate(nadir_polygons)]
        nadCollection = PatchCollection(nadpatches, 
                                        match_original=True)
        _ = axs.add_collection(nadCollection)
        
        axs.legend(loc='upper center',
                   handles = [nyquist[0], rect, nadpatches[0], trect],
                   bbox_to_anchor=(0.5, -0.12), 
                   fancybox=True, 
                   shadow=False, 
                   ncol=2)
        
        plt.grid(False)
        plt.ylabel('PRF / Hz')
        plt.tight_layout()
        
        return myp
        
        
# #%%
# def plot_diamond(persp, 
#                  off_nadir = None, 
#                  prf_range = None, 
#                  field = {"name": "incidence",
#                           "scaling_factor": 1,
#                           "target_incidence_range": [25.5,34.5],
#                           "label": r"Angle incidence $\theta_i$ / deg"},
#                  instr = instrument()):
    
#     dopp_BW = 2*np.linalg.norm(persp.sPosVel[3:])/instr.antenna_az
#     off_nadir = off_nadir or np.radians(np.linspace(15, 40, 40))
#     prf_range = prf_range or np.arange(1000, 5000, 10)
#     nadir_time = 2*persp.satHAE/const.c

#     """ Compute the geometry """
#     #range_slant, incidence, bearing, gSwath = persp.computeGeometry(off_nadir)
#     my_geometry = persp.computeGeometry(off_nadir)
#     range_slant = my_geometry["range"]
#     incidence = my_geometry["incidence"]
#     domain = my_geometry[field["name"]]*field["scaling_factor"]
    
#     time_of_arrival = range_slant * 2 / const.c
#     min_rank = np.floor(time_of_arrival[0]*prf_range[0])
    
    
#     """ Compute forbidden times from nadir returns """
#     nadir_forbidden = [nadir_time*np.ones_like(prf_range),
#                        nadir_time*np.ones_like(prf_range) + 
#                        instr.duty_cycle/prf_range]
    
#     nadir_polygons = computeForbiddenPolygons(time_of_arrival, 
#                                               nadir_forbidden,
#                                               prf_range,
#                                               domain)
    
    
                
#     """ Compute forbidden times from transmit events """  
#     tx_trigger = min_rank/prf_range
#     forbidden = [tx_trigger - instr.duty_cycle/prf_range,
#                  tx_trigger + instr.duty_cycle/prf_range]
    
#     polygons = computeForbiddenPolygons(time_of_arrival, 
#                                         forbidden,
#                                         prf_range,
#                                         domain)

#     """ Plot the diamond diagram """
#     plot_title = "Time: %s, HAE: %0.1f km" % (persp.sTime, persp.satHAE/1e3)
#     fig, axs = plt.subplots(num=str(persp.sTime), figsize=(8,6))
    
#     # def populateSubplot(axs, 
#     #                     prf_range, 
#     #                     domain,  
#     #                     c_map_use = 'viridis'):
#     #     # Design
#     #     # c_map_use = 'viridis'
        
#     #     # Swath
#     #     rect = iPatch(20, 40, prf_range, domain, incidence)
#     #     axs.add_patch(rect)
#     #     if "target_incidence_range" in field.keys():
#     #         imin,imax = field["target_incidence_range"]
#     #     else:
#     #         imin,imax = (25.5,34.5)
#     #     trect = iPatch(imin, 
#     #                    imax, 
#     #                    prf_range, 
#     #                    domain, 
#     #                    incidence, 
#     #                    facecolor='lightgreen')
#     #     axs.add_patch(trect)
    
#     #     # Nyquist PRF
#     #     cmap = matplotlib.colormaps[c_map_use]
#     #     nyquist = axs.plot(
#     #         domain, dopp_BW*np.ones(len(domain),),
#     #         label='Nyquist PRF = %0.2f kHz' % (dopp_BW*1e-3),
#     #         color='k', linestyle='dashed'
#     #     )
    
#     #     # Tx
#     #     color_SF = int(cmap.N/len(polygons))
#     #     mypatches = [patches.Polygon(p, 
#     #                                  closed = True,
#     #                                  facecolor = cmap(color_SF*k),                      
#     #                                  ) for k,p in enumerate(polygons)]
#     #     pCollection = PatchCollection(mypatches, 
#     #                                   match_original=True, 
#     #                                   cmap = cmap,
#     #                                   array = min_rank + np.arange(len(polygons)))
#     #     myp = axs.add_collection(pCollection)
#     #     axcb = fig.colorbar(myp)
#     #     axcb.set_label('Rank / 1')
        
    
#     #     """ Plot Nadir return lines """
#     #     nadpatches = [patches.Polygon(p, 
#     #                                   closed = True,
#     #                                   facecolor = 'darkorange',
#     #                                   label = "Nadir Return"
#     #                                   ) for k,p in enumerate(nadir_polygons)]
#     #     nadCollection = PatchCollection(nadpatches, 
#     #                                     match_original=True)
#     #     _ = axs.add_collection(nadCollection)
        
#     #     axs.legend(loc='upper center',
#     #                handles = [nyquist[0], rect, nadpatches[0], trect],
#     #                bbox_to_anchor=(0.5, -0.12), 
#     #                fancybox=True, 
#     #                shadow=False, 
#     #                ncol=2)
    
        
#     """ Axis labels """
#     populateSubplot(axs, prf_range, domain)
#     plt.grid(False)
#     plt.title(plot_title)
#     plt.xlabel(field["label"])
#     plt.ylabel('PRF / Hz')
#     # axs.legend(loc='upper center',
#     #            handles = [nyquist[0], rect, nadpatches[0], trect],
#     #            bbox_to_anchor=(0.5, -0.12), 
#     #            fancybox=True, 
#     #            shadow=False, 
#     #            ncol=2)
#     plt.tight_layout()
    
