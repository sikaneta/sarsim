#%% Load modules
import numpy as np
import json
import os
from glob import glob
import sys
from itertools import product, combinations
from sklearn.cluster import AgglomerativeClustering

#%% Define the path for writing data
if 'linux' in sys.platform:
    filepath = '/users/isikanet/local/data/cycles'
else:
    filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs"

#%%
class imagingSolutions:
    """
    Class to hold a set of solutions
    
    This class represents a set of imaging pair solutions. It allows 
    retrieval of the minimum cost of the set of solutions, allows
    filtering of the imaging solutions according to some provided
    function and allows the set of imaging solutions to be cast as
    a json snippet.
    
    For instance, the set of imaging pair solutions can be for stereo
    or repeat imaging. 
    
    Methods
    -------
    
    add
        Function to add an imaging option pair
    minimum
        Returns the minimum imaging pair test values. These are the
        computed cost according to some criteria captured in the
        member variable T of the imaging pair
    testFilter
        Returns a list of filtered imaging solutions
    filt
        Returns a new imaging solutions object with a filtered set of
        solutions
    toJson
        Returns a python dictionary that represents the imaging solutions
        object
        
    ToDo
    ----
    
    - Implement a fromJson method
    - Document
    
    """
    def __init__(self, solutions = []):
        self.solutions = [x for x in solutions]
                
    def add(self, pair):
        """
        Add an imaging pair to this object

        Parameters
        ----------
        pair : imagingPair
            An imaging pair object. Can be designed for stereo or repeat.

        Returns
        -------
        None.

        """
        self.solutions.append(pair)
        
    def length(self):
        """
        Return the number of solutions

        Returns
        -------
        int
            The number of solutions.

        """
        return len(self.solutions)
        
    def minimum(self):
        """
        Compute the minimum of the imagingPair costs

        Returns
        -------
        float
            The minimum of all costs imagingPair.T for the pairs
            held by this imagingSolutions class.

        """
        if len(self.solutions) == 0:
            return None
        minidx = np.argmin([x.T for x in self.solutions])
        return self.solutions[minidx].toJson()
    
    def testFilter(self, fn):
        """
        Filter imagingSolutions list

        Parameters
        ----------
        fn : `bool function`
            A function that takes an imagingPair object and returns
            True or False.

        Returns
        -------
        list
            The list of imagingPairs for which fn evaluates to True.

        """
        return [x for x in self.solutions if fn(x)]
    
    def filt(self, fn):
        """
        Filter imagingSolutions list

        Parameters
        ----------
        fn : `bool function`
            A function that takes an imagingPair object and returns
            True or False.

        Returns
        -------
        `imagingSolutions`
            An imaging solutions object containing the filtered 
            imagingPairs.

        """
        imS = imagingSolutions()
        imS.solutions += self.testFilter(fn)
        return imS
    
    def consistentWith(self, other):
        return all([x & y for x,y in product(self.solutions,
                                             other.solutions)])
    
    def toJson(self):   
        """
        Cast imagingSolutions as json

        Returns
        -------
        `dict`
            A python dictionary representation of the this imagingSolutions.

        """
        return [x.toJson() for x in self.solutions]

#%%
""" Define function to return possibility of simultaneous 
    solutions """
def distributeSolutions(myOptions):
    """
    Find all consistent imagingSolutions
    
    Given a list of imagingSolutions of length N, compute a list
    of imagingSolutions, each with N imagingPairs, where one pair is 
    drawn from each of the input imagingSolutions such that the N 
    pairs are consistent with each other. In this context, consistent
    means that no two pairs in each of the return imagingSolutions use the
    same resource in a different manner. 
    
    A resource simply means use of the SAR mode in the neighbourhood of a 
    given time, so two imagingPairs are inconsistent if they both require
    use of the SAR at around the same time, but at two different incidence
    angles.
    
    For instance, the input list may be a list with two imagingSolutions, one
    representing all stereo pairs, one representing all repeat pairs. For each
    consistent choice of one stereo pair and one repeat pair, a new 
    imagingSolution will be created and appended to the list of returned 
    imagingSolutions. 
    
    Alternatively, the input list could be list consisting of 
    imagingSolutions for roiA and imagingSolutions for roiB. The return list
    would consist of all imagingSolutions with one pair corresponding to roiA
    and one pair corresponding to roiB.

    Parameters
    ----------
    myOptions : list of imagingSolutions
        A list of N imagingSolutions.

    Returns
    -------
    list
        A list of consistent imagingSolutions.

    """
    return [imagingSolutions(test)
            for test in product(*[z.solutions for z in myOptions])
            if all([x & y for x,y in combinations(test,2)])]
       
#%%
class imagingPair:
    """
    A class to represent imagingPairs
    
    An imaging pair can represent the time and incidence angles for repeat
    and stereo imaging.
    
    Methods
    -------
    
    computeScore
        Compute the score of the imaging pair according to the criterion. 
    __and__
        Overloaded & Method to check whether imagingPair is consistent with
        another. Returns the same as method independentFrom - True if 
        consistent
    consistentWith
        Method to determine if imagingPair is consistent with another
        imagingPair. Returns True if consistent
    toJson
        Method to return json representation
    """
    def __init__(self, opA, opB, criteria):
        self.ops = [opA, opB]
        self.type = criteria["name"]
        self.offset = criteria["offset"]
        self.threshold = criteria["threshold"]
        self.computeScore()
        
    def __and__(self, Y):
        """
        Overloaded & Method
        
        Checks whether imagingPair is consistent with
        another. Returns the same as method independentFrom - True if 
        consistent

        Parameters
        ----------
        Y : imagingPair
            ImagingPair against which to check consistency.

        Returns
        -------
        bool
            True if consistent, False otherwise.

        """
        return self.consistentWith(Y)
        
    def computeScore(self):
        """
        Compute the score of the imaging pair according to the criterion. This
        score is defined as  
        
        ..math::
            
            T = \lvert\lvert\tehta_1 - \theta_2\rvert - \theta_o\rvert 
            - \delta,
        
        where :math:`\theta_1` and :math:`\theta_2` are the incidence angles 
        of each observation. A value of T less than zero satisfies the 
        criterion. For instance, if the offset is zero, then when the absolute 
        value of incidence angle differences is less than the threshold 
        (criterion met), then T is negative.
        
        A positive value for T indicates by how may degrees the criteria is
        violated. For instance, if the criterion consisted of an offset of 5.5
        and a threshold of 1.5 and the two incidence angles were 32.5 and 24.5,
        then T would evaluate to abs(8-5.5) - 1.5 = 1, so the two incidence
        angles 'exceed' the criterion by 1. (Numbers are all degrees)

        Returns
        -------
        None.

        """
        self.T = np.abs(np.abs(self.ops[0].incidence - self.ops[1].incidence) 
                        - self.offset) - self.threshold
    
    def consistentWith(self, other):
        """
        Check if opportunities are consistent
        
        Method to determine whether imagingPair is consistent with
        another. Returns True if consistent.

        Parameters
        ----------
        other: :py:class:`<orbit.AnalyzePlan.imagingPair>`
            imagingPair object to compare with.

        Returns
        -------
        bool
            True if consistent.

        """
        for x,y in product(self.ops, other.ops):
            if not x.consistentWith(y):
                return False
        return True
    
    def toJson(self):
        """
        Cast imagingPair as json

        Returns
        -------
        `dict`
            A python dictionary representation of the this imagingPair.

        """

        return {"type": self.type,
                "score": self.T,
                "events": [x.toJson() for x in self.ops]}
        
class imagingUnit(imagingPair):
    def __init__(self, x):
        self.ops = [x]
        self.type = "unit"
        self.offset = 30
        self.threshold = 7
        self.computeScore()
        
    def computeScore(self):
        self.T = np.abs(self.ops[0].incidence - self.offset) - self.threshold
    
class opportunity:
    """
    Class to represent an imaging opportunity
    
    Methods
    -------
    
    consistentWith:
        Check whether this opportunity is consistent with another
    toJson:
        Represent this opportunity as a json snippet
    fromJson:
        Load class data from a json snippet
    fromGeoJson:
        Load class data from a geojson snippet
    
    """
    time_threshold = 10
    inc_threshold = 1
    def __init__(self, 
                 incidence = None,
                 orbitNumber = None,
                 cycle = None,
                 UTC = None,
                 ROI = None,
                 time_threshold = None, 
                 inc_threshold = None):
        """
        Class constructor

        Parameters
        ----------
        incidence : float, optional
            The incidence angle (degrees). The default is None.
        orbitNumber : int, optional
            The orbit number on which imaging will occur. The default is None.
        cycle : int, optional
            The cycle on which imaging will occur. The default is None.
        UTC : string, optional
            The time of imaging. The default is None.
        ROI : string, optional
            The name of the ROI that uses this opportunity
        time_threshold : float, optional
            Time threshold (minutes) for comparison. 
            The default is None, in which case 10 is used.
        inc_threshold : float, optional
            Incidence threshold (degrees) for comparison. 
            The default is None, in which case 1 is used.

        Returns
        -------
        None.

        """
        self.incidence = incidence
        self.orbitNumber= orbitNumber
        self.cycle = cycle
        self.UTC = UTC
        self.ROI = ROI
        self.inc_threshold = inc_threshold or self.inc_threshold
        self.time_threshold = time_threshold or self.time_threshold
        
    def consistentWith(self,  other):
        """
        Check for consistency with another opportunity
        
        This method tests to see whether this opportunity is consistent
        with another within the time and incidence thresholds. Consistency
        evaluates to True if the incidence angle can be changed from the
        incidence angle of this object to the incidence angle of myobj in
        the time difference between planned measurements. In all other cases,
        consistency evaluates to False
        
        Parameters
        ----------
        other : `opportunity`
            The imaging opportunity to compare with.

        Returns
        -------
        bool
            Whether or not this opportunity is consistent with myobj within
            the set thresholds (incidence and time) for this opprotunity.

        """
        num = np.abs(self.incidence - other.incidence)
        den = np.abs((np.datetime64(self.UTC) - 
                np.datetime64(other.UTC))/np.timedelta64(1, 'm'))
        
        if num == 0:
            return True
        if den != 0:
            return num/den < self.inc_threshold/self.time_threshold
        else:
            return False
    
    def toJson(self):
        """
        Return opportunity parameters as a dictionary

        Returns
        -------
        dict
            A dictionary with the opportunity parameters.

        """
        return {"incidence": self.incidence,
                "orbitNumber": self.orbitNumber,
                "cycle": self.cycle,
                "UTC": self.UTC,
                "ROI": self.ROI}
    
    def fromJson(self, myjson):
        """
        Set parameters from a dictionary

        Parameters
        ----------
        myjson : `dict`
            Input dictionary from which to set parameters.

        Returns
        -------
        None.

        """
        self.incidence = myjson["incidence"]
        self.orbitNumber= myjson["orbitNumber"]
        self.cycle = myjson["cycle"]
        self.UTC = myjson["UTC"]
        self.ROI = myjson["targetID"]
        
    def fromGeoJson(self, mygeojson):
        """
        Set parameters from a dictionary

        Parameters
        ----------
        mygeojson : `dict`
            Input dictionary from which to set parameters.

        Returns
        -------
        None.

        """
        self.incidence = mygeojson["incidence"]
        self.orbitNumber= mygeojson["orbitNumber"]
        self.cycle = mygeojson["cycle"]
        self.UTC = mygeojson["stateVector"]["IAU_VENUS"]["time"]
        self.ROI = mygeojson["targetID"]

class opportunityJson(opportunity):
    """
    Class derived from opportunity to use json on init
    
    """
    
    def __init__(self, 
                 myjson,
                 time_threshold = None, 
                 inc_threshold = None):
        """
        Constructor

        Parameters
        ----------
        myjson : `dict`
            Input json dictionary from which to set parameters.
        time_threshold : float, optional
            Time threshold (minutes) for comparison. 
            The default is None, in which case 10 is used.
        inc_threshold : float, optional
            Incidence threshold (degrees) for comparison. 
            The default is None, in which case 1 is used.

        Returns
        -------
        None.

        """
        
        self.fromJson(myjson)
        self.inc_threshold = inc_threshold or self.inc_threshold
        self.time_threshold = time_threshold or self.time_threshold
        

class opportunityGeoJson(opportunity):
    """
    Class derived from opportunity to use geojson on init
    
    """
    def __init__(self, 
                 mygeojson,
                 time_threshold = None, 
                 inc_threshold = None):
        """
        Constructor

        Parameters
        ----------
        mygeojson : `dict`
            Input geojson dictionary from which to set parameters.
        time_threshold : float, optional
            Time threshold (minutes) for comparison. 
            The default is None, in which case 10 is used.
        inc_threshold : float, optional
            Incidence threshold (degrees) for comparison. 
            The default is None, in which case 1 is used.

        Returns
        -------
        None.

        """
        
        self.fromGeoJson(mygeojson)
        self.inc_threshold = inc_threshold or self.inc_threshold
        self.time_threshold = time_threshold or self.time_threshold

#%%
class roiPlan:
    """
    Class to define an imaging plan for a given ROI
    
    Methods
    -------
    computeOptions:
        Computes imagingSolutions (stereo, repeat, both) for the roi
    toJson:
        Represents the class as a json snippet
        
    """
    direction = ['Ascending']
    passTypes = ["standard", "stereo", "polarimetry"]
    incidence_range = [23,37]
    version = "Plan"
    criteria = [{"name": "Repeat",
                 "offset": 0.0,
                 "threshold": 1.5},
                {"name": "Stereo",
                 "offset": 5.5,
                 "threshold": 1.5}]

    keymap = {"None": "Neither Stereo nor Repeat",
              "Stereo": "Only Stereo",
              "Repeat": "Only Repeat",
              "RepeatStereo": "Either Repeat or Stereo",
              "StereoRepeat": "Either Repeat or Stereo",
              "RepeatStereoBoth": "Both Stereo and Repeat",
              "StereoRepeatBoth": "Both Stereo and Repeat"}
        
    """ Filtering functions """
    @staticmethod
    def fnCompliant(x):
        return x.T < 0
    
    @staticmethod
    def fnRepeat(x): 
        return x.type == "Repeat"
    
    @staticmethod
    def fnStereo(x): 
        return x.type == "Stereo"
    
    def __init__(self, 
                 roi, 
                 filepath, 
                 sarorbits, 
                 UTC_filter = []):
        self.roi = roi
        self.filepath = filepath
        self.sarorbits = sarorbits
        self.UTC_filter = UTC_filter
        
    def computeOptions(self, omit_UTC = []):
        """
        Compute imagingSolutions for an roi
        
        Based upon the computed observation possibilities for the assigned roi
        and the filters for look direction, ascending/descending, assigned
        cycles, permitted orbit numbers for SAR imaging and desired incidence
        angle range, this function computes
        - an imagingSolutions object for all stereo imagingPairs 
        - an imagingSolutions object for all repeat imagingPairs 
        - an imagingSolutions object for all compliant stereo imagingPairs 
        - an imagingSolutions object for all compliant repeat imagingPairs 

        Parameters
        ----------
        omit_UTC : list of `numpy.datetime64`, optional
            List of times not permitted for imaging. The default is [].

        Returns
        -------
        None.

        """

        roi = self.roi
        direction = self.direction
        passTypes = self.passTypes
        incidence_range = self.incidence_range
        criteria = self.criteria
        sarOrbits = self.sarorbits
        filepath = self.filepath
        version = self.version
        
        # roi = eplan.plan["features"][k]
        # direction = eplan.direction
        # passTypes = eplan.passTypes
        # incidence_range = eplan.incidence_range
        # criteria = eplan.criteria
        # sarOrbits = eplan.sarorbits
        # filepath = eplan.filepath
        # version = eplan.version
        
        ROI = roi["properties"]["ROI_No"]
        
        """ Load the computed incidence angles """
        ifile = os.path.join(filepath, 
                             version, 
                             "incidence", 
                             "%s_incidence.geojson" % ROI)
        
        with open(ifile, "r") as ifl:
            pool = json.loads(ifl.read())
        
        cycles = []
        for pType in passTypes:
            if pType in roi["properties"]["plan"].keys():
                cycles += roi["properties"]["plan"][pType]
                
        UTC_skip = self.UTC_filter + omit_UTC

        cycleOps = [[opportunityGeoJson(p["properties"])
                     for p in pool["features"] if
                    p["properties"]["incidence"] < incidence_range[1] and
                    p["properties"]["incidence"] > incidence_range[0] and
                    p["properties"]["orbitNumber"] in sarOrbits and
                    p["properties"]["orbitDirection"] in direction and
                    p["properties"]["stateVector"]["IAU_VENUS"]["time"] not in 
                    UTC_skip and
                    p["properties"]["cycle"] == cycle] for cycle in 
                    sorted(cycles)]
                
        """ Gather all solutions """
        solutions = imagingSolutions()
        for cycle_pair in combinations(cycleOps, 2):
            for a,b in product(*cycle_pair):
                for score in criteria:
                    solutions.add(imagingPair(a,b, score))
                    
        self.units = imagingSolutions()
        for z in [x for y in cycleOps for x in y]:
            self.units.add(imagingUnit(z))
        
        self.allRepeat = solutions.filt(roiPlan.fnRepeat)
        self.allStereo = solutions.filt(roiPlan.fnStereo)
        self.compliant = solutions.filt(roiPlan.fnCompliant)
        self.fltRepeat = self.compliant.filt(roiPlan.fnRepeat)
        self.fltStereo = self.compliant.filt(roiPlan.fnStereo)
        
    def consistentWith(self, other):
        return self.compliant.consistentWith(other.compliant)
        
    def toJson(self):
        """
        Cast object to json
        
        This method represents this class as a json snippet.
        
        The method computes
        - The minimum repeat score
        - The minimum stereo score
        - a list of self consistent imagingSolutions each with one stereo and 
        one repeat imagingPair.
        - a key representing the imaging possibilities for stereo, repeat or 
        both

        Returns
        -------
        data : `dict`
            Json snippet as python dictionary.

        """
        solutions = self.fltRepeat.toJson() 
        solutions += self.fltStereo.toJson()
        solutions += self.units.filt(roiPlan.fnCompliant).toJson()
        both = distributeSolutions([self.fltRepeat, self.fltStereo])
        fltStereoMin = self.fltStereo.minimum()
        fltRepeatMin = self.fltRepeat.minimum()
        if both:
            minboth = np.argmin([x.filt(roiPlan.fnStereo).minimum()["score"] 
                                 for x in both])
            option = {"type": "Both Stereo and Repeat",
                      "solution": both[minboth].toJson()}
        elif fltStereoMin is not None and fltRepeatMin is not None:
            option = {"type": "Either Stereo or Repeat",
                      "solution": [fltStereoMin, fltRepeatMin]}
        elif fltStereoMin is not None:
            option = {"type": "Only Stereo",
                      "solution": [fltStereoMin]}
        elif fltRepeatMin is not None:
            option = {"type": "Only Repeat",
                      "solution": [fltRepeatMin]}
        else:
            option = {"type": "Neither Stereo nor Repeat",
                      "solution": [self.units.minimum()]}
            
        properties = {"ROI_No": self.roi["properties"]["ROI_No"],
                      "plan": self.roi["properties"]["plan"],
                      "minima": [self.allStereo.minimum(), 
                                 self.allRepeat.minimum()],
                      "imagingSolutions": solutions,
                      "compliant": option}
        
        return {"type": self.roi["type"],
                "geometry": self.roi["geometry"],
                "properties": properties}
        
#%% Define function to see if we have a solution
class plan:
    """
    Class to represent imaging plan
    
    This class reads an roi geojson file to determine the coordinates of a set
    of ROIs as well as the plan for imaging. This plan for imaging defines
    what SAR modes will be used on what cycle for each ROI. Additionally, the
    class relies on a hard-coded list of orbit numbers that identify which
    orbits can be allocated to SAR imaging.
    
    Methods
    -------
    computeElement:
        compute a roiPlan for a given index
    toJson:
        represent this class as a json snippet
    
    """
    validOrbitFile = "quindec_cycle_orbit_table.txt"
    roiFile = "roi.geojson"

    def __init__(self, filepath, version = "Plan"):
        self.filepath = filepath
        self.version = version
        self.readPlan()
        self.readValidOrbits()
        
    def readPlan(self, roiFile = None):
        """
        Ingest the imaging plan
        
        This method reads an roi geojson file to determine the coordinates of 
        a set of ROIs as well as the plan for imaging. This plan for imaging 
        defines what SAR modes will be used on what cycle for each ROI.
        
        If roiFile is not supplied, then the default is used. If roiFile is
        supplied, then the class roiFile variable is updated.

        Parameters
        ----------
        roiFile : str, optional
            The name of the ROI file. The default is None.

        Returns
        -------
        None.

        """
        self.roiFile = roiFile or self.roiFile
        
        # Open the roi file
        with open(os.path.join(self.filepath, 
                               self.version, 
                               self.roiFile), "r") as f:
            self.plan = json.loads(f.read())
        self.nFeatures = len(self.plan["features"])
        self.rois = [None]*self.nFeatures
        self.UTC_filter = [[] for x in range(self.nFeatures)]
        

    def readValidOrbits(self, validOrbitFile = None):
        """
        Ingest the valid SAR orbits
        
        This method reads a file describing which orbit numbers can be
        allocated to SAR imaging.
        
        if validOrbitFile is None, then the default file will be read. If it
        is not None, then the class variable is set to the supplied value.

        Parameters
        ----------
        validOrbitFile : str, optional
            The orbit file to read. The default is None.

        Returns
        -------
        None.

        """
        self.validOrbitFile = validOrbitFile or self.validOrbitFile
        with open(os.path.join(filepath, self.validOrbitFile), "r") as f:
            pool = [x.split() for x in f.read().split("\n")]
            pool = pool[0:-1]
            
        """ Load all orbits """
        pp = [[np.datetime64(x[0]), int(x[1]), int(x[2]), int(x[3])] 
              for x in pool]

        """ Filter for orbits where SAR is possible """
        ppsarstd = [p for p in pp if p[2] in [1,6,11]]
        self.sarorbits = [p[1] for p in ppsarstd]
        
    def appendUTC_filter(self, k, utc_list):
        self.UTC_filter[k] += utc_list
        self.UTC_filter[k] = sorted(list(set(self.UTC_filter[k])))
        
    def computeElements(self):
        for k in range(self.nFeatures):
            self.computeElement(k)
            
    def similarity(self):
        """
        Compute the similarity between ROIs
        
        This method computes the similarity matrix for the set of ROIs. The
        similarity is defined as 1 if the two ROIs can potentially be imaged
        in a way that would require the SAR to change orientation (slew) in
        an unrealisable amount of time. Otherwise, the similarity is 0.

        Returns
        -------
        sim : TYPE
            DESCRIPTION.

        """
        sim = np.zeros((self.nFeatures, self.nFeatures))
        for k in range(self.nFeatures):
            for l in range(k+1, self.nFeatures):
                sim[k,l] = int(not self.rois[k].consistentWith(self.rois[l]))
                sim[l,k] = sim[k,l]
            sim[k,k] = 1
        return sim
    
    def cluster(self):
        """
        Compute clusters of ROIs that use the same SAR resources
        
        This function clusters ROIs according to their use of SAR as a 
        resource. That is, for instance, if the SAR cannot change orientation 
        in time to suit a the potential imaging geometry requirements for a 
        pair different ROIs, then both ROIs compete for the SAR as a resource. 
        The ROIs must either avoid using the potentially inconsistent geometry
        requirements, or if this is not possible, one ROI must use the SAR at 
        the expense of the other. 

        Returns
        -------
        None.

        """
        sim = self.similarity()

        model = AgglomerativeClustering(
          metric='precomputed',
          linkage='complete',
          n_clusters = None,
          distance_threshold=0.5
        ).fit(1-sim)

        labels = model.labels_
        clst = []
        for k in labels:
            if k not in clst:
                clst.append(k)
        self.clusters = [[k for k,l in enumerate(labels) if l==m] 
                         for m in clst]
        self.fltClusters = list(filter(lambda x: len(x) > 1, self.clusters))
        
    def computeElement(self, k):
        """
        Compute the roiPlan for roi at index k

        This method computes the roiPlan for roi indexed by k. The roiPlan is
        appended to a list of roiPlans for each roi so that they can further
        be mutually compared.
        
        Parameters
        ----------
        k : int
            The index for the roi to analyze.

        Returns
        -------
        None.

        """
        pE = roiPlan(self.plan["features"][k], 
                     self.filepath,
                     self.sarorbits,
                     UTC_filter = self.UTC_filter[k])
        pE.computeOptions()
        self.rois[k] = pE
    
    def writePlan(self, filename):
        filepath = self.filepath
        version = self.version
        self.computeNewJson()
        plan = self.plan
        with open(os.path.join(filepath, version, filename), "w") as f:
            f.write(json.dumps(plan, indent=2))

#%% Instantiate a plan object
eplan = plan(filepath)
eplan.computeElements()

#%%
sim = eplan.cluster()

#%%
def fnConsistent(idxs):
    """
    Function to determine imaging consistency.
    
    This function tests whether the filtered repeat and stereo imaging 
    options for each ROI in a set are consistent with each other. That is, 
    that they do not place unrealisable demands on the SAR to switch 
    incidence angles from one ROI to another.

    Parameters
    ----------
    idxs : list of int
        indexes of rois to test.

    Returns
    -------
    bool
        True if the rois are consistent, False otherwise.

    """
    x = [[eplan.rois[k].fltRepeat,
          eplan.rois[k].fltStereo] for k in idxs]
    x = [u for v in x for u in v if u.solutions]      
    y = distributeSolutions(x)
    
    return len(y) > 0
        
noncomp = list(filter(lambda x: not fnConsistent(x), eplan.fltClusters))
        
"""
Note to self here. Some options are showing conflict because the
satellite is asked to change incidence angle by 5 degrees in 20
minutes when the default that I've use is 1 degree in 10 minutes.
Some refinement of this condition seems important
"""

#%%
def bestStereo(idxs, fltFn = roiPlan.fnStereo):
    """
    Find the best options for non-consistent ROIs

    Parameters
    ----------
    idxs : `list` of int
        Indexes of the ROIs that are inconsistent.
    fltFn : `function`, optional
        A function used to filter the solutions. The default is 
        roiPlan.fnStereo and seeks to find the best stereo solution.

    Returns
    -------
    int
        The number of instances of the type selected by fltFn.
    list of imageSolutions. None if there are no consistent solutions.
        Consistent imagingSolutions with the maximum of the type slected
        by fltFn. None if there are no consistent solutions.

    """
    x = [[eplan.rois[k].fltRepeat,
          eplan.rois[k].fltStereo] for k in idxs]
    x = [u for v in x for u in v if u.solutions]
    N = len(x)
    for r in range(N,0,-1):
        y = [distributeSolutions(z) for z in combinations(x,r)]
        y = list(filter(lambda x: len(x) > 1, y))
        y = [x for v in y for x in v]
        mx = [x.filt(fltFn).length() for x in y]
        if len(mx) > 0:
            maxStereo = max(mx)
            return maxStereo, [z for k,z in enumerate(y) if mx[k]==maxStereo] 
    return None, None
            
