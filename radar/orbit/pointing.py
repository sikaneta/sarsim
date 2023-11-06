# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:30:18 2022

@author: ishuwa.sikaneta
"""
import numpy as np
from space.planets import earth
from orbit.orientation import orbit
from orbit.euler import rpyAnglesFromIJK
from orbit.euler import aeuFromRotation
from orbit.euler import RPYfromAEU, AEUfromRPY
from orbit.euler import aeu2rot
from orbit.geometry import getTiming
from measurement.measurement import state_vector
from numpy.random import default_rng
import json
from scipy.constants import c
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

rng = default_rng()


class simulation:
    """
    Class to allow computation satellite pointing errors
    
    This class allows simulation of pointing errors according to a set of 
    input pointing contributors as given in 
    `Reference Systems <../_static/EnvisionReferenceFramesAndPointingAngleDefinitions.pdf>`_
    and
    `Pointing Justification <../_static/PointingRequirementsJustification.pdf>`_
    
    Methods
    -------
    estimatePDF
        Estimate a probability distribution function from data using histogram
        approach
    generateGaussian
        Generate zero-mean joint Guassian distributed random variables. The
        joint variables have the supplied covariance matrix
    state
        This method rotates the inertial coordinate system used by AOCS folks
        into the VCI coordinate system definded in `Reference Systems`_.
    velocity2aeuCovariance
        This method takes a covariance matrix for velocity error (assumed to
        be zero-mean Gaussian) and computes the associated covariance matrix 
        for azimuth, elevation and tilt errors. See section 6.3 of
        `Pointing Justification`_
    timing2aeuCovariance
        This method takes a variance for timing error (assumed to be zero-mean
        Gaussian) and computes the associated covariance matrix 
        for azimuth, elevation and tilt errors. See section 6.2 of
        `Pointing Justification`_
    rpy2aeuCovariance
        This method takes a covariance matrix for roll, pitch and yaw errors,
        (assumed to be zero-mean Gaussian) and computes the associated 
        covariance matrix of azimuth, elevation and tilt as defined in section
        6.1 of `Pointing Justification`_
    simulateError
        This is the main method of the class. Given statistical parameters for
        the pointing error contributors, (in terms of covariances), it
        combines these parameters into a single covariance matrix for azimuth,
        elevation and pitch, as outlined in Section 6 of 
        `Pointing Justification`_. This covariance matrix is then used to
        generate random samples of the azimuth, elevation and tilt errors, 
        and errors are the transformed into realizations of both the Doppler 
        centroid and swath overlap values. These realizations can be grouped
        into histograms and compared with requirements.
    
    
    """
    def __init__(self, 
                 planet = earth(),
                 e_ang = 14.28,
                 azAxis = 6.0,
                 elAxis = 0.6,
                 carrier = 3.15e9,
                 mode = np.eye(3)):
        """
        Constructor

        Parameters
        ----------
        planet : `space.planet`, optional
            A class to describe the planet for the orbit. The default is 
            earth().
        e_ang : `float`, optional, units (degrees)
            The angle between boresight and the i-unit vector of the satellite 
            coordinate system, (see Figure 6. of `Reference Systems`_). 
            The default is 14.28.
        azAxis : `float`, optional, units (meters)
            Dimension of the SAR antenna azimuth axis. The default is 6.0.
        elAxis : `float`, optional, units (meters)
            Dimension of the SAR antenna elevation axis. The default is 0.6.
        carrier : `float`, optional
            Carrier frequency of the SAR signal. The default is 3.15e9.
        mode: `numpy.array`, [3,3]
            A matrix allowing a rotation of the a, e and u axis. The default is 
            the identity (for the VenSAR mode), but could be defined to switch the
            a and e axis to put the system in to altimeter mode.

        Returns
        -------
        None.

        """
        self.planet = planet
        self.e_ang = np.radians(e_ang)
        self.azAxis = azAxis
        self.elAxis = elAxis
        self.carrier = carrier
        self.mode = mode
        
    # Define a function to estimate the PDF given a histogram
    def estimatePDF(self, d, N=400):
        """
        Estimate a probability density function (PDF) from a set of data.

        Parameters
        ----------
        d : `np.ndarray(M,)`
            The data from which a PDF is to be estimated.
        N : int, optional
            The number of bins to use in the histogram. This values can also
            be gven as an `np.ndarray(N,1)` specifying histogram bin
            centers. The default is 400.

        Returns
        -------
        h : `np.ndarray(N-1,1)`
            The estimated PDF.
        x : `np.ndarray(N-1,1)`
            The dependent variable of the PDF.

        """
        h,x = np.histogram(d,N)
        x = (x[:-1] + x[1:])/2
        h = h/(np.mean(np.diff(x))*len(d))
        return h,x
    
    #
    def generateGaussian(self, R, N = 10000):
        """
        Generate realizations of a zero-mean joint Gaussian process.
        
        This function uses a Cholesky decomposition to generate realizations
        of a joint Gaussian process given the desired covariance matrix. The
        values are assumed to be real.

        Parameters
        ----------
        R : `np.ndarray(M,M)`
            Covariance matrix.
        N : int, optional
            Number of sample vectors to generate. The default is 10000.

        Returns
        -------
        `np.ndarray(M, N)`
            An MxN array of generated random vectors.

        """
        m,n = R.shape
            
        # """ Perform Cholesky decomposition """
        # try:
        #     cD = np.linalg.cholesky(R)
        # except LinAlgError:
        #     cD = np.zeros_like(R)
        """ Perform an eigenvale/eigenvector decomposition """
        D, U = np.linalg.eig(R)
        cD = U.dot(np.diag(np.sqrt(np.abs(D))))
        
        """ Generate the random data """
        return cD.dot(np.random.randn(m,N))
    
    # Define a function that will return state vectors in VCI coordinates
    def state(self, svs, idx):
        """
        Transform state vectors to VCI reference frame.
        
        State vectors provided are in a coordinate system defined
        by AOCS folks. The VCI coordinate system, as per `Reference Systems`_ 
        defines the j vector in the direction of the ascending node. Thus, we 
        need to find the components of the state vectors in this coordinate 
        system.

        Parameters
        ----------
        svs : `[np.ndarray(6,)]`
            A list of state vectors in some inertial coordinate system.
        idx : `[int, int, int]`
            A set of three integers to select range(idx) values from svs

        Returns
        -------
        `[np.ndarray(6,)]`
            A set of VCI state vectors as defined in `Reference Systems`_.

        """
        planetOrbit = orbit(planet=self.planet, angleUnits="radians")
        omega = np.array([planetOrbit.state2kepler(svs[k])["ascendingNode"] 
                          for k in range(idx[0], idx[1], idx[2])]).mean()
        co = np.cos(omega)
        so = np.sin(omega)
        Mo = np.array([[co, so, 0],[-so,co,0],[0,0,1]])
        Moo = np.block([[Mo, np.zeros_like(Mo)], [np.zeros_like(Mo), Mo]])
        
        return [svs[k].dot(Moo) for k in range(idx[0], idx[1], idx[2])]


    # angular error from velocity error
    def velocity2aeuCovariance(self,
                               X, 
                               off_nadir,
                               R_v = np.eye(3)*0.04,
                               N = 10000):
        """
        Transform satellite velocity error into AEU covariance
        
        This method transforms satellite velocity error, which is assumed to
        be Gaussian with zero-mean with covariance matrix R_v into an 
        estimate of :math:`\mathbf{R}_v(t)` as defined in 
        `Pointing Justification`_.
        
        In the calculation, the AAEU is calculated from the input state vector
        X. It is assumed that on-board, the computational algorithm has no
        knowledge of any error in the satellite velocity; thus the actual
        AAEU frame is calculated from the predicated state vector only. Random
        errors in the velocity are then generated and added to the input 
        satellite velocity to reflect what the satellite velocity really is, 
        and from this value the DAEU frame is computed. The difference between
        these two frames is then used to generate samples of AEU, and a
        covariance matrix from the generated values is estimated and returned.

        Parameters
        ----------
        X : `np.ndarray(6,)`
            A state vector in the VCI reference frame, around which to .
        off_nadir : `float`
            The off-nadir angle of the desired look-direction. This angle is
            defined as a right-handed rotation around the velocity vector; 
            thus, a positive angle corresponds to left-looking while a negative
            angle corresponds to right-looking.
        R_v : `np.ndarray(3,3)`, optional
            Covariance matrix of the satellite velocity error. The default is 
            np.eye(3)*0.04.
        N : `int`, optional
            The number of random samples to generate for the covariance matrix
            estimation. The default is 10000.

        Returns
        -------
        `np.ndarray(3,3)`
            An AEU covariance matrix estimate.

        """
        
        """ Create an orbit around Venus """
        planetOrbit = orbit(planet=self.planet, angleUnits="degrees")
        
        """ Generate random velocities """
        SVe = np.vstack((np.zeros((3,N)),
                         self.generateGaussian(R_v,N)))
        
        """ Compute the reference aueIe frame that is in error 
            from wrong velocity """
        orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
        aaeu, _ = planetOrbit.computeAEU(orbitAngle, off_nadir, self.mode)
        
        """ Compute the desired aeuI frame for each true velocity """
        daeu = np.zeros((3,3,N), dtype = np.double)
        for k in range(N):
            (orbitAngle, 
             ascendingNode) = planetOrbit.setFromStateVector(X + SVe[:,k])
            daeu[:,:,k], _ = planetOrbit.computeAEU(orbitAngle, off_nadir, self.mode)
            
        """ Compute the tilt, elevation and azimuth angles """
        AEU = aeuFromRotation(daeu.T.dot(aaeu))
        
        """ Return computed standard deviations """
        return AEU.dot(AEU.T)/N


    #%% Timing error
    def timing2aeuCovariance(self, X, off_nadir, Rt=5**2, N=10000):
        """
        Transform along-track timing error into AEU covariance
        
        This method transforms along-track timing error, which is assumed to
        be Gaussian with zero-mean with variance matrix Rt into an 
        estimate of :math:`\mathbf{R}_t(t)` as defined in 
        `Pointing Justification`_.
        
        In the calculation, the DAEU is calculated from the input state vector
        X. It is assumed that on-board, the computational algorithm has no
        knowledge of any error in timing; thus the actual AAEU frame is 
        calculated at the perturbed time. Random times are generated and 
        the state vector at these perturbed times is computed. The AAEU frame
        is then computed from these perturbed state vectors. The difference 
        between DAEU and AAEU is then used to generate samples of the AEU 
        angles, and a covariance matrix from the generated values is estimated 
        and returned. See, section 6.2 of `Pointing Justification`_.

        Parameters
        ----------
        X : `np.ndarray(6,)`
            A state vector in the VCI reference frame.
        off_nadir : `float`
            The off-nadir angle of the desired look-direction. This angle is
            defined as a right-handed rotation around the velocity vector; 
            thus, a positive angle corresponds to left-looking while a negative
            angle corresponds to right-looking.
        Rt : `float`, optional
            Variance of the along-track timing error. The default is 
            25.
        N : `int`, optional
            The number of random samples to generate for the covariance matrix
            estimation. The default is 10000.

        Returns
        -------
        `np.ndarray(3,3)`
            An AEU covariance matrix estimate.

        """
        
        """ Create an orbit around Venus """
        planetOrbit = orbit(planet=self.planet, angleUnits="degrees")
        
        """ Compute the orbit angle for the given state vector"""
        orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
        
        """ Compute constant relating derivative of orbit angle to derivative 
            of time """
        e = planetOrbit.e
        GM = planetOrbit.planet.GM
        a = planetOrbit.a
        U = np.radians(orbitAngle) - planetOrbit.arg_perigee
        C = (1+e*np.cos(U))**2*np.sqrt(GM/
                             (a*(1-e**2))**3)
        
        """ Generate random orbit angle deviances """
        dO = rng.standard_normal(N)*np.sqrt(Rt)*np.degrees(C)
        
        """ Calculate the DAEU frame """
        daeu, _ = planetOrbit.computeAEU(orbitAngle, off_nadir, self.mode)
        
        """ Loop over deviations in orbit angle """
        aaeu = np.zeros((3,3,N), dtype = np.double)
        for k in range(N):
            aaeu[:,:,k], _ = planetOrbit.computeAEU(orbitAngle + dO[k], 
                                                    off_nadir, self.mode)
            
        """ Compute the tilt, elevation and azimuth angles. The moveaxis
            function ensures that the input is (N,3,3) """
        AEU = aeuFromRotation(np.moveaxis((aaeu.T.dot(daeu)).T, -1, 0))
        
        """ Return computed standard deviations """
        return AEU.dot(AEU.T)/N


    #%%
    def aeu2rpyCovariance(self, R_AEU, N=100000):
        """
        Estimate roll, pitch, yaw covariance from azimuth, elevation tilt
        
        By simulation, estimate the covariance matrix for roll, pitch
        and yaw that corresponds to the input covariance matrix for azimuth,
        elevation and tilt errors

        Parameters
        ----------
        R_AEU : `np.ndarray(3,3, dtype=double)`
            The input covariance matrix of azimuth, elevation and tilt.
        N : `int`
            The number of samples to use in the simulatino. The default 
            is 100000.

        Returns
        -------
        `np.ndarray(3,3)`
            The estimated covariance matrix of roll, pitch and yaw angle
            errors.

        """
        
        """ Compute the rotation matrix to go from aeu to ijk_s """
        c_ep = np.cos(self.e_ang)
        s_ep = np.sin(self.e_ang)
        M_er = np.array([
                        [1,     0,    0],
                        [0,  c_ep, s_ep],
                        [0, -s_ep, c_ep]
                       ])
        
        """ Generate N 3-element vectors with covariance matrix given by
            R_AEU """
        AEU = self.generateGaussian(R_AEU, N)
        
        """ Initialize a data set for the transformation of each of the
            vectors generated above from AEU to RPY """
        RPY = np.zeros_like(AEU)
        
        """ For each triplet of AEU (azimuth, elevationand tilt) error 
            angles, compute the equivalent roll, pitch and yaw angles
            that would rotate the spacecraft from nominal pointing, so 
            that it would cause mispointing in accordance with the AEU 
            values. """
        RPYfromAEU(AEU, M_er, RPY)
        
        """ Estimate the covariance matrix of the RPY generated triplets
            and return """
        return RPY.dot(RPY.T)/N

    #%%
    def rpy2aeuCovariance(self, R_RPY, N=100000):
        """
        Estimate the AEU covariance matrix from a RPY covariance matrix.
        
        This function computes an estimate of covariance in AEU (azimuth, 
        elevation, tilt) given covariance in RPY (roll, pitch, yaw). The
        random errors in roll, pitch and yaw are assumed to be zero-mean
        Gaussian.
        
        The transformation used to make the estimation is defined in Sections 
        5.2 and 7.6 of `Reference Systems`_.

        Parameters
        ----------
        R_RPY : `np.ndarray(3,3)`
            Covariance matrix of roll, pitch, yaw error. 
        N : `int`, optional
            Number of random samples to generate to estimate covariance in AEU. 
            The default is 100000.

        Returns
        ------- 
        `np.ndarray(3,3)`
            An AEU covariance matrix estimate.

        """
        
        """ Compute the rotation matrix to go from aeu to ijk_s """
        c_ep = np.cos(self.e_ang)
        s_ep = np.sin(self.e_ang)
        M_er = np.array([
                        [1,     0,    0],
                        [0,  c_ep, s_ep],
                        [0, -s_ep, c_ep]
                       ])
        
        RPY = self.generateGaussian(R_RPY, N)
        AEU = np.zeros_like(RPY)
        AEUfromRPY(RPY, M_er, AEU)
        return AEU.dot(AEU.T)/N

    #%%
    def xtrackOffset2aeuCovariance(self,
                                   X,
                                   off_nadir,
                                   R_xtrack):
        """
        Translate x-track position errors to elevation pointing error
        
        This function translates across-track satellite position errors into
        elevation angle pointing errors. The procedure follows the old
        approach of defining the elevation error as the error between the
        actual incidence angle and the predicted incidence angle. The analysis
        was captured in an old version of the PRJ document.

        Parameters
        ----------
        X : `np.ndarray(6, dtype=float)`
            State vector of the satellite at the time of interest.
        off_nadir : `float`
            The off-nadir angle for the elevation boresight.
        R_xtrack : `np.ndarray(2,2, dtype=float)`
            The covariance matrix of the across-track error in terms of
            errors in the c and n-directions, respecvely.

        Returns
        -------
        `float`
            Variance of the elevation angle error.

        """
        sv = state_vector(planet = self.planet)
        sv.add(np.datetime64("2000-01-01T00:00:00.000000"), sv.toPCR(X, 0))
        
        tcn = self.tcnFromX(X)
        
        r, rhat, inc, _, _ = getTiming(sv, [np.radians(off_nadir)], 0)
        
        q = rhat[0].dot(tcn[:,1:])#.dot(np.array([[0,1],[1,0]]))
        
        return q.dot(R_xtrack).dot(q)/r[0]**2

    #%%
    def xtrackOffset2aeuCovarianceTCN(self,
                                      X,
                                      off_nadir,
                                      R_xtrack):
        """
        Translate x-track position errors to elevation pointing error
        
        This function translates across-track satellite position errors into
        elevation angle pointing errors. The procedure follows the definition
        of pointing error that has nothing to do with the measured imagery.

        Parameters
        ----------
        X : `np.ndarray(6, dtype=float)`
            State vector of the satellite at the time of interest.
        R_xtrack : `np.ndarray(2,2, dtype=float)`
            The covariance matrix of the across-track error in terms of
            errors in the c and n-directions, respecvely.

        Returns
        -------
        `float`
            Variance of the elevation angle error.

        """
        
        q = np.array([1,0])
        
        return q.T.dot(R_xtrack).dot(q)/np.linalg.norm(X[:3])**2
    
    #%%
    def xtrackOffset2aeuCovarianceSwath(self,
                                        X,
                                        off_nadir,
                                        R_xtrack):
        """
        Translate x-track position errors to elevation pointing error
        
        This function translates across-track satellite position errors into
        elevation angle pointing errors. The procedure follows the equation
        defined in section 6.5 of the PRJ

        Parameters
        ----------
        X : `np.ndarray(6, dtype=float)`
            State vector of the satellite at the time of interest.
        off_nadir : `float`
            The off-nadir angle for the elevation boresight.
        R_xtrack : `np.ndarray(2,2, dtype=float)`
            The covariance matrix of the across-track error in terms of
            errors in the c and n-directions, respecvely.

        Returns
        -------
        `float`
            Variance of the elevation angle error.

        """
        sv = state_vector(planet = self.planet)
        sv.add(np.datetime64("2000-01-01T00:00:00.000000"), sv.toPCR(X, 0))
        
        tcn = self.tcnFromX(X)
        
        r, rhat, inc, _, _ = getTiming(sv, [np.radians(off_nadir)], 0)
        
        d = np.cross(rhat[0], tcn[:,0]) - rhat[0]/np.tan(np.radians(inc[0]))
        
        q = d.dot(tcn[:,1:])#np.array([d.dot(c), d.dot(n)])
        
        return q.dot(R_xtrack).dot(q)/r[0]**2

    
    #%%
    def tcnFromX(self, X):
        """
        Generate tcn frame from state vector
        
        Generates the tcn vector from the state vector which is given in some
        reference frame. This tcn frame has a t vector in the direction of
        the velocity vector, a c vector aligned with the orbit angular
        velocity vector and an n vector completing the right-handed coordinate
        system

        Parameters
        ----------
        X : `np.ndarray(6,)`
            A given state vector in some reference frame.

        Returns
        -------
        tcn : `np.ndarray(3,3)`
            A matrix corresponding to the tcn frame with column vectors, in
            order, given by the t, c and n vectors

        """
        
        t = X[3:]/np.linalg.norm(X[3:])
        n = -X[:3]/np.linalg.norm(X[:3])
        c = np.cross(n, t)
        c /= np.linalg.norm(c)
        n = np.cross(t,c)
        n /= np.linalg.norm(n)
        
        return np.array([t,c,n]).T
        
    #%%
    def contributors2aeuCovariance(self,
                                   X,
                                   off_nadir, 
                                   covariances,
                                   N = 100000):
        """
        This function combines source errors into an aeu covariance 
        matrix. 
        
        This function generates the large block-diagonal matrix of Equation 
        (9) in `Pointing Justification`_. It uses the defined member functions
        to generate the block matrix.
        
        
    
        Parameters
        ----------
        X : `np.ndarray(6,)`
            A state vector in the VCI reference frame.
        r : `float`
            The range to the center of the swath of interest.
        off_nadir : `float`
            The off-nadir angle of the desired look-direction. This angle is
            defined as a right-handed rotation around the velocity vector; 
            thus, a positive angle corresponds to left-looking while a negative
            angle corresponds to right-looking.
        R_RPY : `np.ndarray(3,3)`
            Covariance matrix of roll, pitch, yaw error.
        R_v : `np.ndarray(3,3)`, optional
            Covariance matrix of the satellite velocity error. The default is 
            np.eye(3)*0.04.
        R_t : `float`, optional
            Variance of the along-track timing error.
        R_p : `float`, optional
            Variance of the orbit tube in cross-lokk direction. The default is 
            430x430.
        N : `int`, optional
            Number of random samples to generate in estimation process.
    
        Returns
        -------
        `np.ndarray(10,10)`
            An AEU block covariance matrix estimate.
        `np.ndarray(3,10)`
            The transformation matrix to recover azimuth, elevation and tilt
            angle errors.
    
        """
        
        R_att = self.rpy2aeuCovariance(np.array(covariances["spacecraft"]["R"]), N)
        
        """ Check if we need to change frames """
        R_orbvel = np.array(covariances["orbitVelocity"]["R"])
        if covariances["orbitVelocity"]["referenceVariables"] == "VtVcVn":
            M_tcn = self.tcnFromX(X)
            R_orbvel = M_tcn.dot(R_orbvel).dot(M_tcn.T)
        R_vel = self.velocity2aeuCovariance(X, 
                                            off_nadir, 
                                            R_orbvel)
        
        """ Check if we need to scale to time """
        R_orbalong = np.array(covariances["orbitAlongTrack"]["R"])
        if covariances["orbitAlongTrack"]["units"] == "m":
            M_s = 1/np.linalg.norm(X[3:])
            R_orbalong = M_s*R_orbalong*M_s  
        R_tme = self.timing2aeuCovariance(X, 
                                          off_nadir, 
                                          R_orbalong)
        
        """ Select an elevation pointing angle error definition """
        AcrossTrackFn = {"TCN": self.xtrackOffset2aeuCovarianceTCN,
                         "SwathCentre": self.xtrackOffset2aeuCovarianceSwath,
                         "BeamIntersection": self.xtrackOffset2aeuCovariance}
        try:
            errDef = AcrossTrackFn[covariances["orbitAcrossTrack"]["errorDefinition"]]
        except KeyError:
            errDef = AcrossTrackFn["SwathCentre"]
            
        R_pos = errDef(X, 
                       off_nadir, 
                       np.array(covariances["orbitAcrossTrack"]["R"]))
        
        """ Get the instrument contribution and convert accordingly. """
        R_ins = covariances["instrument"]["R"]
        if covariances["instrument"]["referenceVariables"] == "RollPitchYaw":
            R_ins = self.rpy2aeuCovariance(np.array(R_ins), N)
        
        R = np.array(block_diag(R_att, R_tme, R_vel, R_pos, R_ins))
        AEU_m = np.array([[1,0,0,1,0,0,1,0,0,0,1,0,0],
                          [0,1,0,0,1,0,0,1,0,1,0,1,0],
                          [0,0,1,0,0,1,0,0,1,0,0,0,1]])
        return R, AEU_m
    
    #%%
    def simulateError(self,
                      X, 
                      off_nadir,
                      covariances,
                      n_AEU = 1000000,
                      dopErrThreshold = 1/15,
                      eleErrThreshold = 1/15,
                      loglevel = 0
                      ):
        """
        Run a simulation to generate Doppler centroid and swath overlap errors
        
        

        Parameters
        ----------
        X : `np.ndarray(6,)`
            A state vector in the VCI reference frame.
        off_nadir : `float`
            The off-nadir angle of the desired look-direction. This angle is
            defined as a right-handed rotation around the velocity vector; 
            thus, a positive angle corresponds to left-looking while a negative
            angle corresponds to right-looking. Degrees units.
        R_RPY : `np.ndarray(3,3)`
            Covariance matrix of roll, pitch, yaw error.
        R_v : `np.ndarray(3,3)`, optional
            Covariance matrix of the satellite velocity error. The default is 
            np.eye(3)*0.04.
        R_t : `float`, optional
            Variance of the along-track timing error.
        R_p : `float`, optional
            Variance of the orbit tube in cross-look direction. The default is 
            430x430.
        n_AEU : `int`, optional
            Number of AEU sample vectors to generate in the estimation process. 
            The default is 1000000.
        loglevel : `int`, optional
            A variable to drive the logging level. The default is 0. With 0, 
            very little logging is produced.
            
            1. Print the computed dictionary to stdout.
            2. Generate matplotlib plots and print the computed dictionary to
               stdout.
            3. Print intermediate results to stdout, generate matplotlib
               figures and print the computed dictionary to stdout.

        Returns
        -------
        res : `dict`
            A dictionary containing given and computed simulation parameters.

        """
        
        """ Define a return dictionary """
        res = {"given": {"off_nadir": off_nadir,
                         "azAxis": self.azAxis,
                         "elAxis": self.elAxis,
                         "carrier": self.carrier,
                         "covariances": covariances
                         }
               }
        
        wavelength = c/self.carrier
        elBW = wavelength/self.elAxis
        azBW = wavelength/self.azAxis
        
        res["computed"] = {"wavelength": wavelength,
                           "beamwidths": {"units": "degrees",
                                          "azimuth": np.degrees(azBW),
                                          "elevation": np.degrees(elBW)}
                           }
        
        """ Create an orbit around Venus """
        planetOrbit = orbit(planet=self.planet, angleUnits="degrees")
        orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
        
        """ Define the off-nadir angle """
        v = np.cos(np.radians(off_nadir))
        
        XI, rI, vI = planetOrbit.computeSV(orbitAngle)
        res["State Vector Radius"] = np.linalg.norm(X[:3])
        res["Kepler"] = {"radius": np.linalg.norm(rI),
                         "a": planetOrbit.a,
                         "e": planetOrbit.e,
                         "period": planetOrbit.period,
                         "altitude": np.linalg.norm(rI) - planetOrbit.planet.a,
                         "angles": 
                         {"units": "degrees",
                          "orbit": orbitAngle,
                          "ascending node": ascendingNode,
                          "inclination": np.degrees(planetOrbit.inclination),
                          "perigee": np.degrees(planetOrbit.arg_perigee)
                         }
                         }
        e1, e2 = planetOrbit.computeE(orbitAngle, v)
    
        aeuI, aeuTCN = planetOrbit.computeAEU(orbitAngle, off_nadir, self.mode)
        tcnI = aeuI.dot(aeuTCN.T)
        
        """ Compute the rotation matrix to go from aeu to ijk_s """
        c_ep = np.cos(self.e_ang)
        s_ep = np.sin(self.e_ang)
        M_e = np.array([
                        [0,     1,     0],
                        [s_ep,  0,  c_ep],
                        [c_ep,  0, -s_ep]
                       ])
        ijk_s = aeuI.dot(M_e) 
        qq = aeuTCN.dot(M_e)
    
        """ Compute the Euler angles """
        (theta_r, 
         theta_p, 
         theta_y) = rpyAnglesFromIJK(qq.reshape((1,3,3))).flatten()
    
        M, dM = planetOrbit.computeItoR(orbitAngle)
        XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
        vP = np.linalg.norm(XP[3:])
        dopBW = 2*vP/wavelength*azBW
        aeuP = M.dot(aeuI)
        vAngles = np.arange(-np.degrees(elBW/2), np.degrees(elBW/2), 0.01)
    
        if loglevel > 2:
            print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
            print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
            print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
            print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))
            prystring = "Roll: %0.4f, Pitch: %0.4f, Yaw: %0.4f"
            print(prystring % (np.degrees(theta_r),
                               np.degrees(theta_p),
                               np.degrees(theta_y)))
            
        res["attitude"] = {"ijk_s": ijk_s.tolist(),
                           "tcnI": tcnI.tolist(),
                           "aeuI": aeuI.tolist(),
                           "aeuP": aeuP.tolist(),
                           "PRY": {"units": "degrees",
                                   "roll": np.degrees(theta_r),
                                   "pitch": np.degrees(theta_p),
                                   "yaw": np.degrees(theta_y)}
                           }
    
        """ Compute the time """
        t = planetOrbit.computeT(orbitAngle)
        res["Kepler"]["time"] = t
            
        """ Instantiate a state vector object """
        sv = state_vector(planet=self.planet, harmonicsCoeff=180)
        
        """ Compute ECEF state vector """ 
        XP = sv.toPCR(XI, t)
        
        """ Compute the ground position """
        XG = sv.computeGroundPositionU(XP, aeuP[:,2])
        
        """ Test the ground position """
        R = XG - XP[:3]
        rhat = R/np.linalg.norm(R)
        vG = -XP[:3].dot(rhat)/np.linalg.norm(XP[:3])
        pG = ((XG[0]/planetOrbit.planet.a)**2 
              + (XG[1]/planetOrbit.planet.a)**2 
              + (XG[2]/planetOrbit.planet.b)**2)
        dG = XP[3:].dot(rhat)
        res["VCR"] = {"X": list(XG),
                      "cosine off-nadir": vG,
                      "Doppler Centroid": dG,
                      "EllipsoidFn": pG}
        
        """ Generate random errors """ 
        R, AEU_m = self.contributors2aeuCovariance(X,
                                                   off_nadir, 
                                                   covariances)
        
        R_AEU = AEU_m.dot(R).dot(AEU_m.T)
        aeu_e = self.generateGaussian(R_AEU, n_AEU)
        aeu_m = np.zeros((3,3,n_AEU), dtype=aeu_e.dtype)
        aeu2rot(np.eye(3), aeu_e, aeu_m)
        aeuPe = np.einsum("ij,jkl->ikl", aeuP, aeu_m)
                
        """ Make a record of the AEU error covariances """
        res["computed"]["ErrorCovariance"] = {"units": "rad^2",
                                              "BlockCovariance": R.tolist(),
                                              "AEUCovariance": R_AEU.tolist()}
        
        """ Get the rotation matrices to steer boresight to edges of swath """
        eang = np.array([[0]*3 
                         + [vAngles[0], 0.0, vAngles[-1]] 
                         + [0]*3]).reshape((3,3))
        erot = np.zeros((3,3,3), dtype = eang.dtype)
        aeu2rot(np.eye(3), np.radians(eang), erot)
        
        """ Calculate the zeroDoppler values """
        zDop = 2*np.einsum("i,ijk,jmn->kmn", 
                           XP[3:], 
                           aeuPe, 
                           erot)[:,2,:]/wavelength
        dc_max = zDop[range(n_AEU), np.argmax(np.abs(zDop), axis=1)]
    
        """ Plot the histogram of the maximum Doppler centroids across the 
            beam. """
        hD, xD = self.estimatePDF(dc_max, N=200)
        dError = 100*np.sum(np.abs(dc_max) > dopErrThreshold*dopBW)/n_AEU
        if loglevel > 2:
            print("Percent of Doppler centroids in violation: %0.4f" % dError)
            
        if loglevel > 1:
            plt.figure()
            plt.plot(xD, hD)
            plt.axvline(x = dopErrThreshold*dopBW, 
                        color = 'r', 
                        label = '%0.2f of Doppler Bandwdith' % (dopErrThreshold/2))
            plt.axvline(x = -dopErrThreshold*dopBW, 
                        color = 'r', 
                        label = '-%0.2f of Doppler Bandwdith' % (dopErrThreshold/2))
            plt.xlabel('Doppler centroid (Hz)')
            plt.ylabel('Histogram count')
            mytitle = r"Error Rate %0.2e" % (dError/100.0)
            plt.title(mytitle)
            plt.grid()
            plt.show()
    
    
        
        """ Compute the reference min and max ranges """
        uProt = np.einsum("ij,jlm->ilm", aeuP, erot)[:,2,:]
        refR = sv.computeRangeVectorsU(XP, uProt.T)
        refRng = np.linalg.norm(refR, axis=1)
        
        uPerot = np.einsum("ijk,jmn->imnk", 
                           aeuPe, 
                           erot)[:,2,:,:].T.reshape((3*n_AEU,3))
        errR = sv.computeRangeVectorsU(XP, uPerot)
        errRng = np.linalg.norm(errR, axis=-1).reshape((n_AEU,3))
        
        """ define vectorized functions to compute min and max over array """
        minofmax = np.min(np.stack((np.max(errRng, axis=-1), 
                                    np.max(refRng)*np.ones(n_AEU))), axis=0)
        maxofmin = np.max(np.stack((np.min(errRng, axis=-1), 
                                    np.min(refRng)*np.ones(n_AEU))), axis=0)
        coverage = (minofmax-maxofmin)/(np.max(refRng) - np.min(refRng))
        
        hS, xS = self.estimatePDF(coverage, N=50)
        sError = 100*np.sum(coverage < (1-eleErrThreshold))/n_AEU
        if loglevel > 2:
            print("Percent of Swaths in violation: %0.4f" % sError)
            
        if loglevel > 1:
            plt.figure()
            plt.plot(xS, hS)
            plt.axvline(x = 1-eleErrThreshold, color = 'r', 
                        label = '%0.2f of beamwidth' % (1-eleErrThreshold))
            plt.xlabel('Elevation beam overlap on ground (fraction)')
            plt.ylabel('Histogram count')
            mytitle = r"Error Rate %0.2e" % (sError/100.0)
            plt.title(mytitle)
            plt.grid()
            plt.show()
            
        res["ErrorThreshold"] = {"Doppler": dopErrThreshold,
                                 "Swath": eleErrThreshold}
        res["ErrorRate"] = {"Doppler": dError,
                            "Swath": sError}
        
        del aeuPe
        
        if loglevel > 3:
            print(json.dumps(res, indent=2))
        
        return res
