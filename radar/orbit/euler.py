# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:50:20 2022

@author: Ishuwa.Sikaneta
"""
#%%
import numpy as np
from numba import njit, prange
     
@njit
def rpyFromRotation(R, RPY):
    """
    Compute roll and pitch and yaw from rotation matrix
    
    This function computes the Roll angle, Pitch angle
    and Yaw angle from a given rotation matrix. The
    algorithm for computing these values can be found
    in Section 5.2.1 of `Reference Systems`_, where the input
    RPY matrix is the left side of equation (13).
    
    Parameters
    ----------
    R : `numpy.ndarray, [3,3,N]`
        The product of R.dot(P).dot(Y) rotation matrices.
    RPY : `numpy.ndarray, [3, N]`
        Numoy array to hold computed roll pitch yaw vales.

    Returns
    -------
    None

    """
    
    N = len(R)
    
    for k in prange(N):
        """ Compute the yaw angle """
        RPY[2,k] = np.arctan2(-R[k,0,1], R[k,0,0])
        c_y = np.cos(RPY[2,k])
        s_y = np.sin(RPY[2,k])
        
        M_y = np.array([
                        [c_y,  s_y,  0.0],
                        [-s_y, c_y,  0.0],
                        [0.0,  0.0,  1.0]
                        ])
        
        Rp = R[k,:,:].dot(M_y)
        
        """ Compute the pitch angle """
        RPY[1,k] = np.arctan2(Rp[0,2], Rp[0,0])
        c_p = np.cos(RPY[1,k])
        s_p = np.sin(RPY[1,k])
        
        M_p = np.array([
                        [c_p,  0.0, -s_p],
                        [0.0,  1.0,  0.0],
                        [s_p,  0.0,  c_p]
                        ])
        
        Rp = Rp.dot(M_p)
        
        """ Compute the roll angle """
        RPY[0,k] = np.arctan2(-Rp[1,2], Rp[1,1])
             
@njit
def aeu2rot(old_basis, aeu_e, new_basis):
    """
    Rotate a set of basis vectors by aeu angles.
    
    Rotate a set of basis vectors according to eq (25) in `Reference Systems`_. 
    If old_basis is the identity matrix, then the return value is the 
    overall rotation matrix.

    Parameters
    ----------
    old_basis : `numpy.ndarray(3,3)`
        Basis vectors (columnwise) to be rotated.
    aeu_e : `numpy.ndarray(3,N)`
        A list of angles (radians) to be transformed into rotation matrices.
        each of the N columns is a vector of azimuth, elevation and tilt
        angles.
    new_basis : `numpy.ndarray(3,3,N)`
        The rotated bases vectors. If bases is the identity, the return
        value is the rotation matrix. This matrix is a pointer to the
        computed values.

    Returns
    -------
    None.

    """
    c_aeu = np.cos(aeu_e)
    s_aeu = np.sin(aeu_e)
    _,N = aeu_e.shape
    
    for k in prange(N):
        Me = np.array([[1.0, 0.0, 0.0],
                       [0.0, c_aeu[1,k], -s_aeu[1,k]],
                       [0.0, s_aeu[1,k],  c_aeu[1,k]]])

        Ma = np.array([[c_aeu[0,k], 0.0, s_aeu[0,k]],
                        [0.0, 1.0, 0.0],
                        [-s_aeu[0,k], 0.0, c_aeu[0,k]]])

        Mt = np.array([[c_aeu[2,k], -s_aeu[2,k], 0.0],
                       [s_aeu[2,k], c_aeu[2,k],  0.0],
                       [0.0, 0.0, 1.0]])
        new_basis[:,:,k] = old_basis.dot(Me).dot(Ma).dot(Mt)


@njit
def RPYfromAEU(AEU, M_e, RPY):
    """
    Compute Yaw, roll and pitch errors from azimuth, elevation and
    tilt errors
    
    This function computes the Roll angle, Pitch angle
    and Yaw angle errors from given azimuth, elevation and tile
    errors. The algorithm for computing these values can be found
    in my notes.
    
    The angles are computed according to equation (31) of 
    `Reference Systems`_.
    
    Parameters
    ----------
    AEU : `numpy.ndarray, [3, N]`
        The a, e, u values as a matrix of Nx3.
    M_e : `numpy.ndarray, [3,3]`
        The matrix that rotates aeu into jki_s.
    RPY : `numpy.ndarray, [3, N]`
        Numoy array to hold computed roll pitch yaw vales.

    Returns
    -------
    None

    """
    
    _, N = AEU.shape
    
    for k in prange(N):
        """ Compute the rotation matrix from a,e,u values """
        c_a = np.cos(AEU[0,k])
        s_a = np.sin(AEU[0,k])
        c_e = np.cos(AEU[1,k])
        s_e = np.sin(AEU[1,k])
        c_u = np.cos(AEU[2,k])
        s_u = np.sin(AEU[2,k])
        SdSa = (M_e.T
                .dot(np.array([[1.0,0.0,0.0],[0.0,c_e,-s_e],[0.0,s_e,c_e]]))
                .dot(np.array([[c_a,0.0,s_a],[0.0,1.0,0.0],[-s_a,0.0,c_a]]))
                .dot(np.array([[c_u,-s_u,0.0],[s_u,c_u,0.0],[0.0,0.0,1.0]]))
                .dot(M_e))
        
        """ Compute the yaw angle """
        RPY[2,k] = np.arctan2(-SdSa[0,1], SdSa[0,0])
        c_y = np.cos(RPY[2,k])
        s_y = np.sin(RPY[2,k])
        
        M_y = np.array([
                        [c_y,  s_y,  0.0],
                        [-s_y, c_y,  0.0],
                        [0.0,  0.0,  1.0]
                        ])
        
        Rp = SdSa.dot(M_y)
        
        """ Compute the pitch angle """
        RPY[1,k] = np.arctan2(Rp[0,2], Rp[0,0])
        c_p = np.cos(RPY[1,k])
        s_p = np.sin(RPY[1,k])
        
        M_p = np.array([
                        [c_p,  0.0, -s_p],
                        [0.0,  1.0,  0.0],
                        [s_p,  0.0,  c_p]
                        ])
        
        Rp = Rp.dot(M_p)
        
        """ Compute the yaw angle """
        RPY[0,k] = np.arctan2(-Rp[1,2], Rp[1,1])
        
@njit
def AEUfromRPY(RPY, M_e, AEU):
    """
    Compute azimuth, elevation and tilt angle errors from the
    roll, pitch and yaw angular errors.
    
    This function computes the azimuth, elevation and tilt angular erros
    from the roll, pitch and yaw angular errors. A reference for the
    algorithm may be found in the Envision reference frames and pointing
    angle error document.
    
    Specifically, equation (31) of `Reference Systems`_ is inverted to get
    the azimuth, elevation and tilt angle rotation matrix, then equation (25)
    is inverted according to the method of Section 5.2.1. 

    Note
    ----
    
    - epsilon -> roll
    - alpha -> pitch
    - tau -> yaw
    
    Parameters
    ----------
    RPY : `numpy.ndarray, [3, N]`
        The roll, pitch, yaw values as a matrix of 3xN.
    M_e : `numpy.ndarray, [3,3]`
        The matrix that rotates aeu into jki_s.
    AEU : `numpy.ndarray, [3, N]`
        Numoy array to hold computed azimuth, elevation and tilt vales.
        
    Returns
    -------
    None

    """
    
    """
    This is the same transform as RPYfromAEU, but with the inverse of M_e
    """
    
    _, N = RPY.shape
    
    for k in prange(N):
        """ Compute the rotation matrix from a,e,u values """
        c_r = np.cos(RPY[0,k])
        s_r = np.sin(RPY[0,k])
        c_p = np.cos(RPY[1,k])
        s_p = np.sin(RPY[1,k])
        c_y = np.cos(RPY[2,k])
        s_y = np.sin(RPY[2,k])
        SdSa = (M_e
                .dot(np.array([[1.0,0.0,0.0],[0.0,c_r,-s_r],[0.0,s_r,c_r]]))
                .dot(np.array([[c_p,0.0,s_p],[0.0,1.0,0.0],[-s_p,0.0,c_p]]))
                .dot(np.array([[c_y,-s_y,0.0],[s_y,c_y,0.0],[0.0,0.0,1.0]]))
                .dot(M_e.T))
        
        """ Compute the tau angle """
        AEU[2,k] = np.arctan2(-SdSa[0,1], SdSa[0,0])
        c_u = np.cos(AEU[2,k])
        s_u = np.sin(AEU[2,k])
        
        M_u = np.array([
                        [c_u,  s_u,  0.0],
                        [-s_u, c_u,  0.0],
                        [0.0,  0.0,  1.0]
                        ])
        
        Rp = SdSa.dot(M_u)
        
        """ Compute the alpha angle """
        AEU[0,k] = np.arctan2(Rp[0,2], Rp[0,0])
        c_a = np.cos(AEU[0,k])
        s_a = np.sin(AEU[0,k])
        
        M_a = np.array([
                        [c_a,  0.0, -s_a],
                        [0.0,  1.0,  0.0],
                        [s_a,  0.0,  c_a]
                        ])
        
        Rp = Rp.dot(M_a)
        
        """ Compute the epsilon angle """
        AEU[1,k] = np.arctan2(-Rp[1,2], Rp[1,1])

def aeuFromRotation(AEUmatrix):
    """
    Compute the azimuth, elevation and tilt angles from a, e, u basis vectors.
    
    Equation (25) of `Reference Systems`_ is inverted according to the method 
    of Section 5.2.1. 

    Parameters
    ----------
    AAEmatrix : `np.ndarray(N, 3,3)`
        N a,e,u, basis vectors from which to compute azimuth, elevation and 
        tilt angles. These are arranged in columns

    Returns
    -------
    `np.ndarray(3,N)`
        The computed azimuth, elevation and tilt angles for each basis vector
        set of the input.

    """
    
    eau = np.zeros((3, len(AEUmatrix)), dtype = np.double)
    rpyFromRotation(AEUmatrix, eau)
    
    """ Need to switch the order of values so that we get
        aeu order rather than eau """
    aeShift = np.array([[0,1,0],
                        [1,0,0],
                        [0,0,1]])
    
    return aeShift.dot(eau)

#%%
def rpyAnglesFromIJK(IJKmatrix):
    """
    Compute roll, pitch and yaw angles from satellite IJK basis vectors.

    Parameters
    ----------
    IJKmatrix : `np.ndarray(N, 3,3)`
        N i,j,k basis vectors from which to compute azimuth, elevation and 
        tilt angles. These are arranged in columns.

    Returns
    -------
    rpy : `np.ndarray(3,N)`
        The computed roll, pitch and yaw angles for each basis vector
        set of the input.

    """
    JKIshift = np.array([[0,0,1],[1,0,0],[0,1,0]])
    rpy = np.zeros((3, len(IJKmatrix)), dtype = np.double)
    rpyFromRotation(IJKmatrix.dot(JKIshift), rpy)
    return rpy
