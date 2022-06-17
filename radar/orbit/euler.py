# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:50:20 2022

@author: Ishuwa.Sikaneta
"""

import numpy as np
from numba import jit, njit, prange

@njit
def PRYfromRotation(R, R0, PRY):
    """
    Compute Pitch, roll and Yaw from rotation matrix
    
    This function computes the Pitch angle, Roll angle
    and Yaw angle from a given rotation matrix. The
    algorithm for computing these values can be found
    in my notes.
    
    The pitch, roll and yaw angles are given by \theta_p,
    \theta_r and \theta_y, respectively.
    
    The Pitch matrix is given by
    
    M_p = [cos\theta_p  sin\theta_p  0]
          [-sin\theta_p cos\theta_p  0]
          [0            0            1].
          
    
    The roll matrix is given by
    
    M_r = [cos\theta_r  0 -sin\theta_r]
          [0            1            0]
          [sin\theta_r  0  cos\theta_r].
    
    
    And the yaw matrix is given by
    
    M_y = [1            0            0]
          [0  cos\theta_y  sin\theta_y]
          [0  -sin\theta_y cos\theta_y].
          
    The **rotated** basis vectors i_1, j_1 and k_1 relate to the
    original basis vectors, i_0, j_0, k_0 through
    
    [i_1, j_1, k_1] = [i_0, j_0, k_0]M_p^TM_r^TM_y^T
    
    where [i_1, j_1, k_1] and [i_0, j_0, k_0] are 3x3 matrices with columns
    corresponding to the indicated basis vectors.
    
    Assuming that [i_0, j_0, k_0] is the identity matrix, this function
    takes input as R = [i_1, j_1, k_1] and computes M_p, M_r and M_y and
    the corresponding Euler angles.
    
    Parameters
    ----------
    R : `numpy.ndarray, [N,3,3]`
        The final rotation matrices.
    R0 : `numpy.ndarray, [3,3]`
        The matrix from which to rotate.
    PRY : `numpy.ndarray, [N,3]`
        Numoy array to hold computed pitch roll yaw vales.

    Returns
    -------
    None

    """
    
    N = len(R)
    
    for k in prange(N):
        """ Compute the yaw angle """
        Rinit = (R0.T).dot(R[k,:,:])
        PRY[k,2] = np.arctan2(Rinit[2,1], Rinit[2,2])
        c_y = np.cos(PRY[k,2])
        s_y = np.sin(PRY[k,2])
        
        M_y = np.array([
                        [1.0,  0.0,  0.0],
                        [0.0,  c_y,  s_y],
                        [0.0, -s_y,  c_y]
                        ])
        
        Rp = Rinit.dot(M_y)
        
        """ Compute the roll angle """
        PRY[k,1] = np.arctan2(-Rp[2,0], Rp[2,2])
        c_r = np.cos(PRY[k,1])
        s_r = np.sin(PRY[k,1])
        
        M_r = np.array([
                        [c_r,  0.0, -s_r],
                        [0.0,  1.0,  0.0],
                        [s_r,  0.0,  c_r]
                        ])
        
        Rp = Rp.dot(M_r)
        
        """ Compute the pitch angle """
        PRY[k,0] = np.arctan2(Rp[1,0], Rp[1,1])
        
@njit
def YRPfromRotation(R, R0, YRP):
    """
    Compute Yaw, roll and pitch from rotation matrix
    
    This function computes the Pitch angle, Roll angle
    and Yaw angle from a given rotation matrix. The
    algorithm for computing these values can be found
    in my notes.
    
    The pitch, roll and yaw angles are given by \theta_p,
    \theta_r and \theta_y, respectively.
    
    The Pitch matrix is given by
    
    M_p = [cos\theta_p  sin\theta_p  0]
          [-sin\theta_p cos\theta_p  0]
          [0            0            1].
          
    
    The roll matrix is given by
    
    M_r = [cos\theta_r  0 -sin\theta_r]
          [0            1            0]
          [sin\theta_r  0  cos\theta_r].
    
    
    And the yaw matrix is given by
    
    M_y = [1            0            0]
          [0  cos\theta_y  sin\theta_y]
          [0  -sin\theta_y cos\theta_y].
          
    The **rotated** basis vectors i_1, j_1 and k_1 relate to the
    original basis vectors, i_0, j_0, k_0 through
    
    [i_1, j_1, k_1] = [i_0, j_0, k_0]M_y^TM_r^TM_p^T
    
    where [i_1, j_1, k_1] and [i_0, j_0, k_0] are 3x3 matrices with columns
    corresponding to the indicated basis vectors.
    
    Assuming that [i_0, j_0, k_0] is the identity matrix, this function
    takes input as R = [i_1, j_1, k_1] and computes M_p, M_r and M_y and
    the corresponding Euler angles.
    
    Parameters
    ----------
    R : `numpy.ndarray, [N,3,3]`
        The final rotation matrices.
    R0 : `numpy.ndarray, [3,3]`
        The matrix from which to rotate.
    YRP : `numpy.ndarray, [N,3]`
        Numoy array to hold computed pitch roll yaw vales.

    Returns
    -------
    None

    """
    
    N = len(R)
    
    for k in prange(N):
        """ Compute the pitch angle """
        Rinit = (R0.T).dot(R[k,:,:])
        YRP[k,2] = np.arctan2(-Rinit[0,1], Rinit[0,0])
        c_p = np.cos(YRP[k,2])
        s_p = np.sin(YRP[k,2])
        
        M_p = np.array([
                        [c_p,  s_p,  0.0],
                        [-s_p, c_p,  0.0],
                        [0.0,  0.0,  1.0]
                        ])
        
        Rp = Rinit.dot(M_p)
        
        """ Compute the roll angle """
        YRP[k,1] = np.arctan2(Rp[0,2], Rp[0,0])
        c_r = np.cos(YRP[k,1])
        s_r = np.sin(YRP[k,1])
        
        M_r = np.array([
                        [c_r,  0.0, -s_r],
                        [0.0,  1.0,  0.0],
                        [s_r,  0.0,  c_r]
                        ])
        
        Rp = Rp.dot(M_r)
        
        """ Compute the yaw angle """
        YRP[k,0] = np.arctan2(-Rp[1,2], Rp[1,1])


def aeuAnglesAAEUfromDAEU(AAEU, DAEU):
    aeu = np.zeros((len(AAEU),3), dtype = np.double)
    YRPfromRotation(AAEU, DAEU, aeu)
    return aeu.dot(np.array([[0,1,0],
                             [1,0,0],
                             [0,0,1]]))


def aeuAnglesDAEUfromAAEU(DAEU, AAEU):
    aeu = np.zeros((len(DAEU),3), dtype = np.double)
    PRYfromRotation(DAEU, AAEU, aeu)
    return -aeu.dot(np.array([[0,0,1],
                              [1,0,0],
                              [0,1,0]]))

# @njit
# def AETfromRotation(R, R0, AEU):
#     """
#     Compute Tilt, Elevation and Pitch from rotation matrix
    
#     This function computes the Pitch angle, Roll angle
#     and Yaw angle from a given rotation matrix. The
#     algorithm for computing these values can be found
#     in my notes.
    
#     The pitch, roll and yaw angles are given by \theta_p,
#     \theta_r and \theta_y, respectively.
    
#     The Pitch matrix is given by
    
#     M_t = [cos\theta_t  sin\theta_t  0]
#           [-sin\theta_t cos\theta_t  0]
#           [0            0            1].
          
    
#     The roll matrix is given by
    
#     M_e = [cos\theta_e  0 -sin\theta_e]
#           [0            1            0]
#           [sin\theta_e  0  cos\theta_e].
    
    
#     And the yaw matrix is given by
    
#     M_a = [1            0            0]
#           [0  cos\theta_a  sin\theta_a]
#           [0  -sin\theta_a cos\theta_a].
          
#     The **rotated** basis vectors i_1, j_1 and k_1 relate to the
#     original basis vectors, i_0, j_0, k_0 through
    
#     [i_1, j_1, k_1] = [i_0, j_0, k_0]M_p^TM_r^TM_y^T
    
#     where [i_1, j_1, k_1] and [i_0, j_0, k_0] are 3x3 matrices with columns
#     corresponding to the indicated basis vectors.
    
    
#     Parameters
#     ----------
#     R : `numpy.ndarray, [N,3,3]`
#         The final rotation matrices.
#     R0 : `numpy.ndarray, [3,3]`
#         The matrix from which to rotate.
#     PRY : `numpy.ndarray, [N,3]`
#         Numoy array to hold computed pitch roll yaw vales.

#     Returns
#     -------
#     None

#     """
    
#     N = len(R)
    
#     for k in prange(N):
#         """ Compute the tilt angle """
#         Rinit = (R0.T).dot(R[k,:,:])
#         AEU[k,2] = np.arctan2(Rinit[2,1], Rinit[2,2])
#         c_y = np.cos(AEU[k,2])
#         s_y = np.sin(AEU[k,2])
        
#         M_y = np.array([
#                         [1.0,  0.0,  0.0],
#                         [0.0,  c_y, -s_y],
#                         [0.0,  s_y,  c_y]
#                         ])
        
#         Rp = Rinit.dot(M_y)
        
#         """ Compute the elevation angle """
#         AEU[k,1] = np.arctan2(-Rp[2,0], Rp[2,2])
#         c_r = np.cos(AEU[k,1])
#         s_r = np.sin(AEU[k,1])
        
#         M_r = np.array([
#                         [c_r,  0.0,  s_r],
#                         [0.0,  1.0,  0.0],
#                         [-s_r, 0.0,  c_r]
#                         ])
        
#         Rp = Rp.dot(M_r)
        
#         """ Compute the azimuth angle """
#         AEU[k,0] = np.arctan2(Rp[1,0], Rp[1,1])    