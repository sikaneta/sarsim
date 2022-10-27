# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:50:20 2022

@author: Ishuwa.Sikaneta
"""
#%%
import numpy as np
from numba import njit, prange

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

#%%      
@njit
def rpyFromRotation(R, RPY):
    """
    Compute roll and pitch and yaw from rotation matrix
    
    This function computes the Roll angle, Pitch angle
    and Yaw angle from a given rotation matrix. The
    algorithm for computing these values can be found
    in my notes.
    
    The roll, pitch and yaw angles are given by \theta_r,
    \theta_p and \theta_y, respectively.
    
    The roll matrix is given by
    
    M_r = [1            0            0]
          [0  cos\theta_r  sin\theta_r]
          [0  -sin\theta_r cos\theta_r],    
    
    the pitch matrix is given by
    
    M_p = [cos\theta_p  0 -sin\theta_p]
          [0            1            0]
          [sin\theta_p  0  cos\theta_p],
          
    and the Yaw matrix is given by
    
    M_y = [cos\theta_y  sin\theta_y  0]
          [-sin\theta_y cos\theta_y  0]
          [0            0            1].
          
    The **rotated** basis vectors i_1, j_1 and k_1 relate to the
    original basis vectors, i_0, j_0, k_0 through
    
    [i_1, j_1, k_1] = [i_0, j_0, k_0]M_r^TM_p^TM_y^T
    
    where [i_1, j_1, k_1] and [i_0, j_0, k_0] are 3x3 matrices with columns
    corresponding to the indicated basis vectors.
    
    Assuming that [i_0, j_0, k_0] is the identity matrix, this function
    takes input as R = [i_1, j_1, k_1] and computes M_r, M_p and M_y and
    the corresponding Euler angles.
    
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
        
#%%     
@njit
def aeu2rot(old_basis, aeu_e, new_basis):
    """
    Rotate a set of basis vectors by aeu angles.
    
    Rotate a set of basis vectors according to eq (25) in the Envision
    reference frames and pointing angle errors document. If the basis is
    the identity matrix, then the return values are the rotation matrices.

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

#%%
@njit
def RPYfromAEU(AEU, M_e, RPY):
    """
    Compute Yaw, roll and pitch errors from azimuth, elevation and
    tilt errors
    
    This function computes the Roll angle, Pitch angle
    and Yaw angle errors from given azimuth, elevation and tile
    errors. The algorithm for computing these values can be found
    in my notes.
    
    Roll, Pitch Yaw errors
    ----------------------
    
    The roll, pitch and yaw angles are given by \theta_r,
    \theta_p and \theta_y, respectively.
    
    The roll matrix is given by
    
    M_r = [1            0            0]
          [0  cos\theta_r  sin\theta_r]
          [0  -sin\theta_r cos\theta_r],    
    
    the pitch matrix is given by
    
    M_p = [cos\theta_p  0 -sin\theta_p]
          [0            1            0]
          [sin\theta_p  0  cos\theta_p],
          
    and the Yaw matrix is given by
    
    M_y = [cos\theta_y  sin\theta_y  0]
          [-sin\theta_y cos\theta_y  0]
          [0            0            1].
          
    The **rotated** basis vectors i_1, j_1 and k_1 relate to the
    original basis vectors, i_0, j_0, k_0 through
    
    [j_1, k_1, i_1] = [j_0, k_0, i_0]M_r^TM_p^TM_y^T
    
    where [j_1, k_1, i_1] and [j_0, k_0, i_0] are 3x3 matrices with columns
    corresponding to the indicated basis vectors.
    
    AUE angular errors
    ------------------
     
    The elevation, azimuth and tilt angles are given by \epsilon,
    \alpha and \tau, respectively.
    
    The elevation matrix is given by
    
    M_e = [1            0            0]
          [0  cos\epsilon  sin\epsilon]
          [0  -sin\epsilon cos\epsilon],    
    
    the pitch matrix is given by
    
    M_a = [cos\alpha  0 -sin\alpha]
          [0          1          0]
          [sin\alpha  0  cos\alpha],
          
    and the Yaw matrix is given by
    
    M_u = [cos\tau  sin\tau  0]
          [-sin\tau cos\tau  0]
          [0        0        1].
          
    The **rotated** basis vectors a_1, e_1 and u_1 relate to the
    original basis vectors, a_0, e_0, u_0 through
    
    [a_1, e_1, u_1] = [a_0, e_0, u_0]M_e^TM_a^TM_u^T
    
    where [a_1, e_1, u_1] and [a_0, e_0, u_0] are 3x3 matrices with columns
    corresponding to the indicated basis vectors.
    
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
    
    Roll, Pitch and Yaw errors
    --------------------------
    
    The roll, pitch and yaw angles are given by \theta_r,
    \theta_p and \theta_y, respectively.
    
    The roll matrix is given by
    
    M_r = [1            0            0]
          [0  cos\theta_r  sin\theta_r]
          [0  -sin\theta_r cos\theta_r],    
    
    the pitch matrix is given by
    
    M_p = [cos\theta_p  0 -sin\theta_p]
          [0            1            0]
          [sin\theta_p  0  cos\theta_p],
          
    and the Yaw matrix is given by
    
    M_y = [cos\theta_y  sin\theta_y  0]
          [-sin\theta_y cos\theta_y  0]
          [0            0            1].
          
    The **rotated** basis vectors i_1, j_1 and k_1 relate to the
    original basis vectors, i_0, j_0, k_0 through
    
    [j_1, k_1, i_1] = [j_0, k_0, i_0]M_r^TM_p^TM_y^T
    
    AEU errors
    ----------
    
    The elevation, azimuth and tilt angles are given by \epsilon,
    \alpha and \tau, respectively.
    
    The elevation matrix is given by
    
    M_e = [1            0            0]
          [0  cos\epsilon  sin\epsilon]
          [0  -sin\epsilon cos\epsilon],    
    
    the pitch matrix is given by
    
    M_a = [cos\alpha  0 -sin\alpha]
          [0          1          0]
          [sin\alpha  0  cos\alpha],
          
    and the Yaw matrix is given by
    
    M_u = [cos\tau  sin\tau  0]
          [-sin\tau cos\tau  0]
          [0        0        1].
          
    The **rotated** basis vectors a_1, e_1 and u_1 relate to the
    original basis vectors, a_0, e_0, u_0 through
    
    [a_1, e_1, u_1] = [a_0, e_0, u_0]M_e^TM_a^TM_u^T
    
    where [a_1, e_1, u_1] and [a_0, e_0, u_0] are 3x3 matrices with columns
    corresponding to the indicated basis vectors.
    
    Parameters
    ----------
    AEU : `numpy.ndarray, [3, N]`
        The a, e, u values as a matrix of Nx3.
    M_e : `numpy.ndarray, [3,3]`
        The matrix that rotates aeu into jki_s.
    RPY : `numpy.ndarray, [3, N]`
        Numoy array to hold computed roll pitch yaw vales.

    Note
    ----
    epsilon -> roll
    alpha -> pitch
    tau -> yaw
    
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
        
# def aeuAnglesAAEUfromDAEU(AAEU, DAEU):
#     aeu = np.zeros((len(AAEU),3), dtype = np.double)
#     YRPfromRotation(AAEU, DAEU, aeu)
#     return aeu.dot(np.array([[0,1,0],
#                               [1,0,0],
#                               [0,0,1]]))

def aeuFromRotation(AEUmatrix):
    """
    Compute the azimuth, elevation and tilt angles from rotation matrices.
    
    According to equation (25) of the Envision reference frames and pointing
    angle error document, compute the angles corresponding to given
    rotation matrices.

    Parameters
    ----------
    AAEmatrix : TYPE
        DESCRIPTION.
    DAEU : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    eau = np.zeros((3, len(AEUmatrix)), dtype = np.double)
    rpyFromRotation(AEUmatrix, eau)
    
    """ Need to switch the order of values so that we get
        aeu order rather than eau """
    aeShift = np.array([[0,1,0],
                        [1,0,0],
                        [0,0,1]])
    # aeShift = np.array([[1,0,0],
    #                     [0,1,0],
    #                     [0,0,1]])
    
    return aeShift.dot(eau)

#%%
def rpyAnglesFromIJK(IJKmatrix):
    JKIshift = np.array([[0,0,1],[1,0,0],[0,1,0]])
    rpy = np.zeros((3, len(IJKmatrix)), dtype = np.double)
    rpyFromRotation(IJKmatrix.dot(JKIshift), rpy)
    return rpy

# def aeuAnglesDAEUfromAAEU(DAEU, AAEU):
#     aeu = np.zeros((len(DAEU),3), dtype = np.double)
#     PRYfromRotation(DAEU, AAEU, aeu)
#     return -aeu.dot(np.array([[0,0,1],
#                               [1,0,0],
#                               [0,1,0]]))

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