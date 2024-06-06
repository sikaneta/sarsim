# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:01:21 2023

@author: ishuwa.sikaneta
"""

import json
import numpy as np

#%%
def getListElementType(z):
    """
    Determine type of element in a list.
    
    It is assumed that all elements of the list are of the same type as only
    the first list element type is returned.

    Parameters
    ----------
    z : `list`
        The list containing the elements for which we wish to determine the 
        type.

    Returns
    -------
    type
        The type of the elements.

    """
    if type(z) == list:
        return getListElementType(z[0])
    return type(z)

#%%
def ndarray2json(z):
    """
    Convery an ndarray to json
    
    Since json does not store complex data types, a compound json type, 
    composed of simple types, is created to represent the ndarray type. In
    particular, this method helps to deal with complex-valued data.

    Parameters
    ----------
    z : `numpy.ndarray`
        A numpy ndarray to cast to json.

    Returns
    -------
    `dict`
        A dictionary representation that is castable to json.

    """
    if type(z) != np.ndarray:
        raise TypeError("Input should be an numpy.ndarray")
    if getListElementType(z.tolist()) == complex:
        return {
            "_complex_": True,
            "re": np.real(z).tolist(),
            "im": np.imag(z).tolist()
        }
    return z.tolist()
    
#%%
def mtx2svddict(W, threshold=1e-10):
    """
    Convert a matrix into svd representation
    
    Convert a complex-valued matrix into a Singular Value Decomposition (SVD).
    
    See [Wikpedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)
    
    A threshold is provided allowing a any singular values below the threshold
    are ignored. Although somewhat lossy, this allows the key characteristics
    of the matrix to be compressed and stored. Specifically, this approach
    allows a 2-D antenna weightings to be, potentially, stored in compact 
    format.

    Parameters
    ----------
    W : `numpy.ndarray((N,M))`
        A 2-D array to be decomposed.
    threshold : `float`, optional
        The threshold below which singular values and their vectors shall be 
        discarded. The default is 1e-10.

    Returns
    -------
    mydict : `dict`
        The singular value decomposition in dictionary format. This format is
        suitable for writing to a json file.

    """
    u, s, vh = np.linalg.svd(W, full_matrices=False)
    mydict = {"type": "svd",
              "components": []}
    for k, l in enumerate(s):
        if np.abs(l) > threshold:
            mydict["components"].append({
                "s": l,
                "u": u[:,k],
                "vh": vh[k,:]
                })
    return mydict

#%%
def json2ndarray(z):
    """
    Convert json to a numpy.ndarray
    
    This is the reverse operation to :meth:`~cjson.ndarray2json`. This 
    function specifically attempts to convert a complex data type.

    Parameters
    ----------
    z : `dict`
        A json representation of an ndarray.

    Raises
    ------
    TypeError
        If the conversion to an ndarray fails.

    Returns
    -------
    `numpy.ndarray`
        The numpy array representation of the dictionary data.

    """
    try:
        return np.array(z["re"]) + 1j*np.array(z["im"])
    except KeyError:
        raise TypeError("This does not seem to be a complex json array")

#%%
def dict2json(dct):
    """
    Convert a dictionary to json serializable dictionary
    
    Convert a dictionary with numpy.ndarray datatypes into a dictionary that
    can be json serialized.

    Parameters
    ----------
    dct : `dict`
        Dictionary with numpy.ndarray values.

    Returns
    -------
    `dict`
        A json serializable dictionary representation of the input dictionary.

    """
    if type(dct) == list:
        return [dict2json(x) for x in dct]
    if type(dct) != dict:
        return ndarray2json(dct)
    return {k:dict2json(v) for k,v in dct.items()}

#%%
def json2dict(myjson):
    """
    Convert json dictionary to complex data type
    
    In the reverse operation to :meth:`~cjson.dict2json`, convert a json
    serializable dictionary into a dictionary with complex `numpy.ndarray`
    datatype values.

    Parameters
    ----------
    myjson : `dict`
        Input json serializable dictionary.

    Returns
    -------
    `dict`
        Output dictionary with dictionary containing numpy.ndarray complex
        datatype values.

    """
    if type(myjson) == list:
        return [json2dict(x) for x in myjson]
    if type(myjson) == dict:
        if "_complex_" in myjson.keys():
            return json2ndarray(myjson)
    else:
        return myjson
    return {k:json2dict(v) for k,v in myjson.items()}
    
            