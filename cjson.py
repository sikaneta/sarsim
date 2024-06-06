# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:01:21 2023

@author: ishuwa.sikaneta
"""

import json
import numpy as np

#%%
def encode_complex(z):
    if type(z) == complex:
        return (z.real, z.imag)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")
        
#%%
def getListElementType(z):
    if type(z) == list:
        return getListElementType(z[0])
    return type(z)

#%%
def ndarray2json(z):
    if type(z) != np.ndarray:
        return z
    if getListElementType(z.tolist()) == complex:
        return {
            "_complex_": True,
            "re": np.real(z).tolist(),
            "im": np.imag(z).tolist()
        }
    return z.tolist()
    
#%%
def mtx2svddict(W, threshold=1e-10):
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
    try:
        return np.array(z["re"]) + 1j*np.array(z["im"])
    except KeyError:
        raise TypeError("This does not seem to be a complex json array")

#%%
def dict2json(dct):
    if type(dct) == list:
        return [dict2json(x) for x in dct]
    if type(dct) != dict:
        return ndarray2json(dct)
    return {k:dict2json(v) for k,v in dct.items()}

#%%
def json2dict(myjson):
    if type(myjson) == list:
        return [json2dict(x) for x in myjson]
    if type(myjson) == dict:
        if "_complex_" in myjson.keys():
            return json2ndarray(myjson)
    else:
        return myjson
    return {k:json2dict(v) for k,v in myjson.items()}
    
#%%
class ComplexEncoder(json.JSONEncoder):
    def default(self, z):
        if getListElementType(z) == complex:
            z_np = np.array(z)
            return {
                "_complex_": True,
                "re": np.real(z_np).tolist(),
                "im": np.imag(z_np).tolist()
            }
        else:
            return super().default(z)
            