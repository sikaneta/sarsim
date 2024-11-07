# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:48:55 2023

@author: ishuwa.sikaneta
"""
#%%
from elasticsearch import Elasticsearch
import numpy as np
from measurement.measurement import state_vector
from space.planets import venus
import os
import sys

#%%
conf_path = os.path.join(os.environ['HOMEPATH'],".sarsim")
if conf_path not in sys.path:
    sys.path.append(conf_path)
try:
  from sarsim_conf import ELASTIC_USER, ELASTIC_PASSWORD, ELASTIC_URL
except ModuleNotFoundError as MNF:
  print(MNF)

#%%

# Create the client instance
client = Elasticsearch(
    ELASTIC_URL,
    ca_certs=r"C:\Users\ishuwa.sikaneta\local\data\certs\http_ca.crt",
    http_auth=(ELASTIC_USER, ELASTIC_PASSWORD)
)

#%% Successful response!
client.info()

#%%


#%% 
def getSVFromTime(utcTime, frame = "IAU_VENUS", dT = 100):
    myT = np.datetime64(utcTime)
    mydT = np.timedelta64(dT, 's')
    qry = {
        "function_score": {
          "query": {
            "range": {
              "timeUTC": {
                "gte": (myT - mydT).astype(str),
                "lte": (myT + mydT).astype(str)
              }
            }
          },
          "gauss": {
            "timeUTC": {
              "origin": utcTime,
              "scale": "1m",
              "decay": 0.5
            }
          }
        }
    }
    resp = client.search(index="envision_orbit", query = qry)
    sv = state_vector(planet = venus())
    try:
        point = resp['hits']['hits'][0]
        vect = [x for x in point["_source"]["stateVector"] 
                     if x["frame"] == frame][0]
        sv = state_vector(planet = venus())
        sv.add(np.datetime64(vect["time"]),
               np.array(vect["xyzVxVyVz"]))
        return sv
    except IndexError:
        return None
    
#%%
def getSV(utcArray, frame = "IAU_VENUS"):
    
    qry = {
        "range": {
          "timeUTC": {
            "gte": utcArray[0].astype('str'),
            "lte": utcArray[-1].astype('str')
          }
        }
      }
    
    resp = client.search(index="envision_orbit", query = qry)
    sv = state_vector(planet = venus())
    for point in resp['hits']['hits']:
        vect = [x for x in point["_source"]["stateVector"] 
                     if x["frame"] == frame][0]
        sv.add(np.datetime64(vect["time"]),
               np.array(vect["xyzVxVyVz"]))
        
    sVects = sv.estimateTimeRange(utcArray)
    ssv = state_vector(planet = venus())
    for tm, vct in zip(utcArray, sVects):
        ssv.add(tm, vct)
        
    return ssv

#%%
def updateIndexRecord(index, docid, doc):
    resp = client.index(index=index, id=docid, document=doc)
    return resp
    


