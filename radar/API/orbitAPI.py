from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import numpy as np
from measurement.measurement import state_vector, myLegendre_numba
from measurement.arclength import slow
import json

app = Flask(__name__)
api = Api(app)

# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('timeUTC', type=str)
parser.add_argument('x', type=float)
parser.add_argument('y', type=float)
parser.add_argument('z', type=float)
parser.add_argument('vx', type=float)
parser.add_argument('vy', type=float)
parser.add_argument('vz', type=float)
parser.add_argument('dt', type=float)


class Item(Resource):
#     sv = state_vector()
#     """ Force compilation of the numba function"""
#     _ = sv.expandedState(np.array([-5.28682880e+05, 
#                                    -6.12367342e+06,  
#                                    3.49575263e+06,  
#                                    1.41881891e+03,
#                                    -3.79246352e+03, 
#                                    -6.42885957e+03]), 0.0)
#     C = slow([np.datetime64('2015-01-01T00:00:04.721666194')])
    def post(self):
        sv = state_vector()
        args = parser.parse_args()

        timeUTC = args['timeUTC']
        C = slow([np.datetime64(timeUTC)])
        measurementData = np.array([float(args['x']),
                                    float(args['y']),
                                    float(args['z']),
                                    float(args['vx']),
                                    float(args['vy']),
                                    float(args['vz'])])
        eState = sv.expandedState(measurementData, 0.0)
        cdf, tdf, T, N, B, kappa, tau, dkappa = C.diffG(eState)
        diffGeo = {
            "expansion": {
                "cdf": [list(e) for e in cdf],
                "tdf": tdf
            },
            "T": list(T),
            "N": list(N),
            "B": list(B),
            "kappa": kappa,
            "tau": tau,
            "dkappa": dkappa}
        X = list(eState[0])
        dX = list(eState[1])
        ddX = list(eState[2])
        dddX = list(eState[3])
        stateECEF = {
            "timeUTC": timeUTC,
            "X": X,
            "dX": dX,
            "ddX": ddX,
            "dddX": dddX}
        return jsonify(stateECEF=stateECEF,
                        diffGeo=diffGeo)
    def get(self):
        sv = state_vector()
        args = parser.parse_args()

        timeUTC = args['timeUTC']
        C = slow([np.datetime64(timeUTC)])
        measurementData = np.array([float(args['x']),
                                    float(args['y']),
                                    float(args['z']),
                                    float(args['vx']),
                                    float(args['vy']),
                                    float(args['vz'])])
        eState = sv.expandedState(measurementData, 0.0)
        cdf, tdf, T, N, B, kappa, tau, dkappa = C.diffG(eState)
        diffGeo = {
            "expansion": {
                "cdf": [list(e) for e in cdf],
                "tdf": tdf
            },
            "T": list(T),
            "N": list(N),
            "B": list(B),
            "kappa": kappa,
            "tau": tau,
            "dkappa": dkappa}
        X = list(eState[0])
        dX = list(eState[1])
        ddX = list(eState[2])
        dddX = list(eState[3])
        stateECEF = {
            "timeUTC": timeUTC,
            "X": X,
            "dX": dX,
            "ddX": ddX,
            "dddX": dddX}
        return jsonify(stateECEF=stateECEF,
                        diffGeo=diffGeo)
    

    
class Integrate(Resource):
    def post(self):
        sv = state_vector()
        args = parser.parse_args()

        timeUTC = args['timeUTC']
        # C = slow([np.datetime64(timeUTC)])
        measurementData = np.array([float(args['x']),
                                    float(args['y']),
                                    float(args['z']),
                                    float(args['vx']),
                                    float(args['vy']),
                                    float(args['vz'])])
        measurementTime = np.datetime64(timeUTC)
        
        sv.add(measurementTime, measurementData)
        
        dt = float(args['dt'])
        
        desiredSVUTC = measurementTime + dt*1e9*np.timedelta64(1, 'ns')
        
        integratedSV = sv.estimate(desiredSVUTC)
        format_string = "http://192.168.0.174:5000/item?timeUTC=%s&x=%0.6f&y=%0.6f&z=%0.6f&vx=%0.6f&vy=%0.6f&vz=%0.6f"
        urlitem = format_string % (np.datetime_as_string(desiredSVUTC), 
                                   integratedSV[0], 
                                   integratedSV[1], 
                                   integratedSV[2],  
                                   integratedSV[3], 
                                   integratedSV[4], 
                                   integratedSV[5])
        newSV = {"timeUTC": np.datetime_as_string(desiredSVUTC), 
                 "x": integratedSV[0], 
                 "y": integratedSV[1],  
                 "z": integratedSV[2],  
                 "vx": integratedSV[3], 
                 "vy": integratedSV[4], 
                 "vz": integratedSV[5],
                 "url": urlitem}
        return jsonify(integratedSV=newSV)
    
    def get(self):
        sv = state_vector()

        args = parser.parse_args()

        timeUTC = args['timeUTC']
        measurementData = np.array([float(args['x']),
                                    float(args['y']),
                                    float(args['z']),
                                    float(args['vx']),
                                    float(args['vy']),
                                    float(args['vz'])])
        measurementTime = np.datetime64(timeUTC)
        
        sv.add(measurementTime, measurementData)
        
        dt = float(args['dt'])
        
        desiredSVUTC = measurementTime + dt*1e9*np.timedelta64(1, 'ns')
        
        integratedSV = sv.estimate(desiredSVUTC)
        format_string = "http://192.168.0.174:5000/item?timeUTC=%s&x=%0.6f&y=%0.6f&z=%0.6f&vx=%0.6f&vy=%0.6f&vz=%0.6f"
        urlitem = format_string % (np.datetime_as_string(desiredSVUTC), 
                                   integratedSV[0], 
                                   integratedSV[1], 
                                   integratedSV[2],  
                                   integratedSV[3], 
                                   integratedSV[4], 
                                   integratedSV[5])
        
        newSV = {"timeUTC": np.datetime_as_string(desiredSVUTC), 
                 "x": integratedSV[0], 
                 "y": integratedSV[1],  
                 "z": integratedSV[2],  
                 "vx": integratedSV[3], 
                 "vy": integratedSV[4], 
                 "vz": integratedSV[5],
                 "url": urlitem}
        
        return jsonify(integratedSV=newSV)

api.add_resource(Item, '/item')
api.add_resource(Integrate, '/integrate')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')