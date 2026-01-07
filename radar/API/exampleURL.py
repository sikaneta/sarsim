url = r"http://localhost:5000/integrate?timeUTC=2020-07-28T17:13:58.123756826&x=-5.28682880e05&y=-6.12367342e06&z=3.49575263e06&vx=1.41881891e03&vy=-3.79246352e03&vz=-6.42885957e03&dt=2.0"
      

""" Below, true anomaly corresponds to the time of periapsis """

fldsA = {
    "name": "example_kepler_semi_major",
    "propagation_method": "Keplerian",
    "propagation_parameter": {
        "a": 7000e3,
        "e": 0.001,
        "inclination": 98.6,
        "ascendingNode": 0,
        "arg_perigee": 40,
        "true_anomaly": 20,
        "epoch": "2025-09-21T10:00:00.234867251",
        "delta_t": 1/5000,
        "n_samples": 10
    }
}

fldsB = {
    "name": "example_kepler_period",
    "propagation_method": "Keplerian",
    "propagation_parameter": {
        "period": 100*60,
        "e": 0.001,
        "inclination": 98.6,
        "ascendingNode": 0,
        "arg_perigee": 0,
        "true_anomaly": 0,
        "epoch": "2025-09-21T10:00:00.234867251",
        "delta_t": 1/5000,
        "n_samples": 10
    }
}

fldsC = {
    "name": "example_kepler_orbits_per_day",
    "propagation_method": "Keplerian",
    "propagation_parameter": {
        "orbits_per_day": 11.2,
        "e": 0.001,
        "inclination": 98.6,
        "ascendingNode": 0,
        "arg_perigee": 0,
        "true_anomaly": 0,
        "epoch": "2025-09-21T10:00:00.234867251",
        "delta_t": 1/5000,
        "n_samples": 10
    }
}