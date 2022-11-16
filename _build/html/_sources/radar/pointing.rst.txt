Simulation of Pointing Errors
=============================

This page outlines how pointing errors are Monte-Carlo-simulated.

Python classes and modules have been developed to assist with pointing
error analysis. This code takes a satellite orbit around a given planet
as input. It also takes parameters of various state-vector-related pointing 
errors, as described in `Pointing Justification`_, as input parameters.

Orientation Orbit Class
~~~~~~~~~~~~~~~~~~~~~~~
The class :py:class:`orbit.orientation.orbit` contains code that allows for computation
of the required zero-Doppler steering law for a given orbit around a given planet. Currently,
:py:class:`space.planets.earth` and :py:class:`space.planets.venus` are defined as possible planets.

.. autoclass:: orbit.orientation.orbit

Simulation Class
~~~~~~~~~~~~~~~~
The class :py:class:`orbit.pointing.simulation` allows pointing error contributions to be simlated for
a given orbit. The class methods allow implementation of the pointing error contributions described 
in `Pointing Justification`_. 

.. autoclass:: orbit.pointing.simulation
   
Euler Angle Calculation Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module provides computations between Euler angles and rotation matrices as specified in 
`Reference Systems`_. Several methods have been decorated with numba decorators to accelerate computations.

.. automodule:: orbit.euler

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   orbit/euler.rst