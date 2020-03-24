#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:39:34 2020

@author: ishuwa
"""

base_XML_CBand = """
<gmtiAuxDataParams schemaVersion="1.0">
  <acquisitionParameters>
    <mode>Ish_Burst_Mode</mode>
    <instrument>
      <antennaArray>     
        <carrierFrequency unit="Hz">5.405e9</carrierFrequency>
        <azimuthPositions unit="m">0.4688 1.4062 2.3438 3.2812 4.2188 5.1562 6.0938 7.0312 7.9688 8.9062 9.8438 10.7812 11.7188 12.6562 13.5938 14.5312</azimuthPositions>
        <elevationPositions unit="m">0.75</elevationPositions>
        <azimuthElementLengths unit="m">0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286</azimuthElementLengths>
        <elevationElementLengths unit="m">1.5</elevationElementLengths>
        <transmitPowers unit="dB">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</transmitPowers>
        <systemTemperature unit="degrees" system="Kelvin">297</systemTemperature>
        <systemLosses unit="dB">-5.3</systemLosses>
      </antennaArray>
                  
      <lookDirection>Right</lookDirection>
      <orbit>
        <inclination unit="degrees">98.6</inclination>
        <eccentricity>0.00</eccentricity>
        <semiMajorAxis unit="m">7071009</semiMajorAxis>
        <angleOfPerigee unit="degrees">90.0</angleOfPerigee>
        <orbitAngle unit="degrees">60</orbitAngle>
      </orbit>
    </instrument>
    <platformLongitude unit="degrees" referenceEllipsoid="WGS84">90</platformLongitude>
  </acquisitionParameters>
</gmtiAuxDataParams>"""

base_XML_XBand = """
<gmtiAuxDataParams schemaVersion="1.0">
  <acquisitionParameters>
    <mode>Ish_Burst_Mode</mode>
    <instrument>
      <antennaArray>     
        <carrierFrequency unit="Hz">9.650e9</carrierFrequency>
        <azimuthPositions unit="m">0.4688 1.4062 2.3438 3.2812 4.2188 5.1562 6.0938 7.0312 7.9688 8.9062 9.8438 10.7812 11.7188 12.6562 13.5938 14.5312</azimuthPositions>
        <elevationPositions unit="m">0.75</elevationPositions>
        <azimuthElementLengths unit="m">0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286</azimuthElementLengths>
        <elevationElementLengths unit="m">1.5</elevationElementLengths>
        <transmitPowers unit="dB">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</transmitPowers>
        <systemTemperature unit="degrees" system="Kelvin">297</systemTemperature>
        <systemLosses unit="dB">-5.3</systemLosses>
      </antennaArray>
                  
      <lookDirection>Right</lookDirection>
      <orbit>
        <inclination unit="degrees">98.6</inclination>
        <eccentricity>0.00</eccentricity>
        <semiMajorAxis unit="m">7071009</semiMajorAxis>
        <angleOfPerigee unit="degrees">90.0</angleOfPerigee>
        <orbitAngle unit="degrees">60</orbitAngle>
      </orbit>
    </instrument>
    <platformLongitude unit="degrees" referenceEllipsoid="WGS84">90</platformLongitude>
  </acquisitionParameters>
</gmtiAuxDataParams>"""