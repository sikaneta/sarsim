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

base_SentinelNG = """
<gmtiAuxDataParams schemaVersion="1.0">
  <acquisitionParameters>
    <mode>Ish_Burst_Mode</mode>
    <instrument>
      <antennaArray>     
        <carrierFrequency unit="Hz">5.4050e9</carrierFrequency>
        <azimuthPositions unit="m">0.682219 2.046658 3.411097 4.775536 6.139975 7.504414 8.868853 10.233292 11.597731</azimuthPositions>
        <elevationPositions unit="m">0.019643 0.058928 0.098213 0.137497 0.176783 0.216067 0.255352 0.294637 0.333923 0.373208 0.412492 0.451777 0.491063 0.530347 0.569632 0.608918 0.648203 0.687488 0.726773 0.766058 0.805342 0.844627 0.883912 0.923198</elevationPositions>
        <azimuthElementLengths unit="m">1.364439 1.364439 1.364439 1.364439 1.364439 1.364439 1.364439 1.364439 1.364439</azimuthElementLengths>
        <elevationElementLengths unit="m">0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285 0.039285</elevationElementLengths>
        <transmitPowers unit="dB">1 1 1 1 1 1 1 1 1</transmitPowers>
        <systemTemperature unit="degrees" system="Kelvin">297</systemTemperature>
        <systemLosses unit="dB">-5.3</systemLosses>
      </antennaArray>
                  
      <lookDirection>Right</lookDirection>
      <orbit>
        <inclination unit="degrees">98.6</inclination>
        <eccentricity>0.00</eccentricity>
        <semiMajorAxis unit="m">6985960</semiMajorAxis>
        <angleOfPerigee unit="degrees">90.0</angleOfPerigee>
        <orbitAngle unit="degrees">60</orbitAngle>
      </orbit>
    </instrument>
    <platformLongitude unit="degrees" referenceEllipsoid="WGS84">90</platformLongitude>
  </acquisitionParameters>
</gmtiAuxDataParams>"""


base_ROSEL = """
<gmtiAuxDataParams schemaVersion="1.0">
  <acquisitionParameters>
    <mode>Ish_Burst_Mode</mode>
    <instrument>
      <antennaArray>     
        <carrierFrequency unit="Hz">1.2575e9</carrierFrequency>
        <azimuthPositions unit="m">0.09166666666666666 0.27499999999999997 0.4583333333333333 0.6416666666666667 0.825 1.0083333333333333 1.1916666666666667 1.375 1.5583333333333331 1.7416666666666665 1.9249999999999998 2.1083333333333334 2.291666666666667 2.475 2.6583333333333337 2.841666666666667 3.025 3.2083333333333335 3.3916666666666666 3.575 3.7583333333333333 3.941666666666667 4.125 4.308333333333334 4.491666666666667 4.675 4.858333333333333 5.041666666666667 5.2250000000000005 5.408333333333333 5.591666666666667 5.775 5.958333333333333 6.141666666666667 6.325 6.508333333333334 6.691666666666666 6.875 7.058333333333334 7.241666666666667 7.425 7.608333333333333 7.791666666666667 7.9750000000000005 8.158333333333333 8.341666666666667 8.525 8.708333333333334 8.891666666666667 9.075 9.258333333333333 9.441666666666666 9.625 9.808333333333334 9.991666666666667 10.175 10.358333333333334 10.541666666666666 10.725 10.908333333333333</azimuthPositions>
        <elevationPositions unit="m">0.075 0.225 0.375 0.525 0.675 0.825 0.975 1.125 1.275 1.425 1.575 1.725 1.875 2.025 2.175 2.325 2.475 2.625 2.775 2.925 3.075 3.225 3.375 3.525</elevationPositions>
        <azimuthElementLengths unit="m">0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332 0.18333333333333332</azimuthElementLengths>
        <elevationElementLengths unit="m">0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15</elevationElementLengths>
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