# Radvel_Introduction
Public repository for helping people to learn radvel

Example code for creating an anaconda environment that uses radvel:
  conda create --name newrad python=3.6
  conda activate newrad

  conda install --channel "conda-forge" numpy
  conda install scipy matplotlib cython astropy pandas
  pip install corner celerite emcee
  pip install radvel --upgrade
  pip install h5py

-------------------------------
-------   Session 1.    -------
-------------------------------

1) Installation
2) Review Fulton et al. 2018
3) Introduction to RadVel, basic fit
    - Probably Pegasi 51b. An easy, large amplitude 1 planet fit, no trend, activity, etc.


-------------------------------
-------   Session 2.    -------
-------------------------------

1) Something a little more complicated. Let's try a multi-planet run, and let's include a system with some trend and curvature.
2) HD 191939 public data is probably a good choice for this



-------------------------------
-------   Session 3.    -------
-------------------------------

1) Let's get spicy. Let's do a fit that is hampered by activity, and requires a GP.
2) Kepler-21 is a good option, or possibly CoRoT-7


-------------------------------
-------   Session 4.    -------
-------------------------------

1) Advanced GP fits
    - Let's introduce the Chromatic GP Kernels from Cale et al. 2021. AU Mic should be our test case here.
2) More Advanced Radvel
    - Keplerdrive, modifying plots, decorrelation
