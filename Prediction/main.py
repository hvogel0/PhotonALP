#! /usr/bin/env python 3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
"""This code initiates the computation of the statistical prediciton."""

#--- IMPORTS ----------------
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy. integrate import nquad
import sys
sys.path.insert(0, '../Parameters/')
import constants as cst
import parameters as para
from astropy.coordinates import SkyCoord
#----------------------------

#Check if the right amount of arguments are supplied. Exit if not. 
#Arguments are Host Galaxy magnetic field 'mag' [muG], ALP mass in log_10 (neV), photon-ALP coupling in 10^-11 GeV-1
if len(sys.argv)<3:
    print("Not enough arguments \n")
    sys.exit()
if len(sys.argv)>3:
    print("Too many arguments \n")
    sys.exit()
mag = sys.argv[1]
gag = sys.argv[2]
mass = sys.argv[3]

sys.exit()
#TODO: THink about file structure
#load gamma ray bkg
GammaSigRaw = np.loadtxt('../Milky_Way/Combined/GFlux_ma'+mass+'_g'+gag+'.dat')
GammaBKGRaw = np.loadtxt('../../MW_GR/GRMisiriotis/Combined/GFlux_ma'+mass+'_g0.dat')
ALPSigRaw = np.loadtxt('../../GammaALPFluxesKopper/SYICFlux/Data/GFlux_B'+mag+'_ma'+mass+'_g'+gag+'.dat')
