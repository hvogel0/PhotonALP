#! /usr/bin/env python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

"""
This code is for testing purposes
"""

#--- IMPORTS ---------
from scipy.integrate import ode
from scipy.integrate import quad
from scipy import interpolate
import numpy as np
import sys
sys.path.insert(0, '../Parameters/')
import constants as cst
import parameters as para
import mw_parameters as mw
#---------------------


#Testing of integration methods
options = mw.mw_options
for cl in options:
    mwInstance = mw.mwModel(cl,"000","000")
    le = len(mwInstance.eng_dat)
    ld = len(mwInstance.dist_dat)
    print('Gamma test')
    print(mwInstance.gamma_data[0,2],mwInstance.gamma_int(mwInstance.dist_dat[0],mwInstance.eng_dat[0]))
    print(mwInstance.gamma_data[2*le+3,2],mwInstance.gamma_int(mwInstance.dist_dat[2],mwInstance.eng_dat[3]))
    print(mwInstance.gamma_data[5*le+12,2],mwInstance.gamma_int(mwInstance.dist_dat[5],mwInstance.eng_dat[12]))
    print('Chi test')
    le = len(mwInstance.eng_datgg)
    ld = len(mwInstance.dist_datgg)
    print(mwInstance.dgg_data[0,2],mwInstance.dgg_int(mwInstance.dist_datgg[0],mwInstance.eng_datgg[0]))
    print(mwInstance.dgg_data[2*le+3,2],mwInstance.dgg_int(mwInstance.dist_datgg[2],mwInstance.eng_datgg[3]))
    print(mwInstance.dgg_data[5*le+12,2],mwInstance.dgg_int(mwInstance.dist_datgg[5],mwInstance.eng_datgg[12]))



