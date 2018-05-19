#! /usr/bin/env python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

"""
This script converts the conversion probabilities in Data/ to fluxes saved in Flux/

Functions
flux_wgt        Multiplies the flux from galaxy at redshift z and energy E 'normInt' with the ALP fraction 'da'.\
                Function of redshift z, photon energy at redshift z=0 in [GeV], and flux object.
flux_integrate  Integrand that will be integrated over redshift z to give the full flux at the Milky Way. Function\
                of photon energy at redshift z=0 in [GeV] and flux object.

Variables
options         List with the currently implemented options for the neutrino flux models.
"""

#--- IMPORTS --------
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import sys
sys.path.insert(0, '../Parameters/')
sys.path.insert(0, 'Galaxy_Models/')
import galaxy_specs as gp
import constants as cst
import cosmology as co
import neutrino_model as nm
import parameters as para
#--------------------

pi = np.pi

#Check if the right amount of arguments are supplied. Exit if not. 
#Arguments are #1 magnetic field strength in muG, photon-ALP coupling in 10^-11 GeV-1, ALP mass in log_10 (neV)
if len(sys.argv)<4:
    print("Not enough arguments \n")
    sys.exit()
if len(sys.argv)>4:
    print("Too many arguments \n")
    sys.exit()

#Read arugments
magS = sys.argv[1]
gagS = sys.argv[2]
massS = sys.argv[3]

gm_name = para.galaxy_model # Name of galaxy model defined in parameters.py
gm_options = gp.galaxy_options
if gm_name not in gm_options:
    print('Model not known. Please choose one of:\n', gm_options)
    print('Aborting...')
    sys.exit()
gm = gp.galaxyModel(gm_name)

fm_name = para.flux_model
fm_options = nm.flux_options
if fm_name not in fm_options:
    print('Model not known. Please choose one of:\n', fm_options)
    print('Aborting...')
    sys.exit()

flux = nm.ICFlux(fm_name,gm)    #make flux object

#load ALP data
daRaw=np.loadtxt('Data/da_B'+magS+'_g'+gagS+'_m'+massS+'_r'+gm_name+'.dat')

#interpolate data
#Extract energy array
enLDat=np.asarray(np.log10(sorted(list(set(daRaw[:,0])))))
#second entry redshift
zzDat=np.asarray(sorted(list(set(daRaw[:,1]))))
daDat=daRaw[:,2].reshape(len(enLDat),len(zzDat)) #reshape data array
daInt=interpolate.RectBivariateSpline(enLDat,zzDat,daDat,kx=1,ky=1)#interpolate axion fraction da

def flux_wgt(zz,Eg,flux_class):#Weighted flux from galaxy at redshift z as a function of redshift, photon energy Eg [GeV], and break energy
    return flux_class.normIntPh(zz,Eg,gm)*daInt(np.log10(Eg)-3,zz) #here the argument of daInt converts the energy in GeV to the log10 energy in TeV

def flux_integrate(Eg,flux_class):#function that return total flux at the Milky Way as a function of redshift z=0 photon energy, Eg in [GeV]
    pref=1./3.*Eg**2*cst.cv/(4.*pi)*flux_class.normC #normalization and conversion to GeV cm-2 s-1 sr-1
    res, err = quad(flux_wgt,0,gp.zmax,args=(Eg,flux_class,),limit=300, epsrel=10**(-4)) #integration
    return res*pref

data =[]
for EgL in enLDat:
    data.append([10**3*10**EgL,flux_integrate(10**3*10**EgL,flux)])#convert EgL energy, which is in TeV to GeV
    
np.savetxt('Flux/ALPflux_B'+magS+'_g'+gagS+'_m'+massS+'_f'+fm_name+'_r'+gm_name+'.dat',data,fmt='%.4e')
