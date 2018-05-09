#!/u/th/hvogel/.local/bin/python3.6
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
#This file defines the parameters of the galaxies that emit the IceCube neutrinos and the corresponding photons

"""
Parameters
L	magnetic field's coherence length at z=0 in kpc
dis 	size of galaxy at z=0 in kpc
ne0	electron number density at z=0 in cm-3
T0	Initial photon fraction. The ALP fraction is Ta = 1 - T0
z0	First redshift bin
zmax 	Maximal redishift to which galaxies are considered
zstep 	step size in arange(z0,zmax,zstep)

Functions
GammaInt    Absorption interpolation function
DispInt     Dispersion interpolation function
evo         Redshift evolution model
ne          Electron evolution model of host galaxies with redshift
"""
import numpy as np
from scipy import interpolate

#Constants
L = 1 		#magnetic field's coherence length at z=0 in kpc
dis = 7 	#size of galaxy at z=0 in kpc
ne0 = 3 	#electron number density at z=0 in cm-3
T0 = 1		#initial photon fraction. The ALP fraction is  Ta = 1-T0
z0 = 0.		#First redshift bin
zmax = 6	#Maximal redishift to which galaxies are considered
zstep = 0.1	#step size in arange(z0,zmax,zstep)

#Functions
#Absorption:
modelGamma="Galaxy_Models/Gamma_Schober_normal_YukselGRB.dat" #change this to use a different absorption model, model: [Energy [eV], redshift, Gamma [s-1]]
GammaDataRaw=np.loadtxt(modelGamma) #load data
enLDataG=np.asarray(np.log10(sorted(list(set(GammaDataRaw[:,0])))))-12 #extract energy data. Convert to logarithmic form and to TeV
zzDataG=np.asarray(sorted(list(set(GammaDataRaw[:,1])))) #redshift data
GDataG=GammaDataRaw[:,2].reshape(len(enLDataG),len(zzDataG)) #reshape Gamma
GammaInt=interpolate.RectBivariateSpline(enLDataG,zzDataG, GDataG,kx=1,ky=1) #Interpolate with linear spline

#load dispersion data
modelChi = "Galaxy_Models/norm_chi_Schober_normal_YukselGRB.dat" #Change this to use a different photon-photon dispersion model
DispDataRaw=np.loadtxt(modelChi) #load data
enLDataD=np.asarray(np.log10(sorted(list(set(DispDataRaw[:,0])))))-12 #extract energy data. Convert to logarithmic form and to TeV
#second entry redhift
zzDataD=np.asarray(sorted(list(set(DispDataRaw[:,1])))) #redshift data
DDataD=DispDataRaw[:,2].reshape(len(enLDataD),len(zzDataD))
DispInt=interpolate.RectBivariateSpline(enLDataD,zzDataD,DDataD,kx=1,ky=1) #Interpolate with linear spline

#Galaxy evolution
def evo(zz):#Evolution model by Yuksel and Kistler
    return ((1.+zz)**(-34.)+((1.+zz)/5160.64)**3+((1.+zz)/9.06)**35.)**(-1./10.)

def ne(zz): #electron density evolution following Schober
    return ne0*(1.+zz)**(3-2.14)*evo(zz)**(1./1.4)

