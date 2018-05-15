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
zmax 	Maximal redshift to which galaxies are considered
zstep 	step size in arange(z0,zmax,zstep)
B_var   Variance of magnetic field. We assume here that the magnetic field components (B_x, B_y, B_z)\
        each follow a Gaussian distribution with zero mean and variance B_var

Classes
galaxyModel Class that defined the model for the host galaxies. Currently 'Schober' and 'Yuksel' models are implemented.

"""
import numpy as np
import sys
from scipy import interpolate

#Constants
L = 1 		#magnetic field's coherence length at z=0 in kpc
dis = 7 	#size of galaxy at z=0 in kpc
ne0 = 3 	#electron number density at z=0 in cm-3
T0 = 1		#initial photon fraction. The ALP fraction is  Ta = 1-T0
z0 = 0.		#First redshift bin
zstep = 0.1	#step size in arange(z0,zmax,zstep)
zmax = 6	#Maximal redishift to which galaxies are considered
B_var = 2./3.   #Variance of magnetic field model.

galaxy_options = ['Schober','Yuksel']

class galaxyModel:
    """
    Functions
    GammaInt    Absorption interpolation function
    DispInt     Dispersion interpolation function
    evo         Redshift evolution model
    ne          Electron evolution model of host galaxies with redshift
    magZ        Absolute value of magnetic field as a function fo normalization mag [muG] and redshift.\
            The implementation follows Schober.
    """
    def __init__(self,galaxy_model):
        if galaxy_model == 'Schober':
            self.modelGamma="Galaxy_Models/Gamma_Schober_normal.dat" #absorption model, model: [Energy [eV], redshift, Gamma [s-1]]
            self.modelChi = "Galaxy_Models/norm_chi_Schober_normal.dat" #photon-photon dispersion model
            
            #Galaxy evolution
            def evo(self,zz):#Redshift evolution of host galaxies
                k1=3./5.
                k2=14./15.
                zm=5.4
                return k2*np.exp(k1*(zz-zm))/(k2-k1+k1*np.exp(k2*(zz-zm)))*(1+zz)**3
            self.evo = evo 

        if galaxy_model == 'Yuksel':
            self.modelGamma="Galaxy_Models/Gamma_Schober_normal_YukselGRB.dat" #absorption model, model: [Energy [eV], redshift, Gamma [s-1]]
            self.modelChi = "Galaxy_Models/norm_chi_Schober_normal_YukselGRB.dat" #photon-photon dispersion model
            
            #Galaxy evolution
            def evo(self,zz):#Redshift evolution of host galaxies
                return ((1.+zz)**(-34.)+((1.+zz)/5160.64)**3+((1.+zz)/9.06)**35.)**(-1./10.)
            self.evo = evo

        #Absorption
        self.GammaDataRaw=np.loadtxt(self.modelGamma) #load data
        self.enLDataG=np.asarray(np.log10(sorted(list(set(self.GammaDataRaw[:,0])))))-12 #extract energy data. Convert to logarithmic form and to TeV
        self.zzDataG=np.asarray(sorted(list(set(self.GammaDataRaw[:,1])))) #redshift data
        self.GDataG=self.GammaDataRaw[:,2].reshape(len(self.enLDataG),len(self.zzDataG)) #reshape Gamma
        self.GammaInt=interpolate.RectBivariateSpline(self.enLDataG,self.zzDataG, self.GDataG,kx=1,ky=1) #Interpolate with linear spline

        #Dispersion
        self.DispDataRaw=np.loadtxt(self.modelChi) #load data
        self.enLDataD=np.asarray(np.log10(sorted(list(set(self.DispDataRaw[:,0])))))-12 #extract energy data. Convert to logarithmic form and to TeV
        #second entry redhift
        self.zzDataD=np.asarray(sorted(list(set(self.DispDataRaw[:,1])))) #redshift data
        self.DDataD=self.DispDataRaw[:,2].reshape(len(self.enLDataD),len(self.zzDataD))
        self.DispInt=interpolate.RectBivariateSpline(self.enLDataD,self.zzDataD,self.DDataD,kx=1,ky=1) #Interpolate with linear spline

    def ne(self,zz): #electron density evolution following Schober
        return ne0*(1.+zz)**(3-2.14)*self.evo(self,zz)**(1./1.4)

    def magZ(self,mag,zz):
        return mag*(self.ne(zz)/self.ne(0))**(1./6.)*(self.evo(self,zz)/(1.+zz))**(1./3.)
