# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
import numpy as np
from scipy import interpolate

"""
Defines parameters for the propagation in the Milky Way

Parameters
ne	    electron number density [cm-3]
mw_options  Options for the Milky Way model. Currently only VernettoLipari model is implemented

Functions
gamma_int   Interpolation function for absorption of photons as a function of [distance [kpc] , log(energy) [eV]]
chi_int     Interpolation function for photon-photon dispersion as function of [distance [kpc], log(energy) [eV]]
"""

ne = 1 #electron number density [cm-3]
mw_options = ['VernettoLipari']

class mwModel:

    def __init__(self,mwModel,bstring,lstring):
        if mwModel == 'VernettoLipari':
            self.modelMW_Gamma = "Gamma/gamma_"+bstring+"_"+lstring+".dat"
            self.modelMW_Chi = "Chi/chi_"+bstring+"_"+lstring+".dat"
    
    #load absorption
        self.gamma_data=np.loadtxt(self.modelMW_Gamma) #load data
        self.dist_dat=np.asarray(sorted(list(set(self.gamma_data[:,1]))))#extract distance data
        self.eng_dat=np.asarray(np.log10(sorted(list(set(self.gamma_data[:,0])))))#extract energy data. Convert to logarithmic form
        self.gamma_data_reshaped=np.asarray((self.gamma_data[:,2]).reshape(len(self.dist_dat),len(self.eng_dat)))
        self.gamma_int=interpolate.RectBivariateSpline(self.dist_dat,self.eng_dat,self.gamma_data_reshaped,kx=1,ky=1)#Checked correct implementation on 08/09/2018

        #load delta_gamma_gamma
        self.dgg_data=np.loadtxt(self.modelMW_Chi)
        self.dist_datgg=np.asarray(sorted(list(set(self.dgg_data[:,1]))))
        self.eng_datgg=np.asarray(np.log10(sorted(list(set(self.dgg_data[:,0])))))
        self.dgg_data_reshaped = np.asarray((self.dgg_data[:,2]).reshape(len(self.dist_datgg),len(self.eng_datgg)))
        self.dgg_int=interpolate.RectBivariateSpline(self.dist_datgg,self.eng_datgg,self.dgg_data_reshaped,kx=1,ky=1)

    def fgamma(self,d,omega):
        if d<=0.5:#We cut off at 0.5 since the first bin is special. If we interpolated, the local absorption rate would go to zero at d=0
            return self.gamma_int(0.5,np.log10(omega)+12) #omega is in TeV but the argument is in eV. 
        return self.gamma_int(d,np.log10(omega)+12)

