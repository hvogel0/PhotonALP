# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
import numpy as np
from scipy import interpolate

"""
Defines parameters for the propagation in the Milky Way

Parameters
ne	    electron number density [cm-3]
mw_radiation_options  Options for the Milky Way model. Currently only VernettoLipari model is implemented

Functions
gamma_int   Interpolation function for absorption of photons as a function of [distance [kpc] , log(energy) [eV]]
chi_int     Interpolation function for photon-photon dispersion as function of [distance [kpc], log(energy) [eV]]
fgamma      Local absorption rate [kpc-1] as a function of [distance [kpc], energy [TeV]]
fdgg        Photon photon dispersion [kpc-1] as a function of [distance [kpc], energy [TeV]]

Classes
mwRadiationModel    Defines the Milky Way's radiation model.. Its shape and spectrum determines\
                    the local absorption rate and the photon-photon dispersion
mwMagneticModel     Defines the Milky Way's magnetic field model.

"""

ne = 1 #electron number density [cm-3]
mw_radiation_options = ['VernettoLipari']
mw_mf_options = ['JanssonFarrar','Pshirkov']

class mwRadiationModel:

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

    def fgamma(self,d,omega):#local absorption rate [kpc-1] as a function of distance d [kpc] and energy omega [TeV]
        if d<=0.5:#We cut off at 0.5 since the first bin is special. If we interpolated, the local absorption rate would go to zero at d=0
            return self.gamma_int(0.5,np.log10(omega)+12) #omega is in TeV but the argument is in eV. 
        return self.gamma_int(d,np.log10(omega)+12)
    
    def fdgg(self,d,omega):#photon-photon dispersion [kpc-1] as a function of distance d[kpc] and energy omega [TeV]
        if d<=0.5:# We cut off at 0.5 since the first bin is special. If we interpolated, the local absorption rate would go to zero at d=0
            return dgg_int(0.5,np.log10(omega)+12) #conversion to eV
        return dgg_int(d,np.log10(omega)+12) #conversion to eV


class mwMagneticField:

    def __init__(self,mfModel):
        if mfModel == 'JanssonFarrar': #JanssonFarrar model following arxiv:1204.3662
            self.iDeg=11.5 #opening angle of the spiral in degrees 
            self.rList=np.asarray([5.1,6.3,7.1,8.3,9.8,11.4,12.7,15.5]) #radii [kpc] where the spirals cross the negative x-axis
            self.bList=np.asarray([0.1,3.0,-0.9,-0.8,-2.0,-4.2,0,2.7]) #value of magnetic fields [muG] in spiral
            self.PhiList=np.asarray([-3*np.pi+2*ii*np.pi for ii in range(0,6)]) #List of phi-angles in rad

            def fBDisk(rDisk,x,y):# Disk magnetic field as a function of the radius in the disk [rDisk [kpc], x [kpc], y [kpc]]
                if rDisk<3: # inner core is set to zero
                    return np.asarray([0,0,0])
                Phi=np.arctan2(y/rDisk,x/rDisk) # angle inside disk
                if Phi<0: #coordinate transformation from [-pi,pi] to [0,2pi]
                    Phi=Phi+2*np.pi
                if rDisk<=5: #molecular radius
                    Bring=0.1 #molecular magnetic field in [muG]
                    return np.asarray([-Bring*np.sin(Phi),Bring*np.cos(Phi),0]) #purely azimuthal magnetic field
    if rDisk>20:
        return np.asarray([0,0,0])
    spiralList=[[rList[k]*np.exp((PhiList[j]+Phi)*np.tan(np.pi/180*iDeg)),int(k+1)] for j in range(0,len(PhiList))                for k in range(0,len(rList))]
    spiralList.append([rDisk,0])
    spiralList.sort(key=sortF)
    for s in range(0,len(spiralList)):
        if spiralList[s][1]==0:
            ss=spiralList[s+1][1]-1
    return 5*bList[ss]/rDisk*np.asarray([np.sin(iDeg*np.pi/180)*np.cos(Phi)-np.cos(iDeg*np.pi/180)*np.sin(Phi),                                         np.sin(iDeg*np.pi/180)*np.sin(Phi)+np.cos(iDeg*np.pi/180)*np.cos(Phi),0])

            
        if mfModel == 'Pshirkov':


    def sortF(item):#sorting function to select certain element
        return item[0]

