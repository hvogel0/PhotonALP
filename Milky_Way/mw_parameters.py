# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

"""
Defines parameters for the propagation in the Milky Way

Parameters
ne	electron number density [cm-3]

Functions
fgamma   Absorption rate 
"""
#--- IMPORTS -----------
import numpy as np
import sys
from scipy import interpolate
#-----------------------
ne = 1 #electron number density [cm-3]
mw_photon_options = ['Schober']
mw_mag_options = ['JF', 'Pshirkov']

def sortF(item):#sorting function
    return item[0]

class MWModel:
    """
    Model of the Milky Way

    Functions
    fgamma      Absorption rate in kpc-1 as a function of distance from the Sun [kpc] and energy [TeV]
    fdgg        Photon photon dispersion in kpc-1 as a function of distance from the Sun [kpc] and energy [TeV]


    """
    def __init__(self, photon_model,bstring,lstring):#bstring latitude, lstring longitude
        if photon_model == 'Schober':
            self.gammaDir = "Gamma/gamma_"+bstring+"_"+lstring+".dat" #directory of absorption data
            self.dispDir = "Chi/chi_"+bstring+"_"+lstring+".dat" #directory of dispersion data

        self.GammaDataRaw = np.loadtxt(self.gammaDir) #load absorption data
        self.enLDataG=np.asarray(np.log10(sorted(list(set(self.GammaDataRaw[:,0])))))-12 #extract energy and convert eV to TeV
        self.distDataG=np.asarray(sorted(list(set(self.GammaDataRaw[:,1]))))    #extract distance data
        self.GammaDataReshape=np.asarray((self.GammaDataRaw[:,2]).reshape(len(self.distDataG),len(self.enLdataG))) #reshape absorption data
        self.tauInt=interpolate.RectBivariateSpline(self.distDataG,self.enLdataG,self.GammaDataReshape,kx=1,ky=1) #interpolate with linear spline

        def fgamma(self,d,omega):#absorption rate as a function of distance [kpc] from the sun, and energy in [TeV]
            if d <=0.5: # the data contains piece-wise constant optical depth. We let the last step have a constant rate
                return self.tauInt(0.5,np.log10(omega))
            return self.tauInt(d,np.log10(omega))
        self.fgamma = fgamma

        self.DispDataRaw = np.loadtxt(self.dispDir) #load dispersion data
        self.enLDataD=np.asarray(np.log10(sorted(list(set(self.DispDataRaw[:,0])))))-12 #extract energy and convert from eV to TeV
        self.distDataD=np.asarray(sorted(list(set(self.DispDataRaw[:,1]))))    #extract distance data
        self.DispDataReshape=np.asarray((self.DispDataRaw[:,2]).reshape(len(self.distDataD),len(self.enLdataD))) #rehape dispersion data
        self.dispInt=interpolate.RectBivariateSpline(self.distDataD,self.enLdataD,self.DispDataReshape,kx=1,ky=1) #interpolate with linear spline
        
        def fdgg(self,d,omega):#dispersion [kpc-1] as a function of distance [kpc] from the sun, and energy in [TeV]
            if d <=0.5:
                return self.dispInt(0.5,np.log10(omega))
            return self.dispInt(d,np.log10(omega))
        self.fdgg = fdgg


class magModel:
    """
    Implementation of the magnetic fields models. Currently Jansson and Farrar (JF) and Pshirkov (Pshirkov) is implemented
    Parameters
    iDeg    Pitch angle of magnetic spiral
    rList   Radii of spirals in [kpc]
    bList   Magnetic fields of spirals in [muG]

    Functions
    BDisk   Disk magnetic field as a function of disk radius [kpc], x coordinate [kpc] and y coordinate [kpc]
    """
    def __init__(self,mag_model):
        if mag_model =='JF':#according to Jansson and Farrar 1204.3662
        
            def BDisk(self,rDisk,x,y):#Disk
                iDeg = 11.5 #pitch angle of the magnetic field spiral
                rList=np.asarray([5.1,6.3,7.1,8.3,9.8,11.4,12.7,15.5]) #radius list of spirals in [kpc]
                bList=np.asarray([0.1,3.0,-0.9,-0.8,-2.0,-4.2,0,2.7])   #magnetic field list of spirals in [muG]
                PhiList=np.asarray([-3*np.pi+2*ii*np.pi for ii in range(0,6)])
                if rDisk < 3:#cut out galactic center
                    return np.asarray([0.,0.,0.])
                Phi = np.arctan2(y/rDisk,x/rDisk)
                if Phi <0: #Phi is defined to be in the positive range
                    Phi = Phi+2.*np.pi
                if rDisk<=5:
                    Bring = 0.1 # moldecular ring magnetic field for <5 kpc
                    return np.asarray([-Bring*np.sin(Phi),Bring*np.cos(Phi),0])
                if rDisk>20:#we set the magnetic field to 0 far away from the galaxy
                    return np.asarray([0.,0.,0.])
                spiralList=[[rList[k]*np.exp((PhiList[j]+Phi)*np.tan(np.pi/180*iDeg)),int(k+1)] for j in range(0,len(PhiList)) for k in range(0,len(rList))]#set up set that has coodinates of the spiral in the given direction
                spiralList.append([rDisk,0])
                spiralList.sort(key=sortF) #sort
                for s in range(0,len(spiralList)):
                    if spiralList[s][1]==0:#found the spiral arm
                        ss=spiralList[s+1][1]-1 #magnetic field of the spiral arm
                r0=5 #radius where spiral B-fields are normalized
                return bList[ss]*r0/rDisk*np.asarray([np.sin(iDeg*np.pi/180)*np.cos(Phi)-np.cos(iDeg*np.pi/180)*np.sin(Phi),np.sin(iDeg*np.pi/180)*np.sin(Phi)+np.cos(iDeg*np.pi/180)*np.cos(Phi),0])
            self.BDisk = BDisk
            
            def Btor(rDisk,x,y,z):#toroidal magnetic field as a function of disk radius rDisk [kpc], x-coordinate [kpc], y-coordinate [kpc], and z-coordinate [kpc]
                z0=5.3 #scale height in kpc
                h=0.4   #
                w=0.27
                Phi=np.arctan2(y/rDisk,x/rDisk) #angle
                Bn=1.4 #northern hemisphere magnetic field in [muG]
                Bs=1.1 #southern hemisphere magnetic field in [muG]
                diskExp=np.exp(-abs(z)/z0)
                L1=1./(1+np.exp(-2*(abs(z)-h)/w))   #fall-off in z-direction
                Ln=(1-1./(1+np.exp(-2*(rDisk-9.22)/0.2))) #northern fall-off in disk direction
                Ls=(1-1/(1+np.exp(-2*(rDisk-16.7)/0.2)))    #southern fall-off in disk direction
                if Phi<0:
                    Phi=Phi+2*np.pi
                if z>=0: #northern hemisphere?
                    return Bn*diskExp*L1*Ln*np.asarray([-np.sin(Phi),np.cos(Phi),0]) #northern hemisphere?
                return Bs*diskExp*L1*Ls*np.asarray([np.sin(Phi),-np.cos(Phi),0])
            self.Btor = Btor

            def Bx (rDisk,x,y,z):#X-halo as a function of disk radius [kpc], x-coordinate [kpc], y-coordinate [kpc], z-coordinate [kpc]
                Bx=4.6 #halo magnetic field strength in [muG]
                rx=2.9 #transition radius in [kpc]
                rxc=4.8 #transition radius 2 [kpc]
                Theta0=49 #in degrees
                Phi=np.arctan2(y/rDisk,x/rDisk)
                rp=rxc*rDisk/(rxc+abs(z)/np.tan(Theta0*np.pi/180))
                if Phi<0:
                    Phi=Phi+2*np.pi
                if rp>rxc:
                    rp=rDisk-abs(z)/np.tan(Theta0*np.pi/180)
                    bx=Bx*np.exp(-rp/rx)
                    Btot=bx*rp/rDisk
                    if z>=0: #northern hemisphere?
                        if z==0:
                            return Btot*np.asarray([np.sin(0*np.pi/180)*np.cos(Phi),np.sin(0*np.pi/180)*np.sin(Phi),np.cos(0*np.pi/180)])
                        else:
                            return Btot*np.asarray([np.cos(Theta0*np.pi/180)*np.cos(Phi),np.cos(Theta0*np.pi/180)*np.sin(Phi),                                    np.sin(Theta0*np.pi/180)])
                    return Btot*np.asarray([-np.cos(Theta0*np.pi/180)*np.cos(Phi),-np.cos(Theta0*np.pi/180)*np.sin(Phi),                                np.sin(Theta0*np.pi/180)])
                else:
                    rp=rxc*rDisk/(rxc+abs(z)/np.tan(Theta0*np.pi/180))
                    bx=Bx*np.exp(-rp/rx)
                    Btot=bx*(rp/rDisk)**2
                    if abs(z**2+(rDisk-rp)**2)<10**(-5):#numerical cut-off
                        ThetaX=0
                    else:
                        ThetaX=np.arctan2(rDisk-rp,abs(z))
                    if z>=0:#northern hemisphere?
                        return Btot*np.asarray([np.sin(ThetaX)*np.cos(Phi),np.sin(ThetaX)*np.sin(Phi),np.cos(ThetaX)])
                    else:
                        return Btot*np.asarray([-np.sin(ThetaX)*np.cos(Phi),-np.sin(ThetaX)*np.sin(Phi),np.cos(ThetaX)])

