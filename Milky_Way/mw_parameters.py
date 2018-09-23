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
            return self.dgg_int(0.5,np.log10(omega)+12) #conversion to eV
        return self.dgg_int(d,np.log10(omega)+12) #conversion to eV


class mwMagneticField:

    def __init__(self,mfModel):
        if mfModel == 'JanssonFarrar': #JanssonFarrar model following arxiv:1204.3662
            self.iDeg=11.5 #opening angle of the spiral in degrees 
            self.rList=np.asarray([5.1,6.3,7.1,8.3,9.8,11.4,12.7,15.5]) #radii [kpc] where the spirals cross the negative x-axis
            self.bList=np.asarray([0.1,3.0,-0.9,-0.8,-2.0,-4.2,0,2.7]) #value of magnetic fields [muG] in spiral
            self.PhiList=np.asarray([-3*np.pi+2*ii*np.pi for ii in range(0,6)]) #List of phi-angles in rad

            def sortF(item):#sorting function to select certain element
                return item[0]
            self.sortF =sortF

            def fBDisk(rDisk,x,y):# Disk magnetic field as a function of the radius in the disk [rDisk [kpc], x [kpc], y [kpc]]
                if rDisk<3: # inner core is set to zero
                    return np.asarray([0,0,0])
                Phi=np.arctan2(y/rDisk,x/rDisk) # angle inside disk
                if Phi<0: #coordinate transformation from [-pi,pi] to [0,2pi]
                    Phi=Phi+2*np.pi
                if rDisk<=5: #molecular radius
                    Bring=0.1 #molecular magnetic field in [muG]
                    return np.asarray([-Bring*np.sin(Phi),Bring*np.cos(Phi),0]) #purely azimuthal magnetic field
                if rDisk>20:# we set the magnetic field to zero at the edge of the galaxy
                    return np.asarray([0,0,0])
                spiralList=[[self.rList[k]*np.exp((self.PhiList[j]+Phi)*np.tan(np.pi/180*self.iDeg)),int(k+1)]\
                        for j in range(0,len(self.PhiList)) for k in range(0,len(self.rList))]#each spiral has multiple values crossings a line from the center of the galaxy outwards.. Every crossing in the direction of Phi of any spiral is enumerated here.
                spiralList.append([rDisk,0]) #dummy value that we want to find. It has the radius where the mf is defined in the first argument. 
                spiralList.sort(key=self.sortF) #sort according to the radius.
                for s in range(0,len(spiralList)):
                    if spiralList[s][1]==0:
                        ss=spiralList[s+1][1]-1 #get correct spiral arm index
                return 5.*self.bList[ss]/rDisk*np.asarray([np.sin(self.iDeg*np.pi/180)*np.cos(Phi)\
                        -np.cos(self.iDeg*np.pi/180)*np.sin(Phi), np.sin(self.iDeg*np.pi/180)*np.sin(Phi)\
                        +np.cos(self.iDeg*np.pi/180)*np.cos(Phi),0])
            self.fBDisk = fBDisk

            def fBtor (rDisk,x,y,z):#toroidal magnetic field
                z0=5.3
                h=0.4
                w=0.27
                Phi=np.arctan2(y/rDisk,x/rDisk)
                Bn=1.4
                Bs=1.1 # the negative sign is contained in the final return statement for the southern hemisphere
                diskExp=np.exp(-abs(z)/z0) #exponential fall-off in z-direction
                L1=1./(1+np.exp(-2*(abs(z)-h)/w)) #transition between disk and halo field
                Ln=(1-1./(1+np.exp(-2*(rDisk-9.22)/0.2))) #cut-off of the northern halo field
                Ls=(1-1/(1+np.exp(-2*(rDisk-16.7)/0.2))) #cut-off of the southern halo field
                if Phi<0:
                    Phi=Phi+2*np.pi
                if z>=0:
                    return Bn*diskExp*L1*Ln*np.asarray([-np.sin(Phi),np.cos(Phi),0])
                return Bs*diskExp*L1*Ls*np.asarray([np.sin(Phi),-np.cos(Phi),0])
            self.fBtor = fBtor

            def fBx (rDisk,x,y,z):#halo magnetic field as a function of disk radius, and x,y,z coordinates
                Bx=4.6
                rx=2.9
                rxc=4.8
                Theta0=49 #in degrees
                Phi=np.arctan2(y/rDisk,x/rDisk)
                rp=rxc*rDisk/(rxc+abs(z)/np.tan(Theta0*np.pi/180))
                if Phi<0:
                    Phi=Phi+2*np.pi
                if rp>rxc:
                    rp=rDisk-abs(z)/np.tan(Theta0*np.pi/180)
                    bx=Bx*np.exp(-rp/rx)
                    Btot=bx*rp/rDisk
                    if z>=0:
                        if z==0:
                            return Btot*np.asarray([np.sin(0*np.pi/180)*np.cos(Phi),np.sin(0*np.pi/180)*np.sin(Phi),np.cos(0*np.pi/180)])
                        else:
                            return Btot*np.asarray([np.cos(Theta0*np.pi/180)*np.cos(Phi),np.cos(Theta0*np.pi/180)*np.sin(Phi),                                    np.sin(Theta0*np.pi/180)])
                    return Btot*np.asarray([-np.cos(Theta0*np.pi/180)*np.cos(Phi),-np.cos(Theta0*np.pi/180)*np.sin(Phi),                                np.sin(Theta0*np.pi/180)])
                else:
                    rp=rxc*rDisk/(rxc+abs(z)/np.tan(Theta0*np.pi/180))
                    bx=Bx*np.exp(-rp/rx)
                    Btot=bx*(rp/rDisk)**2
                    if abs(z**2+(rDisk-rp)**2)<10**(-5):#numerical cutoff
                        ThetaX=0
                    else:
                        ThetaX=np.arctan2(rDisk-rp,abs(z))
                    if z>=0:
                        return Btot*np.asarray([np.sin(ThetaX)*np.cos(Phi),np.sin(ThetaX)*np.sin(Phi),np.cos(ThetaX)])# checked correctness of this line with definition of ThetaX
                    else:
                        return Btot*np.asarray([-np.sin(ThetaX)*np.cos(Phi),-np.sin(ThetaX)*np.sin(Phi),np.cos(ThetaX)])

                self.fBx = fBx

            def fB(rE,b,l):#full magnetic field as a function of distance from earth [rE [kpc], b latitude, l longitude]
                z=rE*np.sin(b*np.pi/180)
                rX=rE*np.cos(b*np.pi/180) #radius in plane from earth
                x=rX*np.cos((l*np.pi)/180)-8.5 # we assume distance of Earth from galactic center = 8.5 kpc
                y=rX*np.sin((l*np.pi)/180)
                rDisk=np.sqrt(x**2+y**2)#disk radius
                rCenter=np.sqrt(x**2+y**2+z**2) #radius from center of Galaxy
                if rCenter<1 or rCenter>20:# we switch of the mf in the galactic center and 20kpc away from the center
                    return np.asarray([0,0,0])
                else:
                    h=0.4
                    w=0.27
                    BDisk=fBDisk(rDisk,x,y)*(1-1/(1+np.exp(-2*(abs(z)-h)/w))) #mupliply disk magnetic field by smooth function to get transition to toroidal halo
                    Btor=fBtor(rDisk,x,y,z)
                    Bx=fBx(rDisk,x,y,z)
                return BDisk+Btor+Bx
            self.fB = fB

        if mfModel == 'Pshirkov':
            print("Model not yet implemented. Aborting...")
            sys.exit()


