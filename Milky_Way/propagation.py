# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import sys
sys.path.insert(0, '../Parameters/')
import constants as cst
import mw_parameters as mw

"""
Defines the Hamiltonian and other function used for the propataion of the photon-ALP system.

Parameters:
npl     Normalization constant for plasma frequency
na      Normalization of the ALP mass
nB      Normalization for magnetic birefringence
nag     Normalization for photon-ALP coupling
nGG     Normalization for photon-photon dispersion
ne      electron density from mw_parameters

Functions:
H       Hamiltonian matrix as a function of distance from Earth 'd' [kpc], photon energy 'omega' [TeV], mass of the ALP 'ma' [neV], latitude 'b', longitude 'l', a radiation model 'radModel' and a magnetic field model 'mfModel'
integratedH     Integrated Hamiltonian for the numerical difficult parts as a function of distance from Earth 'd' [kpc], photon energy 'omega' [TeV], mass of the ALP 'ma' [neV], latitude 'b', longitude 'l', a radiation model 'radModel' and a magnetic field model 'mfModel'


"""
#normalizations
npl=cst.npl
na=cst.na
nB=cst.nB
nag=cst.nag
nGG=cst.nGG
ne = mw.ne



#functions
#Source terms from cosmic ray proton interactions
def Qintegrand(Ep, omega,CRModel,DustModel): #integrand of source term. Arguments: Ep: proton energy [TeV], omega: photon energy [TeV]
    x=omega/Ep
    lE=np.log(Ep)
    return sigma(lE)*10**(-1)*CRModel.H3AEquivalentAllProtonFlux(10**3*Ep)*DustModel.F(x,lE)/Ep #note that the proton energy put into the cosmic ray flux is in GeV

def Q(rE,omega, b, l,CRModel,DustModel): #Source term as a function of distance from Earth rE [kpc-1], photon energy omega [TeV]
    z=rE*np.sin(b*np.pi/180)
    rX=rE*np.cos(b*np.pi/180) #radius in the disk as seen from Earth
    x=rX*np.cos(l*np.pi/180)-8.5
    y=rX*np.sin(l*np.pi/180)
    rDisk=np.sqrt(x**2+y**2)
    rateInt, err=quad(Qintegrand,omega,np.infty,args=(omega,CRModel,DustModel,),limit=100)#rate per proton
    dNdEcgs=DustModel.nH(rDisk,z)*rateInt #rateDensity
    dNdEnatural=dNdEcgs
    dNdE=10**(10)*omega**3*(2*np.pi)**3*dNdEnatural/cst.cmTokpc #multiplying by several normalization factors to facilitate numerical integration
    return omega*dNdE


def H(d,omega,gag,ma,b,l,radModel,mfModel):#Hamiltonian matrix as a function of function of distance from Earth 'd' [kpc], photon energy 'omega' [TeV], mass of the ALP 'ma' [neV], latitude 'b', longitude 'l', a radiation model and a magnetic field model 'mfModel'
    BVec=mfModel.fB(d,b,l)
    #define dreibein of vectors. kUnit is the vector that points in the direction of the photon propagation. e1 and e2 are orthogonal to kUnit. e2 lies in the xy-plane.
    kUnit=np.asarray([np.cos(np.pi/180.*b)*np.cos(np.pi/180.*l),np.cos(np.pi/180.*b)*np.sin(np.pi/180.*l),np.sin(np.pi/180.*b)])
    e1=np.asarray([-np.sin(np.pi/180.*b)*np.cos(np.pi/180.*l),-np.sin(np.pi/180.*b)*np.sin(np.pi/180.*l),np.cos(np.pi/180.*b)]);
    e2=np.asarray([-np.sin(np.pi/180.*l),np.cos(np.pi/180.*l),0.]);

    BT=BVec-np.dot(kUnit,BVec)*kUnit #Project out components that are parallel to the photon's direction of motion
    Bnorm=np.linalg.norm(BT)
    if Bnorm >10**(-12): #numerical cut-off
        cphi=np.dot(BT,e1)/Bnorm #cos phi
        sphi=np.dot(BT,e2)/Bnorm #sin phi
        dgg =(radModel.fdgg(d,omega))[0,0] # get dispersion 
        gamma =(radModel.fgamma(d,omega))[0,0] # get absorption
        h11=ne/omega*npl+nB*omega*Bnorm**2*(3.5*cphi**2+2*sphi**2)+dgg-1j*gamma/2
        h22=ne/omega*npl+nB*omega*Bnorm**2*(3.5*sphi**2+2*cphi**2)+dgg-1j*gamma/2
        h33=ma**2/omega*na
        h12=1.5*nB*omega*Bnorm**2*sphi*cphi
        h13=nag*cphi*Bnorm*gag
        h23=nag*sphi*Bnorm*gag
    else: #small magnetic field means negligible contribution to off-diagonals and birefringence
        dgg =(radModel.fdgg(d,omega))[0,0]
        gamma =(radModel.fgamma(d,omega))[0,0]
        h11=ne/omega*npl+dgg-1.j*gamma/2
        h22=ne/omega*npl+dgg-1.j*gamma/2
        h33=ma**2/omega*na
        h12=0
        h13=0
        h23=0
    l1=[h11,h12,h13]
    l2=[h12,h22,h23]
    l3=[h13,h23,h33]
    return np.asarray([l1,l2,l3])

def integratedH(d,omega,gag,ma,b,l,radModel,mfModel): # integrated Hamiltonian for the numerical diffcult parts when mode='ALP' and source_flag='False'
    h=H(d,omega,gag,ma,b,l,radModel,mfModel)
    return (abs(h[0,2])**2+abs(h[1,2])**2)/(abs(h[0,0]+h[1,1]-h[2,2]))**2

def integratedQ(d,y,omega,b,radModel,mfModel,CRModel,DustModel): # integrated Hamil
    h = H(t,omega,b,l,radModel,mfModel)
    halt=2*(h[0,0]).imag
    Qlocal=Q(t,omega, b, l,CRModel,DustModel)
    eM=-y*halt-Qlocal
    return eM

def find_initial(d0,dd,b,l,mfModel,DustModel,initialCondition): # find initial distance by finding a magnetic field that is non-negligible. There we start integration
    if initialCondition == 'ALP':
        while d0>0:
            B=mfModel.fB(d0,b,l)
            if B[0]>10**(-3):
                return d0
            else:
                if B[1]>10**(-3):
                    return d0
                else:
                    if B[2]>10**(-3):
                        return d0
                    else:
                        d0=d0-dd # if magnetic field is not big enough, step forward towards earth.
    elif initialCondition == 'None':
        while True:
            z=d0*np.sin(b*np.pi/180)
            rX=d0*np.cos(b*np.pi/180)
            x=rX*np.cos(l*np.pi/180)-8.5 #we assume Earth is 8.5 kpc from the Galactic Center
            y=rX*np.sin(l*np.pi/180)
            rDisk=np.sqrt(x**2+y**2)
            if DustModel.nH(rDisk,z) <10**(-4):
                d0=d0-dd
                if d0<0:
                    print("problem with finding initial value", d0)
                    print('Aborting...')
                    sys.exit()
            else:
                return d0-0.05
    else:
        print('initialCondition not known. Only ALP and None implemented.')
        print('Aborting...')
        sys.exit()


def fsep(d,y,omega,gag,ma,b,l,radModel,mfModel,CRModel,DustModel,source_flag): # rhs of propagation equation split in real and imaginary part
    #spilt vector y in real and imaginary parts
    if source_flag ==True:
        Qlocal = Q(d,omega,b,l,CRModel,DustModel)
    else:
        Qlocal=0
    y11=y[0]
    y22=y[1]
    y33=y[2]
    y12r=y[3]
    y12i=y[4]
    y13r=y[5]
    y13i=y[6]
    y23r=y[7]
    y23i=y[8]
    h = H(d,omega,gag,ma,b,l,radModel,mfModel)
    hr=h.real
    h11i= (h.imag)[0,0]
    h22i= (h.imag)[1,1]
    h11r=hr[0,0]
    h22r=hr[1,1]
    h33=hr[2,2]
    h12=hr[0,1]
    h13=hr[0,2]
    h23=hr[1,2]
    f11=-2*h11i*y11 + 2*h12*y12i + 2*h13*y13i - Qlocal/2
    f22=-2*h12*y12i - 2*h22i*y22 + 2*h23*y23i - Qlocal/2
    f33=-2*h13*y13i - 2*h23*y23i
    f12r=-h11r*y12i + h22r*y12i - h11i*y12r - h22i*y12r + h23*y13i + h13*y23i
    f12i=-(h12*y11) - h11i*y12i - h22i*y12i + h11r*y12r - h22r*y12r - h23*y13r + h12*y22 + h13*y23r
    f13r=h23*y12i - h11r*y13i + h33*y13i - h11i*y13r - h12*y23i
    f13i=-(h13*y11) - h23*y12r - h11i*y13i + h11r*y13r - h33*y13r + h12*y23r + h13*y33
    f23r=-(h13*y12i) - h12*y13i - h22r*y23i + h33*y23i - h22i*y23r
    f23i=-(h13*y12r) + h12*y13r - h23*y22 - h22i*y23i + h22r*y23r - h33*y23r + h23*y33
    return np.asarray([f11,f22,f33,f12r,f12i,f13r,f13i,f23r,f23i])

