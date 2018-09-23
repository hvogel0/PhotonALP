# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
import numpy as np
from scipy import interpolate
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

def integratedH(d,omega,gag,ma,b,l,radModel,mfModel): # integrated Hamiltonian for the numerical diffcult parts
    h=H(d,omega,gag,ma,b,l,radModel,mfModel)
    return (abs(h[0,2])**2+abs(h[1,2])**2)/(abs(h[0,0]+h[1,1]-h[2,2]))**2

def find_initial(d0,dd,b,l,mfModel): # find initial distance by finding a magnetic field that is non-negligible. There we start integration
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


def fsep(d,y,omega,gag,ma,b,l,radModel,mfModel): # rhs of propagation equation split in real and imaginary part
    #spilt vector y in real and imaginary parts
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
    f11=-2*h11i*y11 + 2*h12*y12i + 2*h13*y13i
    f22=-2*h12*y12i - 2*h22i*y22 + 2*h23*y23i
    f33=-2*h13*y13i - 2*h23*y23i
    f12r=-h11r*y12i + h22r*y12i - h11i*y12r - h22i*y12r + h23*y13i + h13*y23i
    f12i=-(h12*y11) - h11i*y12i - h22i*y12i + h11r*y12r - h22r*y12r - h23*y13r + h12*y22 + h13*y23r
    f13r=h23*y12i - h11r*y13i + h33*y13i - h11i*y13r - h12*y23i
    f13i=-(h13*y11) - h23*y12r - h11i*y13i + h11r*y13r - h33*y13r + h12*y23r + h13*y33
    f23r=-(h13*y12i) - h12*y13i - h22r*y23i + h33*y23i - h22i*y23r
    f23i=-(h13*y12r) + h12*y13r - h23*y22 - h22i*y23i + h22r*y23r - h33*y23r + h23*y33
    return np.asarray([f11,f22,f33,f12r,f12i,f13r,f13i,f23r,f23i])

