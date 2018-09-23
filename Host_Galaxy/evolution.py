#!/u/th/hvogel/.local/bin/python3.6
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

#--- IMPORTS -------
import numpy as np
import sys
sys.path.insert(0, '../Parameters/')
sys.path.insert(0, 'Galaxy_Models/')
import constants as cst
#-------------------

"""
We define the evolution equation and its components. We follow the formalism introduced in 1611.04526

Functions
dDispAndPl  Dispersion due to plasma and photh-photon dispersion as a function of zero-redshift energy and redshift

fU11        U_11 component of evolution matrix. This is the photon polarization that decouples from the axion.\
        The arguments are Br: magnetic field integration variable, en: zero redshift energy in [TeV],\
        redshift zz, B: magnetic field normalization in [muG], ALP mass in [neV], Delta: photon dispersion from \
        plasma frequency and photon-photon dispersion in [kpc-1], Gamma: local absorption rate in [kpc-1], B_var\
        normalized variance of magnetic field components (assumed to be isotropic and Gaussian).\
        L: coherence length of the magnetic field at redshift zero in [kpc]

fU22        U_22 component of evolution matrix. This is the photon polarization that decouples from the axion.\
        The arguments are Br: magnetic field integration variable, en: zero redshift energy in [TeV],\
        redshift zz, B: magnetic field normalization in [muG], ALP mass in [neV], Delta: photon dispersion from \
        plasma frequency and photon-photon dispersion in [kpc-1], Gamma: local absorption rate in [kpc-1], B_var\
        normalized variance of magnetic field components (assumed to be isotropic and Gaussian).\
        L: coherence length of the magnetic field at redshift zero in [kpc]

fU33        U_33 component of evolution matrix. This is the photon polarization that decouples from the axion.\
        The arguments are Br: magnetic field integration variable, en: zero redshift energy in [TeV],\
        redshift zz, B: magnetic field normalization in [muG], ALP mass in [neV], Delta: photon dispersion from \
        plasma frequency and photon-photon dispersion in [kpc-1], Gamma: local absorption rate in [kpc-1], B_var\
        normalized variance of magnetic field components (assumed to be isotropic and Gaussian), and\
        L: coherence length of the magnetic field at redshift zero in [kpc]

fU23        U_23 component of evolution matrix. This is the photon polarization that decouples from the axion.\
        The arguments are Br: magnetic field integration variable, en: zero redshift energy in [TeV],\
        redshift zz, B: magnetic field normalization in [muG], ALP mass in [neV], Delta: photon dispersion from \
        plasma frequency and photon-photon dispersion in [kpc-1], Gamma: local absorption rate in [kpc-1], B_var\
        normalized variance of magnetic field components (assumed to be isotropic and Gaussian).\
        L: coherence length of the magnetic field at redshift zero in [kpc]
"""
#normalizations
npl=cst.npl #plasma frequency normalization for ne in cm-3
na=cst.na   #ALP mass normalization for ma in neV
nB=cst.nB   #Magnetic birefringence for B in muG
nag=cst.nag #Mixing normalization for B in muG
nGG=cst.nGG #Photon-photon dispersion normalization

#components of mixing matrix
def dDispAndPl(en,zz,gm):#Dispersion due to plasma and photon-photon dispersion as a function of energy and redshift, and a galaxy_model object
    enZ=en*(1.+zz)
    return npl/enZ*gm.ne(zz)+enZ*nGG*gm.DispInt(np.log10(en),zz)[0][0]

#functions to integrate
def fU11(Br,en,zz,B,gag,mass,Delta,Gamma,B_var,L):#computes U_11
    enZ=en*(1.+zz)#redshifted energy
    Lz=L/(1.+zz) #redshifted domain length. We assume galaxies get contracted by 1/(1+z)
    Delta_perp=Delta+2.*nB*B**2*Br**2*enZ #full dispersion of photon polarization perpendicular to the magnetic field
    E1=Delta_perp-Gamma/2.*1.j #eigenvalue E1
    U11=np.exp(-E1*Lz*1.j)
    return 1./B_var*abs(U11)**2*Br*np.exp(-Br**2/(2*B_var))

def fU22(Br,en,zz,B,gag,mass,Delta,Gamma,B_var,L):#compute U_22
    enZ=en*(1.+zz) #redshifted energy
    Lz=L/(1.+zz) #redshifted domain length. We assume galaxies get contracted by 1/(1+z)
    Delta_para=Delta+3.5*nB*B**2*Br**2*enZ #dispersion of photon polarization parallel to the magnetic field
    Delta_ag=nag*B*Br*gag #photon-ALP mixing
    Delta_a=na/enZ*mass**2# ALP-mass dispersion
    if (np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)).imag < 0:#check if photon is photon-like. see discussion in 1712.01839 .
        Delta_osc=np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)
    else:#if not, switch sign
        Delta_osc=-np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)
    Theta=np.arctan(2*Delta_ag/(Delta_para-Delta_a-Gamma/2*1.j))/2.#mixing angle
    E2=(Delta_para-Gamma/2.*1.j+Delta_a+Delta_osc)/2.#eigenvalue 2, see 1611.04526
    E3=(Delta_para-Gamma/2.*1.j+Delta_a-Delta_osc)/2.#eigenvalue 3, see 1611.04526
    U22=np.cos(Theta)**2*np.exp(-E2*Lz*1.j)+np.sin(Theta)**2*np.exp(-E3*Lz*1.j)
    return 1./B_var*abs(U22)**2*Br*np.exp(-Br**2/(2.*B_var))

def fU33(Br,en,zz,B,gag,mass,Delta,Gamma,B_var,L):
    enZ=en*(1.+zz) #redshifted energy
    Lz=L/(1.+zz) #redshifted domain length. We assume galaxies get contracted by 1/(1+z)
    Delta_para=Delta+3.5*nB*B**2*Br**2*enZ #dispersion of photon polarization parallel to the magnetic field
    Delta_ag=nag*B*Br*gag #photon-ALP mixing
    Delta_a=na/enZ*mass**2 #ALP-mass dispersion
    if (np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)).imag < 0:#check if ALP is ALP-like
        Delta_osc=np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)
    else:#if not, switch sign
        Delta_osc=-np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    Theta=np.arctan(2*Delta_ag/(Delta_para-Delta_a-Gamma/2.*1.j))/2. #mixing angle
    E2=(Delta_para-Gamma/2.*1.j+Delta_a+Delta_osc)/2. #eigenvalue 2, see 1611.04526
    E3=(Delta_para-Gamma/2.*1.j+Delta_a-Delta_osc)/2. #eigenvalue 3, see 1611.04526
    U33=np.sin(Theta)**2*np.exp(-E2*Lz*1.j)+np.cos(Theta)**2*np.exp(-E3*Lz*1.j)
    return 1/B_var*abs(U33)**2*Br*np.exp(-Br**2/(2*B_var))

def fU23(Br,en,zz,B,gag,mass,Delta,Gamma,B_var,L):
    enZ=en*(1.+zz) #redshifted energy
    Lz=L/(1.+zz) #redshifted domain length. We assume galaxies get contracted by 1/(1+z)
    Delta_para=Delta+3.5*nB*B**2*Br**2*enZ #dispersion of photon polarization parallel to the magntic field
    Delta_ag=nag*B*Br*gag #photon-ALP mixing
    Delta_a=na/enZ*mass**2 #ALP-mass dispersion
    if (np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)).imag < 0:#check if photon is photon-like
        Delta_osc=np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)
    else:# if not, switch sign
        Delta_osc=-np.sqrt((Delta_para-Delta_a-Gamma/2.*1.j)**2+4.*Delta_ag**2)
    Theta=np.arctan(2*Delta_ag/(Delta_para-Delta_a-Gamma/2.*1.j))/2. #mixing angle
    E2=(Delta_para-Gamma/2*1.j+Delta_a+Delta_osc)/2. #eigenvalue 2, see 1611.04526
    E3=(Delta_para-Gamma/2*1.j+Delta_a-Delta_osc)/2. #eigenvalue 3, see 1611.04526
    U23=np.sin(Theta)*np.cos(Theta)*(np.exp(-E2*Lz*1.j)-np.exp(-E3*Lz*1.j))
    return 1/B_var*abs(U23)**2*Br*np.exp(-Br**2/(2.*B_var))
