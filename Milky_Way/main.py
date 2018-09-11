#! /usr/bin/env python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
"""
This code initiates the computation of the propagation of photons and ALPs through the Milky Way.
"""

#--- IMPORTS ---------
from scipy.integrate import ode
from scipy.integrate import quad
from scipy import interpolate
import numpy as np
import sys
sys.path.insert(0, '../Parameters/')
import constants as cst
import parameters as para
import mw_parameters as mw
#---------------------

#Check if the right amount of arguments are supplied. Exit if not. 
#Arguments are #1 latitude in degrees and galactic coordinates, #2 longitude in degrees and galactic coodinates
if len(sys.argv)<3:
    print("Not enough arguments \n")
    sys.exit()
if len(sys.argv)>3:
    print("Too many arguments \n")
    sys.exit()

bstring = sys.argv[1]   #get latitude
lstring = sys.argv[2]   #get longitude
bb=float(bstring)       #convert string to float
ll=float(lstring)       #convert string to float

#create data files that we store the data in
dgstring ='Data/dg_'+bstring+'_'+lstring+'.txt' #name of photon data
dastring ='Data/da_'+bstring+'_'+lstring+'.txt' #name of ALP data
fg = open('Data/dg_'+bstring+'_'+lstring+'.txt','wb')   #open photon file
fa = open('Data/da_'+bstring+'_'+lstring+'.txt','wb')   #open ALP file

#define dreibein of vectors. kUnit is the vector that points in the direction of the photon propagation. e1 and e2 are orthogonal to kUnit. e2 lies in the xy-plane.
kUnit=np.asarray([np.cos(np.pi/180.*bb)*np.cos(np.pi/180.*ll),np.cos(np.pi/180.*bb)*np.sin(np.pi/180.*ll),np.sin(np.pi/180.*bb)])
e1=np.asarray([-np.sin(np.pi/180.*bb)*np.cos(np.pi/180.*ll),-np.sin(np.pi/180.*bb)*np.sin(np.pi/180.*ll),np.cos(np.pi/180.*bb)]);
e2=np.asarray([-np.sin(np.pi/180.*ll),np.cos(np.pi/180.*ll),0.]);

#load parametes
enList = para.enList
maList =para.maList
gagList = para.gagList

#normalizations
npl=cst.npl
na=cst.na
nB=cst.nB
nag=cst.nag
nGG=cst.nGG

#load Milky Way model
mw_radiation_name = para.mw_radiation_model
mw_radiation_options = mw.mw_radiation_options
if mw_radiation_name not in mw_radiation_options:
    print('Model not known. Please choose one of:\n', mw_radiation_options)
    print('Aborting...')
    sys.exit()

mwRadiation = mw.mwRadiationModel(mw_name,bstring,lstring)

#----STOP------
def fBtor (rDisc,x,y,z):
    z0=5.3
    h=0.4
    w=0.27
    Phi=np.arctan2(y/rDisc,x/rDisc)
    Bn=1.4
    Bs=1.1
    discExp=np.exp(-abs(z)/z0)
    L1=1./(1+np.exp(-2*(abs(z)-h)/w))
    Ln=(1-1./(1+np.exp(-2*(rDisc-9.22)/0.2)))
    Ls=(1-1/(1+np.exp(-2*(rDisc-16.7)/0.2)))
    if Phi<0:
        Phi=Phi+2*np.pi
    if z>=0:
        return Bn*discExp*L1*Ln*np.asarray([-np.sin(Phi),np.cos(Phi),0])
    return Bs*discExp*L1*Ls*np.asarray([np.sin(Phi),-np.cos(Phi),0])

def fBx (rDisc,x,y,z):
    Bx=4.6
    rx=2.9
    rxc=4.8
    Theta0=49 #in degrees
    Phi=np.arctan2(y/rDisc,x/rDisc)
    rp=rxc*rDisc/(rxc+abs(z)/np.tan(Theta0*np.pi/180))
    if Phi<0:
        Phi=Phi+2*np.pi
    if rp>rxc:
        rp=rDisc-abs(z)/np.tan(Theta0*np.pi/180)
        bx=Bx*np.exp(-rp/rx)
        Btot=bx*rp/rDisc
        if z>=0:
            if z==0:
                return Btot*np.asarray([np.sin(0*np.pi/180)*np.cos(Phi),np.sin(0*np.pi/180)*np.sin(Phi),np.cos(0*np.pi/180)])
            else:
                return Btot*np.asarray([np.cos(Theta0*np.pi/180)*np.cos(Phi),np.cos(Theta0*np.pi/180)*np.sin(Phi),                                    np.sin(Theta0*np.pi/180)])
        return Btot*np.asarray([-np.cos(Theta0*np.pi/180)*np.cos(Phi),-np.cos(Theta0*np.pi/180)*np.sin(Phi),                                np.sin(Theta0*np.pi/180)])
    else:
        rp=rxc*rDisc/(rxc+abs(z)/np.tan(Theta0*np.pi/180))
        bx=Bx*np.exp(-rp/rx)
        Btot=bx*(rp/rDisc)**2
        if abs(z**2+(rDisc-rp)**2)<10**(-5):
            ThetaX=0
        else:
            ThetaX=np.arctan2(rDisc-rp,abs(z))
        if z>=0:
            return Btot*np.asarray([np.sin(ThetaX)*np.cos(Phi),np.sin(ThetaX)*np.sin(Phi),np.cos(ThetaX)])
        else:
            return Btot*np.asarray([-np.sin(ThetaX)*np.cos(Phi),-np.sin(ThetaX)*np.sin(Phi),np.cos(ThetaX)])
        
def fB(rE,b,l):
    z=rE*np.sin(b*np.pi/180)
    rX=rE*np.cos(b*np.pi/180)
    x=rX*np.cos((l*np.pi)/180)-8.5
    y=rX*np.sin((l*np.pi)/180)
    rDisc=np.sqrt(x**2+y**2)
    rCenter=np.sqrt(x**2+y**2+z**2)
    if rCenter<1 or rCenter>20:
        return np.asarray([0,0,0])
    else:
        h=0.4
        w=0.27
        BDisc=fBDisc(rDisc,x,y)*(1-1/(1+np.exp(-2*(abs(z)-h)/w)))
        Btor=fBtor(rDisc,x,y,z)
        Bx=fBx(rDisc,x,y,z)
        return BDisc+Btor+Bx


# In[8]:

def H(d,omega,b,l):
    BVec=fB(d,b,l)
    BT=BVec-np.dot(kUnit,BVec)*kUnit #Project out components that are parallel to the photon's direction of motion
    Bnorm=np.linalg.norm(BT)
    if Bnorm >10**(-12):
        cphi=np.dot(BT,e1)/Bnorm
        sphi=np.dot(BT,e2)/Bnorm
        dgg =(fdgg(d,omega))[0,0]
        gamma =(fgamma(d,omega))[0,0]
        h11=ne/omega*npl+nB*omega*Bnorm**2*(3.5*cphi**2+2*sphi**2)+dgg-1j*gamma/2
        h22=ne/omega*npl+nB*omega*Bnorm**2*(3.5*sphi**2+2*cphi**2)+dgg-1j*gamma/2
        h33=ma**2/omega*na
        h12=1.5*nB*omega*Bnorm**2*sphi*cphi
        h13=nag*cphi*Bnorm*gag
        h23=nag*sphi*Bnorm*gag
    else:
        dgg =(fdgg(d,omega))[0,0]
        gamma =(fgamma(d,omega))[0,0]
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

def integ(t,omega,b,l):
    h=H(t,omega,b,l)
    return (abs(h[0,2])**2+abs(h[1,2])**2)/(abs(h[0,0]+h[1,1]-h[2,2]))**2

# In[9]:

#function for ode
def f(t,y,omega,b,l):
    h = H(t,omega,b,l)
    M=np.asarray(y).reshape(3,3)
    eM=1j*(np.dot(h,M) - np.dot(M,np.conjugate(h)))
    return np.asarray(eM).reshape(9,)

def fsep(t,y,omega,b,l):
    y11=y[0]
    y22=y[1]
    y33=y[2]
    y12r=y[3]
    y12i=y[4]
    y13r=y[5]
    y13i=y[6]
    y23r=y[7]
    y23i=y[8]
    h = H(t,omega,b,l)
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

# In[12]:

#inital conditions and parameters
y0=[0,0,1,0,0,0,0,0,0]
t0=dist_dat[len(dist_dat)-1]
t1 = 0
dt = 0.01
def find_initial(t0,b,l):
    while t0>0:
        B=fB(t0,b,l)
        if B[0]>10**(-3):
            return t0
        else:
            if B[1]>10**(-3):
                return t0
            else:
                if B[2]>10**(-3):
                    return t0
                else:
                    t0=t0-dt

t0=find_initial(t0,bb,ll)
omegaList=np.logspace(0,4,num=41,endpoint=True)


# In[ ]:

dg=[]
da=[]
counter =0
fg.close()
fa.close()
for gagC in gagList:
    for maC in maList:
        dg=[]
        da=[]
        for omega in omegaList:
            gag=gagC
            ma=maC
            y0=[0,0,1,0,0,0,0,0,0]
            ytry=H(0,omega,bb,ll).real
            if 900*1.52*10**(-2)*gag<(ytry[0,0]+ytry[1,1]-ytry[2,2]):
                rg, err2=quad(integ,t1,t0,args=(omega,bb,ll,),limit=100)
                dg.append([omega, ma, gag, rg])
                da.append([omega, ma, gag, 1])
            else:
                r = ode(fsep, jac=None).set_integrator('lsoda', nsteps=10**5, rtol=10**(-4))
                r.set_initial_value(y0, t0).set_f_params(omega,bb,ll)
                r.integrate(t1)
                dg.append([omega, ma, gag, ((r.y)[0]+(r.y)[1])])
                da.append([omega, ma, gag, (r.y)[2]])
                """r = ode(f, jac=None).set_integrator('zvode', method='adams', nsteps=10**5)
                r.set_initial_value(y0, t0).set_f_params(omega,bb,ll)
                while r.successful() and r.t > t1:
                    r.integrate(r.t-dt)
                dg.append([omega, ma, gag, r.y.real[0]+r.y.real[4]])
                da.append([omega, ma, gag, r.y.real[8]])"""
            if counter % 100 ==0:
                 print(omega, maC, gagC, flush=True)
            else:
                if counter % 10 ==0:
                    print(omega, maC, gagC)
            counter= counter +1
        with open(dgstring, 'ba') as fg_name:
            np.savetxt(fg_name,dg,fmt='%.4e')
        with open(dastring, 'ba') as fa_name:
            np.savetxt(fa_name,da,fmt='%.4e')

