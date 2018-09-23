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
import propagation as prop
#---------------------

#Check if the right amount of arguments are supplied. Exit if not. 
#Arguments are #1 latitude in degrees and galactic coordinates, #2 longitude in degrees and galactic coodinates, #3 photon or ALP mode
if len(sys.argv)<4:
    print("Not enough arguments \n")
    sys.exit()
if len(sys.argv)>4:
    print("Too many arguments \n")
    sys.exit()

bstring = sys.argv[1]   #get latitude
lstring = sys.argv[2]   #get longitude
bb=float(bstring)       #convert string to float
ll=float(lstring)       #convert string to float

mode = sys.argv[3]      #get mode (photon or ALP)
if mode == 'None':
    y0=[0,0,0,0,0,0,0,0,0] # empty initial state
elif mode == 'ALP':
    y0=[0,0,1,0,0,0,0,0,0] # pure ALP initial condition
else:
    print('mode not known. Choose either None or ALP')
    print('Aborting...')
    sys.exit()


#create data files that we store the data in
dgstring ='Data/dg_'+bstring+'_'+lstring+'.txt' #name of photon data
dastring ='Data/da_'+bstring+'_'+lstring+'.txt' #name of ALP data
#Create files to write into
fg = open('Data/dg_'+bstring+'_'+lstring+'.txt','wb')   #open photon file
fa = open('Data/da_'+bstring+'_'+lstring+'.txt','wb')   #open ALP file
fg.close()
fa.close()


#load parametes
enList = para.enList
maList =para.maList
gagList = para.gagList


#load Milky Way model and check if implemented
mw_radiation_name = para.mw_radiation_model
if mw_radiation_name not in mw.mw_radiation_options:
    print('Radiation model not known. Please choose one of:\n', mw_radiation_options)
    print('Aborting...')
    sys.exit()

#load Milky Way model and check if implemented
mw_magnetic_field_name = para.mw_magnetic_field_model
if mw_magnetic_field_name not in mw.mw_mf_options:
    print('Magnetic field model not known. Please choose one of:\n', mw.mw_mf_options)
    print('Aborting...')
    sys.exit()

mwRadiation = mw.mwRadiationModel(mw_radiation_name,bstring,lstring)
mwMagField = mw.mwMagneticField(mw_magnetic_field_name)

d0=mwRadiation.dist_dat[len(mwRadiation.dist_dat)-1] # initial distance from earth
d1 = 0 # position of Earth
dd = 0.01 # length of step
d0 = prop.find_initial(d0,dd,bb,ll,mwMagField)
 
#omegaList=np.logspace(0,4,num=41,endpoint=True)


#dg=[] # initialize photon array
#da=[] # initialize ALP array
counter =0
# put mode option if initial condition 0
for gagC in gagList:
    for maC in maList:
        dg=[]
        da=[]
        for omega in para.enList:
            ytry=prop.H(0,omega,gagC,maC,bb,ll,mwRadiation,mwMagField).real # to check if conversion around earth is small
            if 900*1.52*10**(-2)*gagC<(ytry[0,0]+ytry[1,1]-ytry[2,2]): # if conversion rate is small, do simplified integration
                rg, err2=quad(prop.integratedH,d1,d0,args=(omega,gagC,maC,bb,ll,mwRadiation,mwMagField,),limit=100)
                dg.append([omega, maC, gagC, rg])
                da.append([omega, maC, gagC, 1])
            else:
                r = ode(prop.fsep, jac=None).set_integrator('lsoda', nsteps=10**5, rtol=10**(-4))
                r.set_initial_value(y0, d0).set_f_params(omega,gagC,maC,bb,ll,mwRadiation,mwMagField)
                r.integrate(d1)
                dg.append([omega, maC, gagC, ((r.y)[0]+(r.y)[1])])
                da.append([omega, maC, gagC, (r.y)[2]])
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

