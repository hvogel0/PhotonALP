#!/u/th/hvogel/.local/bin/python3.6
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

#--- Imports ----
import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from scipy.integrate import ode
import sys
sys.path.insert(0, '../Parameters/')
sys.path.insert(0, 'Galaxy_Models/')
import constants as cst
import parameters as para
import galaxy_specs as gp
import evolution as evol
#----------------

#Check if the right amount of arguments are supplied. Exit if not. 
#Arguments are #1 magnetic field strength in muG, photon-ALP coupling in 10^-11 GeV-1, ALP mass in log_10 (neV)
if len(sys.argv)<4:
    print("Not enough arguments \n")
    sys.exit()
if len(sys.argv)>4:
    print("Too many arguments \n")
    sys.exit()
  
#Read arguments
mag = sys.argv[1] #magnetic field in muG
gag = sys.argv[2] #coupling in 10^-11 GeV-1
mass = sys.argv[3] #ALP mass in log_10(neV)

#Convert string parameters to float values
mag=float(mag)
gag=float(gag)
mass=10**float(mass) #ALP mass in neV


#Define constants
cv=cst.cv #speed of light
pi=np.pi
L=gp.L  #coherence length of magnetic field at z=0 in kpc
dis=gp.dis  #extend of magnetic field at z=0in kpc
z0=gp.z0    #first redshift for source galaxies
zmax=gp.zmax    #Maximal redshift to which source galaxies are considered
zstep=gp.zstep  #step size from z0 to zmax
zzList = np.arange(z0,zmax,zstep)   #redhift grid considered for source galaxies
T0=gp.T0        #initial photon fraction
enList=para.enList  #Energy grid in TeV to compute propagation
B_var = gp.B_var    #Variance of magnetic field distribution.

#Conversion factors
SecInvTokpcInv=cst.SecInvTokpcInv #conversion of s-1 to kpc-1

#prepare ode, see 1611.04526
def fs(y,avU11,avU22,avU33,avU23):#rhs of ode, y: state vector (photon/ALP), avU11: averaged squared matrix elements
    Tg=y[0]#photon component
    Ta=y[1]#ALP components
    f0=(avU11/2.+avU22/2.-1.)*Tg+avU23*Ta
    f1=avU23/(2.)*Tg+(avU33-1.)*Ta
    return np.asarray([f0,f1])

#Integrate
#Initialize arrays:
dg=[] #photon array
da=[] #ALP array
y0=[T0,1-T0] #initial condition
d1=dis #destination
counter=0
for omega in enList: # loop over energy bins
    for zzC in zzList: #loop over redshift bins
        Gamma=SecInvTokpcInv*gp.GammaInt(np.log10(omega),zzC)[0][0] #get Gamma and convert to kpc-1
        Delta=evol.dDispAndPl(omega,zzC)
        magC=gp.magZ(mag,zzC)
        avU11, err11 = quad(evol.fU11,0,np.inf,args=(omega,zzC,magC,gag,mass,Delta,Gamma,B_var,L,),limit=100)#averaging of U11
        avU22, err22 = quad(evol.fU22,0,np.inf,args=(omega,zzC,magC,gag,mass,Delta,Gamma,B_var,L,),limit=100)#averaging of U22
        avU33, err33 = quad(evol.fU33,0,np.inf,args=(omega,zzC,magC,gag,mass,Delta,Gamma,B_var,L,),limit=100)#averaging of U33
        avU23, err23 = quad(evol.fU23,0,np.inf,args=(omega,zzC,magC,gag,mass,Delta,Gamma,B_var,L,),limit=100)#averaginf of U23
        y=y0
        d0=0. #intial step
        disz=dis/(1.+zzC) #total size of galaxy at redshift zzC
        while d0<disz-0.01:
            y=fs(y,avU11,avU22,avU33,avU23)+y #propatation for one domain
            d0=d0+L/(1.+zzC) #shift one domain further
        dg.append([omega,zzC,y[0]])                               
        da.append([omega,zzC,y[1]])
        if counter %10 ==0:
            print(omega, zzC)
        counter=counter+1

np.savetxt('Data/dg_B'+str(mag)+'_g'+str(gag)+'_m'+str(mass)+'.dat',dg, fmt='%.4e')
np.savetxt('Data/da_B'+str(mag)+'_g'+str(gag)+'_m'+str(mass)+'.dat',da, fmt='%.4e')

