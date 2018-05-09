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
ne0=gp.ne0  #electron density at z=0 in 1/cm**3
z0=gp.z0    #first redshift for source galaxies
zmax=gp.zmax    #Maximal redshift to which source galaxies are considered
zstep=gp.zstep  #step size from z0 to zmax
zzList = np.arange(z0,zmax,zstep)   #redhift grid considered for source galaxies
T0=gp.T0        #initial photon fraction
enList=para.enList  #Energy grid in TeV to compute propagation

#Conversion factors
SecInvTokpcInv=gp.SecInvTokpcInv #conversion of s-1 to kpc-1

#normalizations
npl=cst.npl #plasma frequency normalization for ne in cm-3
na=cst.na   #ALP mass normalization for ma in neV
nB=cst.nB   #Magnetic birefringence for B in muG
nag=cst.nag #Mixing normalization for B in muG
nGG=cst.nGG #Photon-photon dispersion normalization

#Function
#load gamma data
GammaDataRaw=np.loadtxt("Gamma_for_Schoberetal_normal_galaxies_redshift_with_YukselKistlerGRBEvolution.dat")
enLDataG=np.asarray(np.log10(sorted(list(set(GammaDataRaw[:,0])))))-12
#extract energy data. COnvert to logarithmic form and to TeV
#second entry redhift
zzDataG=np.asarray(sorted(list(set(GammaDataRaw[:,1]))))
GDataG=GammaDataRaw[:,2].reshape(101,101)
GammaInt=interpolate.RectBivariateSpline(enLDataG,zzDataG,GDataG,kx=1,ky=1)


# In[6]:

#load dispersion data
DispDataRaw=np.loadtxt("normalised_chiSchoberetalNormalGalaxiesRedshiftWithYukselKistlerGRBEvolution.dat")
enLDataD=np.asarray(np.log10(sorted(list(set(DispDataRaw[:,0])))))-12
#extract energy data. Convert to logarithmic form and to TeV
#second entry redhift
zzDataD=np.asarray(sorted(list(set(DispDataRaw[:,1]))))
DDataD=DispDataRaw[:,2].reshape(50,1000)
DispInt=interpolate.RectBivariateSpline(zzDataD,enLDataD,DDataD,kx=1,ky=1)


# In[7]:

def evo(zz):#Evolution model
    return ((1+zz)**(-34)+((1+zz)/5160.64)**3+((1+zz)/9.06)**35)**(-1/10)

def ne(zz):
    return ne0*(1+zz)**(3-2.14)*evo(zz)**(1/1.4)

#components of mixing matrix
def dDispAndPl(en,zz):
    enZ=en*(1+zz)
    return npl/enZ*ne(zz)+enZ*nGG*DispInt(zz,np.log10(en))[0][0]


# In[198]:

#functions to integrate
def fU11(Br,en,zz,B,gag,mass,Delta,Gamma):
    var = 2/3
    enZ=en*(1+zz)
    Lz=L/(1+zz)
    Delta_perp=Delta+2*nB*B**2*Br**2*enZ
    E1=Delta_perp-Gamma/2*1.j
    U11=np.exp(-E1*Lz*1.j)
    return 1/var*abs(U11)**2*Br*np.exp(-Br**2/(2*var))

def fU22(Br,en,zz,B,gag,mass,Delta,Gamma):
    var=2/3
    enZ=en*(1+zz)
    Lz=L/(1+zz)
    Delta_para=Delta+3.5*nB*B**2*Br**2*enZ
    Delta_ag=nag*B*Br*gag
    Delta_a=na/enZ*mass**2
    if (np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)).imag < 0:
        Delta_osc=np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    else:
        Delta_osc=-np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    Theta=np.arctan(2*Delta_ag/(Delta_para-Delta_a-Gamma/2*1.j))/2
    #sT=2*Delta_ag/(Delta_osc)
    E2=(Delta_para-Gamma/2*1.j+Delta_a+Delta_osc)/2
    E3=(Delta_para-Gamma/2*1.j+Delta_a-Delta_osc)/2
    U22=np.cos(Theta)**2*np.exp(-E2*Lz*1.j)+np.sin(Theta)**2*np.exp(-E3*Lz*1.j)
    #U22=sT**2*np.exp(-E3*L*1.j)+(1-sT**2)*np.exp(-E2*L*1.j)
    #print(Delta_para,Delta_ag,Delta_a,Delta_osc,sT,E2,E3,U22)
    return 1/var*abs(U22)**2*Br*np.exp(-Br**2/(2*var))

def fU33(Br,en,zz,B,gag,mass,Delta,Gamma):
    var= 2/3
    enZ=en*(1+zz)
    Lz=L/(1+zz)
    Delta_para=Delta+3.5*nB*B**2*Br**2*enZ
    Delta_ag=nag*B*Br*gag
    Delta_a=na/enZ*mass**2
    if (np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)).imag < 0:
        Delta_osc=np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    else:
        Delta_osc=-np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    Theta=np.arctan(2*Delta_ag/(Delta_para-Delta_a-Gamma/2*1.j))/2
    #sT=2*Delta_ag/(Delta_osc)
    E2=(Delta_para-Gamma/2*1.j+Delta_a+Delta_osc)/2
    E3=(Delta_para-Gamma/2*1.j+Delta_a-Delta_osc)/2
    U33=np.sin(Theta)**2*np.exp(-E2*Lz*1.j)+np.cos(Theta)**2*np.exp(-E3*Lz*1.j)
    #U33=sT**2*np.exp(-E2*L*1.j)+(1-sT**2)*np.exp(-E3*L*1.j)
    return 1/var*abs(U33)**2*Br*np.exp(-Br**2/(2*var))

def fU23(Br,en,zz,B,gag,mass,Delta,Gamma):
    var= 2/3
    enZ=en*(1+zz)
    Lz=L/(1+zz)
    Delta_para=Delta+3.5*nB*B**2*Br**2*enZ
    Delta_ag=nag*B*Br*gag
    Delta_a=na/enZ*mass**2
    if (np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)).imag < 0:
        Delta_osc=np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    else:
        Delta_osc=-np.sqrt((Delta_para-Delta_a-Gamma/2*1.j)**2+4*Delta_ag**2)
    Theta=np.arctan(2*Delta_ag/(Delta_para-Delta_a-Gamma/2*1.j))/2
    #sT=2*Delta_ag/(Delta_osc)
    E2=(Delta_para-Gamma/2*1.j+Delta_a+Delta_osc)/2
    E3=(Delta_para-Gamma/2*1.j+Delta_a-Delta_osc)/2
    #U23=sT*np.sqrt((1-sT**2))*(np.exp(-E2*L*1.j)-np.exp(-E3*L*1.j))
    U23=np.sin(Theta)*np.cos(Theta)*(np.exp(-E2*Lz*1.j)-np.exp(-E3*Lz*1.j))
    return 1/var*abs(U23)**2*Br*np.exp(-Br**2/(2*var))


# In[79]:

#rhs of ode
#def f(d,y,avU11,avU22,avU33,avU23):
#    Tg=y[0]
#    Ta=y[1]
#    f0=(avU11/2+avU22/2-1)/L*Tg+avU23/L*Ta
#    f1=avU23/(2*L)*Tg+(avU33-1)/L*Ta
#    return [f0,f1]

#rhs of ode
def fs(d,y,avU11,avU22,avU33,avU23):
    Tg=y[0]
    Ta=y[1]
    f0=(avU11/2+avU22/2-1)*Tg+avU23*Ta
    f1=avU23/(2)*Tg+(avU33-1)*Ta
    return np.asarray([f0,f1])


# In[202]:

#Integrate
dg=[]
da=[]
gagC=gag
massC=mass
y0=[T0,1-T0]
t0=0
t1=dis
counter=0
for omega in enList:
    for zzC in zzList:
        Gamma=sitokpci*GammaInt(np.log10(omega),zzC)[0][0]
        Delta=dDispAndPl(omega,zzC)
        magC=mag*(ne(zzC)/ne(0))**(1/6)*(evo(zzC)/(1+zzC))**(1/3)
        avU11, err11 = quad(fU11,0,np.inf,args=(omega,zzC,magC,gagC,massC,Delta,Gamma,),limit=100)
        avU22, err22 = quad(fU22,0,np.inf,args=(omega,zzC,magC,gagC,massC,Delta,Gamma,),limit=100)
        avU33, err33 = quad(fU33,0,np.inf,args=(omega,zzC,magC,gagC,massC,Delta,Gamma,),limit=100)
        avU23, err23 = quad(fU23,0,np.inf,args=(omega,zzC,magC,gagC,massC,Delta,Gamma,),limit=100)
        y=y0
        t0=0
        disz=dis/(1+zzC)
        while t0<disz-0.01:
            y=fs(t0,y,avU11,avU22,avU33,avU23)+y
            t0=t0+L/(1+zzC)
        #r = ode(f, jac=None).set_integrator('lsoda', nsteps=10**5)
        #r.set_initial_value(y0, t0).set_f_params(avU11,avU22,avU33,avU23)
        #r.integrate(t1)
        dg.append([omega,zzC,y[0]])                               
        da.append([omega,zzC,y[1]])
        #dg.append([omega,zzC,avU11])
        #da.append([omega,zzC,avU22])
        if counter %10 ==0:
            print(omega, zzC)
        counter=counter+1

np.savetxt('Data/dg_B'+magS+'_g'+gagS+'_m'+massS+'.dat',dg, fmt='%.4e')
np.savetxt('Data/da_B'+magS+'_g'+gagS+'_m'+massS+'.dat',da, fmt='%.4e')

