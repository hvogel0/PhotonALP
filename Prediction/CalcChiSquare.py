#!/u/th/hvogel/.local/bin/python3.6
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

# In[31]:

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy. integrate import nquad
import sys
from astropy.coordinates import SkyCoord

#conversion factors
kmsqTocmsq=10**10
kmsqTomsq=10**6

if len(sys.argv)<3:
	print("Not enough arguments \n")
	sys.exit()
mag = sys.argv[1]
mass = sys.argv[2]
gag = sys.argv[3]


#load gamma ray bkg
GammaSigRaw = np.loadtxt('../../MW_GR/GRMisiriotis/Combined/GFlux_ma'+mass+'_g'+gag+'.dat')
GammaBKGRaw = np.loadtxt('../../MW_GR/GRMisiriotis/Combined/GFlux_ma'+mass+'_g0.dat')
ALPSigRaw = np.loadtxt('../../GammaALPFluxesKopper/SYICFlux/Data/GFlux_B'+mag+'_ma'+mass+'_g'+gag+'.dat')

#number of energy bins used in previous calculations
enArr= np.logspace(0,4,num=41)
lenArr= len(enArr)

#load and interpolate effective areas
effAreaGammaRaw = np.asarray(np.loadtxt('../AuxFiles/effectiveArea.dat'))
effAreaProtonRaw = np.asarray(np.loadtxt('../AuxFiles/effectiveAreaProton.dat'))
eAg=interpolate.interp1d(effAreaGammaRaw[:,0],effAreaGammaRaw[:,1], kind='linear',bounds_error=False,fill_value=(0,1))
eAp=interpolate.interp1d(effAreaProtonRaw[:,0],effAreaProtonRaw[:,1],kind='linear',bounds_error=False,fill_value=(0.06,1))

#observation time in seconds divided by 4
timeArr=np.asarray([1,3,5,7,10])*31536000/4 

#determine energy bins with energy in TeV
binG=[]
en_bin=effAreaGammaRaw[:,0]
for ii in range(1,len(en_bin)-1):
    binG.append([en_bin[ii]-3-(en_bin[ii]-en_bin[ii-1])/2,en_bin[ii]-3+(en_bin[ii+1]-en_bin[ii])/2])

binP=[]
en_binP=effAreaProtonRaw[:,0]
for ii in range(1,len(en_binP)-1):
    binP.append([en_binP[ii]-3-(en_binP[ii]-en_binP[ii-1])/2,en_binP[ii]-3+(en_binP[ii+1]-en_binP[ii])/2])

#miss-ID and interpolation
missIDRaw=np.asarray([[-3,0],[np.log10(10),np.log10(0.01*10**(-2))],[np.log10(30),np.log10(0.004*10**(-2))],[np.log10(80),np.log10(0.001*10**(-2))]\
        ,[np.log10(10**6),np.log10(0.00001*10**(-2))]])
missID=interpolate.interp1d(missIDRaw[:,0],missIDRaw[:,1],kind='linear',bounds_error=False,fill_value=(0,np.log10(0.00001*10**(-2))))

#Cosmic ray fluxes
def H3AProtonFlux(EnergyProton):
    return 1/EnergyProton*(7860*(EnergyProton)**(-1.66)* np.exp(-EnergyProton/(4*10**6))+20*(EnergyProton)**(-1.4)                    *np.exp(-EnergyProton/(30*10**(6)))+1.7*(EnergyProton)**(-1.4)*np.exp(-EnergyProton/(2*10**(9))))
def H3AHeliumFlux(EnergyProton):
    return 1/EnergyProton*(3550*(4*EnergyProton)**(-1.58)*np.exp(-(4*EnergyProton)/(2*4*10**(6)))+20*(4*EnergyProton)**(-1.4)                    *np.exp(-(4*EnergyProton)/(2*30*10**(6)))+1.7*(4*EnergyProton)**(-1.4)*np.exp(-(4*EnergyProton)                                                                                                  /(2*2*10**(9))))
def H3ACNOFlux(EnergyProton):
    return 1/EnergyProton*(2200*(14*EnergyProton)**(-1.63)*np.exp(-(14*EnergyProton)/(7*4*10**(6)))+13.4*                           (14*EnergyProton)**(-1.4)*np.exp(-(14*EnergyProton)/(7*30*10**(6)))+1.14                           *(14*EnergyProton)**(-1.4)*np.exp(-(14*EnergyProton)/(7*2*10**(9))))
def H3AMgSiFlux(EnergyProton):
    return 1/EnergyProton*(1430*(27*EnergyProton)**(-1.67)*np.exp(-(27*EnergyProton)/(13*4*10**(6)))+13.4*                           (27*EnergyProton)**(-1.4)*np.exp(-(27*EnergyProton)/(13*30*10**(6)))                           +1.14*(27*EnergyProton)**(-1.4)*np.exp(-(27*EnergyProton)/(13*2*10**(9))))
def H3AFeFlux(EnergyProton):
    return 1/EnergyProton*(2120*(56*EnergyProton)**(-1.63)*np.exp(-(56*EnergyProton)/(26*4*10**(6)))+13.4*                           (56*EnergyProton)**(-1.4)*np.exp(-(56*EnergyProton)/(26*30*10**(6)))+1.14*                                               (56*EnergyProton)**(-1.4)*np.exp(-(56*EnergyProton)/(26*2*10**(9))))
def PFlux(EnergyProton):
    return 1*H3AProtonFlux(EnergyProton)+4*H3AHeliumFlux(EnergyProton)+14*H3ACNOFlux(EnergyProton)+27*H3AMgSiFlux(EnergyProton)+56*H3AFeFlux(EnergyProton)

#Compute number of proton events
def ArTimesFluxP(Ep):
    flux = PFlux(Ep*10**3)#energy given to PFlux is in GeV, while Ep is in TeV
    area = eAp(np.log10(Ep*10**3))#eAP is a function of log(Ep/GeV)
    events=10**3*kmsqTomsq*timeArr[-1]#per steradian, the factor 10**3 converts GeV-1 to TeV-1
    return flux*area*events

nP=[]
for ii in range(0,len(binG)):
    nInt, err = quad(ArTimesFluxP,10**(binG[ii][0]),10**(binG[ii][1]))
    nP.append([en_bin[ii+1]-3,nInt])#we cut out some bins above so binG[[ii]] corresponds to en_bin[[ii+1]], en_bin in TeV
nP=np.asarray(nP)

#improved sensitivity with water cherenkov WE SKIP THIS
"""ar1=np.asarray([[0,1],[3.8148036, 92.89562], [7.0729938, 22.88397], [12.887701, 8.994616], [21.799194, 4.7386436],\
        [35.42126,3.4814465], [62.691696, 3.3220615], [112.19813, 5.220865],[10**6,5.5]])
ar2=np.asarray([[0,1],[3.94554, 236.515], [7.35602, 77.4848], [13.1751, 33.2271], [22.2843,18.9477], [35.3994, 9.82469],\
        [57.1999, 5.82852], [105.372, 4.93896],[10**6,5.5]])
ar1I=interpolate.interp1d(ar1[:,0],ar1[:,1],kind='linear', bounds_error=False, fill_value=(1,5.5))
ar2I=interpolate.interp1d(ar2[:,0],ar2[:,1],kind='linear', bounds_error=False, fill_value=(1,5.5))"""


#Compute number of gamma BKG events

def ArTimesFluxG(Ep):
    flux =10**(GFlux(np.log10(Ep)))/Ep**2#GFlux is in GeV cm^-2 s-1 sr-1
    area = eAg(np.log10(Ep*10**3))#argument of the effective area is in log(Ep/GeV)
    events=10**(-3)*kmsqTocmsq*timeArr[-1]#per steradian, 10**(-3) to convert to TeV
    return flux*area*events

en_data=np.log10(GammaBKGRaw[0:lenArr,2])#in TeV
numData=[]
denData=[]
numTest=[]
denTest=[]
chiSq=[]
chiMax=0
blPoints=round(len(GammaBKGRaw)/lenArr)
for ii in range(0,blPoints):#rangeL
    nG=[]
    GFlux=interpolate.interp1d(en_data,np.log10(GammaBKGRaw[ii*lenArr:(ii+1)*lenArr,3]),\
            bounds_error=False,fill_value="extrapolate")
    for jj in range(0,len(binG)):
        nInt, err = quad(ArTimesFluxG,10**(binG[jj][0]),10**(binG[jj][1]),epsrel=10**(-3))
        nG.append([en_bin[jj+1],nInt])
    nG=np.asarray(nG)
    
    nGSIG=[]
    GFlux=interpolate.interp1d(en_data,np.log10(GammaSigRaw[ii*lenArr:(ii+1)*lenArr,3]),\
            bounds_error=False,fill_value="extrapolate")
    for jj in range(0,len(binG)):
        nInt, err = quad(ArTimesFluxG,10**(binG[jj][0]),10**(binG[jj][1]),epsrel=10**(-3))
        nGSIG.append([en_bin[jj+1],nInt])
    nGSIG=np.asarray(nGSIG)
    
    nALP=[]
    if str(mag)=='0':
        for jj in range(0,len(binG)):
            nALP.append([en_bin[jj+1],0])
    else:
        GFlux=interpolate.interp1d(np.log10(ALPSigRaw[0:lenArr,2]),np.log10(ALPSigRaw[ii*lenArr:(ii+1)*lenArr,3]),\
            bounds_error=False,fill_value="extrapolate")
        for jj in range(0,len(binG)):
            nInt, err = quad(ArTimesFluxG,10**(binG[jj][0]),10**(binG[jj][1]),epsrel=10**(-3))
            nALP.append([en_bin[jj+1],nInt])
    nALP=np.asarray(nALP)
    

    for jj in range(0,len(binG)):
        enL = en_bin[jj+1]
        chinum_loc=(nALP[jj,1]+nGSIG[jj,1]-nG[jj,1])
        chiden_loc=(nG[jj,1]+nP[jj,1]*10**(missID(enL-3)))#*(ar1I(10**(enL-3))/ar2I(10**(enL-3)))**2)
        numData.append([enL,GammaBKGRaw[ii*lenArr,0],GammaBKGRaw[ii*lenArr,1],chinum_loc])
        denData.append([enL,GammaBKGRaw[ii*lenArr,0],GammaBKGRaw[ii*lenArr,1],chiden_loc])
    
    if ii %10 ==0:
        print(ii)

#sort numData and denData
numData=np.asarray(sorted(numData, key=lambda tup: (tup[0],tup[1],tup[2])))
denData=np.asarray(sorted(denData, key=lambda tup: (tup[0],tup[1],tup[2])))

np.savetxt('DataInt/FullNum_B'+str(mag)+'_ma'+str(mass)+'_g'+str(gag)+'.txt',numData)
np.savetxt('DataInt/FullDen_B'+str(mag)+'_ma'+str(mass)+'_g'+str(gag)+'.txt',denData)#FIXME

#interpolate in each grid point
bListp=np.asarray([float(ele) for ele in ['0', '001', '002', '003', '004', '005', '006', '007', '008', '010',\
        '015', '020', '030', '040', '050', '060', '070', '080', '090']])
bListm=np.asarray([-1.*float(ele) for ele in ['001', '002', '003', '004', '005', '006', '007', '008', '010',\
        '015', '020', '030', '040', '050', '060', '070', '080', '090']])
bList=np.asarray(sorted(np.r_[bListp, bListm]))
lListp=np.asarray([float(ele) for ele in ['000', '002', '004', '006', '008', '010', '015', '020', '025', '030',\
        '040', '050', '060', '070', '080', '090', '110', '130', '150', '180']])
lListm=np.asarray([-1.*float(ele) for ele in ['002', '004', '006', '008', '010', '015', '020', '025', '030',\
        '040', '050', '060', '070', '080', '090', '110', '130', '150', '180']])
lList=np.asarray(sorted(np.r_[lListp, lListm]))


nInt=[]
dInt=[]
nnAr=[]
ddAr=[]
for ee in range(0,len(binG)):
    num=np.asarray(numData[ee*blPoints:(ee+1)*blPoints,3].reshape(len(bList),len(lList)))
    den=np.asarray(denData[ee*blPoints:(ee+1)*blPoints,3].reshape(len(bList),len(lList)))
    nInt.append(interpolate.RectBivariateSpline(np.sin(bList*np.pi/180),lList,num,kx=1,ky=1))
    dInt.append(interpolate.RectBivariateSpline(np.sin(bList*np.pi/180),lList,den,kx=1,ky=1))

#build grid
coarseness=2
ragrid=np.arange(0,360,coarseness)
decgrid=np.arange(-10.65,69.35,coarseness)
#evaluate chisq on grid
def convert_this(dec,RA):
    conv=np.pi/180
    cdec = conv*dec
    cRA=conv*RA
    cdg=27.13*conv
    cag=192.85*conv
    sb=np.sin(cdg)*np.sin(cdec)+np.cos(cdg)*np.cos(cdec)*np.cos(cRA-cag)#sin of latitude
    lvalue=122.9-180/np.pi*np.arctan2(np.cos(cdec)*np.sin(cRA-cag),(np.cos(cdg)*np.sin(cdec)-np.sin(cdg)*np.cos(cdec)*np.cos(cRA-cag)))#tan(122.9-l)
    return [180/np.pi*np.arcsin(sb),lvalue]

print(convert_this(-10.65,0))

def drnum(dec,RA,energy_index):
    conv=np.pi/180
    cdec = conv*dec
    cRA=conv*RA
    cdg=27.13*conv
    cag=192.85*conv
    sb=np.sin(cdg)*np.sin(cdec)+np.cos(cdg)*np.cos(cdec)*np.cos(cRA-cag)#sin of latitude
    lvalue=122.9-180/np.pi*np.arctan2(np.cos(cdec)*np.sin(cRA-cag),(np.cos(cdg)*np.sin(cdec)-np.sin(cdg)*np.cos(cdec)*np.cos(cRA-cag)))#tan(122.9-l)
    pre=conv**2*np.cos(cdec)
    return pre*nInt[energy_index](sb,lvalue)

def drden(dec,RA,energy_index):
    conv=np.pi/180
    cdec = conv*dec
    cRA=conv*RA
    cdg=27.13*conv
    cag=192.85*conv
    sb=np.sin(cdg)*np.sin(cdec)+np.cos(cdg)*np.cos(cdec)*np.cos(cRA-cag)#sin of latitude
    lvalue=122.9-180/np.pi*np.arctan2(np.cos(cdec)*np.sin(cRA-cag),(np.cos(cdg)*np.sin(cdec)-np.sin(cdg)*np.cos(cdec)*np.cos(cRA-cag)))#tan(122.9-l)
    pre=conv**2*np.cos(cdec)
    return pre*dInt[energy_index](sb,lvalue)

"""def area(dec,RA):
    conv=np.pi/180
    cdec = conv*dec
    cRA=conv*RA
    cdg=27.13*conv
    cag=192.85*conv
    sb=np.sin(cdg)*np.sin(cdec)+np.cos(cdg)*np.cos(cdec)*np.cos(cRA-cag)#sin of latitude
    lvalue=122.9-180/np.pi*np.arctan2(np.cos(cdec)*np.sin(cRA-cag),(np.cos(cdg)*np.sin(cdec)-np.sin(cdg)*np.cos(cdec)*np.cos(cRA-cag)))#tan(122.9-l)
    ar=-np.pi/180*(np.sin(cdec)-np.sin(cdec+conv*1))
    return ar"""

chiSq=0
grid=[]
for dec in decgrid:
    for ra in ragrid:
        chi_loc=0
        for ee in range(0,len(binG)):
            (resNum, err)=nquad(drnum,[[dec,dec+coarseness],[ra,ra+coarseness]],args=[ee,],\
            opts=[{'epsrel': 10**(-3),'limit': 100},{'epsrel': 10**(-3),'limit': 100}])
            (resDen, err)=nquad(drden,[[dec,dec+coarseness],[ra,ra+coarseness]],args=[ee,],\
            opts=[{'epsrel': 10**(-3),'limit': 100},{'epsrel': 10**(-3),'limit': 100}])
            chi_loc+=resNum**2/resDen
        grid.append([dec+coarseness/2,ra+coarseness/2,chi_loc])        
        chiSq+=chi_loc
        print(dec,ra,chiSq)

toSave=[[float(mag),float(mass),float(gag),chiSq]]
np.savetxt('DataRes/resF_B'+mag+'_ma'+mass+'_g'+gag+'.dat',toSave, fmt='%.4e')
np.savetxt('DataGrid/gridF_B'+mag+'_ma'+mass+'_g'+gag+'.dat',grid, fmt='%.4e')
