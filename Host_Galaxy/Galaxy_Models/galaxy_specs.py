#This file defines the parameters of the galaxies that emit the IceCube neutrinos and the corresponding photons

"""
L	magnetic field's coherence length at z=0 in kpc
dis 	size of galaxy at z=0 in kpc
ne0	electron number density at z=0 in cm-3
T0	Initial photon fraction. The ALP fraction is Ta = 1 - T0
z0	First redshift bin
zmax 	Maximal redishift to which galaxies are considered
zstep 	step size in arange(z0,zmax,zstep)
"""
#Constants
L = 1 		#magnetic field's coherence length at z=0 in kpc
dis = 7 	#size of galaxy at z=0 in kpc
ne0 = 3 	#electron number density at z=0 in cm-3
T0 = 1		#initial photon fraction. The ALP fraction is  Ta = 1-T0
z0 = 0.		#First redshift bin
zmax = 6	#Maximal redishift to which galaxies are considered
zstep = 0.1	#step size in arange(z0,zmax,zstep)

#Functions
#Absorption:
model="Gamma_Schober_normal_YukselGRB.dat" #change this to use a different absorption model, model: [Energy [eV], redshift, Gamma [s-1]]
GammaDataRaw=np.loadtxt(model) #load data
enLDataG=np.asarray(np.log10(sorted(list(set(GammaDataRaw[:,0])))))-12 #extract energy data. Convert to logarithmic form and to TeV
zzDataG=np.asarray(sorted(list(set(GammaDataRaw[:,1])))) #redshift data
GDataG=GammaDataRaw[:,2].reshape(len(enLDataG),101) 
GammaInt=interpolate.RectBivariateSpline(enLDataG,zzDataG,GDataG,kx=1,ky=1)

#load dispersion data
DispDataRaw=np.loadtxt("norm_chi_Schober_normal_YukselGRB.dat")
enLDataD=np.asarray(np.log10(sorted(list(set(DispDataRaw[:,0])))))-12
#extract energy data. Convert to logarithmic form and to TeV
#second entry redhift
zzDataD=np.asarray(sorted(list(set(DispDataRaw[:,1]))))
DDataD=DispDataRaw[:,2].reshape(50,1000)
DispInt=interpolate.RectBivariateSpline(zzDataD,enLDataD,DDataD,kx=1,ky=1)


