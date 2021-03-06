# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
#User specified parameters and functions are defined here
import numpy as np

"""
Parameters
magList         Grid of magnetic field values in [muG]
maList          Grid of mass values in [log10(neV)]
gagList         Grid of coupling values in [10^(-11) GeV-1]
enList 		Grid of energy values in TeV for which the evolution is computed
galaxy_model    Model for the redshift evolution of the galaxies. Currently we have implemented 'Yuksel' and 'Schober'
flux_model      Model for the neutrino spectrum. Currently we have implemented 'Kopper' and 'Nieder' (Niederhausen analysis)
mw_radiation_model        Model for the light field of the Milky Way. Currently we have implemented 'VernettoLipari'
mw_magnetic_field_model	  Model for the magnetic field of the Milky Way. Currently we have implemented 'JanssonFarrar'.
mw_CR_model     Model for the cosmic ray flux at earth. Currently, we only have a homogeneous and isotropic flux that mimics the flux at earth implemented.
mw_dust_model   Distribution of gas and dust in the Milky Way. Currently, we have implemented Models according to 'Misiriotis'.

"""
magList = [1,3,5,8,10,15]
maList=np.logspace(1,4,num=31)
gagList = [0,0.1,0.3,0.6,0.8,1,2,3,4,5,6] #ALP-photon couplings in [10^(-11) Gev-1]
enList = 10**np.arange(0,4.1,0.1)
galaxy_model = 'Schober' #['Yuksel', 'Schober']
flux_model = 'Nieder' #['Kopper','Nieder']
mw_radiation_model = 'VernettoLipari' #['VernettoLipari']
mw_magnetic_field_model = 'JanssonFarrar' #['JanssonFarrar']
mw_CR_model ='Default' #['Default']
mw_dust_model = 'Misiriotis' #['Misiriotis']


