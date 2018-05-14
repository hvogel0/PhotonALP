# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

"""
Defines IceCube neutrino fluxes that are used to normalize against for flux.py. We currently implement the Kopper and the Niederhausen fluxes, see 1712.01839.
"""

#--- IMPORTS ------
import numpy as np
from scipy.integrate import quad
import sys
sys.path.insert(0, '../Parameters/')
sys.path.insert(0, 'Galaxy_Models/')
import constants as cst
import galaxy_specs as gp
import cosmology as co
#------------------

pi=np.pi

class ICFlux:
    """
    Flux model.

    Constants
    Ebnu        Neutrino spectral break energy in GeV
    norm        Per flavor normalization of neutrino flux at Enorm [GeV] in GeV-1 cm-2 s-1 sr-1
    Enorm       Energy in GeV at which norm is defined.
    spec        spectral index of neutrino flux

    Functions
    Phi         Neutrino flux as function of neutrino energy in GeV-1 cm-2 s-1 sr-1
    normInt     Integrand for the overall flux normalization. Function of redshift z, neutrino energy [GeV], and self
    normIntPh   Flux for photons. The function is identical to normInt but with self.Ebnu replaced by 2*self.Ebnu.\
                This holds for photons. Function of redshift z, photon energy in [GeV], and self.
    """

    
    def __init__(self, model):
        if model == 'Kopper':
            self.Ebnu = 40.*10**3 #break energy in GeV
            self.norm = 2.46*10**(-18) #normalization constant in GeV-1 cm-2 s-1 sr-1
            self.Enorm = 100*10**3 #energy in GeV at which norm is defined
            self.spec = 2.92 #spectral index
            def Phi(self,Enu):#flux dependence on neutrino energy Enu. In GeV-1 cm-2 s-1 sr-1
                return self.norm*3.*(Enu/self.Enorm)**(-self.spec)

            def normInt(z,E,self):#integrand of norm(Enu) as a function of redshift
                spectr = ((E*(1.+z)/self.Ebnu)**2+(E*(1.+z)/self.Ebnu)**(2*self.spec))**(-1./2.)
                #broken neutrino spectrum from Kistler (eq. 1 1511.01530),with break energy Ebnu
                dEpdE=(1.+z)#dEprime/dE
                evoC=gp.evo(z)
                cosmo=1./(co.H0*(1.+z)*np.sqrt(co.Omega_lambda+(1.+z)**3*co.Omega_M))#Cosmological evolution
                return spectr*dEpdE*evoC*cosmo


            self.normI, self.err =quad(normInt,0,gp.zmax,args=(self.Enorm,self,),limit=300, epsrel=10**(-4))#(E, Ebreak)
            self.normC=4.*pi*Phi(self,self.Enorm)/cst.cv/self.normI #normalization constant

        if model == 'Nieder':
            self.Ebnu = 12.*10**3 #break energy in GeV
            self.norm = 1.57*10**(-18) #normalization constant in GeV-1 cm-2 s-1 sr-1
            self.Enorm = 100*10**3 #energy in GeV at which norm is defined
            self.spec = 2.48 #spectral index
            def Phi(self,Enu):#flux dependence on neutrino energy Enu. In GeV-1 cm-2 s-1 sr-1
                return self.norm*3.*(Enu/self.Enorm)**(-self.spec)

            def normInt(z,E,self):#integrand of norm(Enu) as a function of redshift
                spectr = ((E*(1.+z)/self.Ebnu)**2+(E*(1.+z)/self.Ebnu)**(2*self.spec))**(-1./2.)
                #broken neutrino spectrum from Kistler (eq. 1 1511.01530),with break energy Ebnu
                dEpdE=(1.+z)#dEprime/dE
                evoC=gp.evo(z)
                cosmo=1./(co.H0*(1.+z)*np.sqrt(co.Omega_lambda+(1.+z)**3*co.Omega_M))#Cosmological evolution
                return spectr*dEpdE*evoC*cosmo


            self.normI, self.err =quad(normInt,0,gp.zmax,args=(self.Enorm,self,),limit=300, epsrel=10**(-4))#(E, Ebreak)
            self.normC=4.*pi*Phi(self,self.Enorm)/cst.cv/self.normI #normalization constant

    def normIntPh(self,z,E):#same as normInt but with a break at 2*self.Ebnu. This holds for photons
        spectr = ((E*(1.+z)/(2*self.Ebnu))**2+(E*(1.+z)/(2*self.Ebnu))**(2*self.spec))**(-1./2.)
        #broken neutrino spectrum from Kistler (eq. 1 1511.01530),with break energy Ebnu
        dEpdE=(1.+z) #dEprime/dE
        evoC=gp.evo(z)
        cosmo=1./(co.H0*(1.+z)*np.sqrt(co.Omega_lambda+(1.+z)**3*co.Omega_M))#Cosmological evolution
        return spectr*dEpdE*evoC*cosmo
