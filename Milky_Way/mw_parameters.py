# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8

"""
Defines parameters for the propagation in the Milky Way

Parameters
ne	    electron number density [cm-3]
mw_options  Options for the Milky Way model. Currently only VernettoLipari model is implemented

Functions

"""

ne = 1 #electron number density [cm-3]
mw_options = ['VernettoLipari']

class mwModel:


    def __init__(self,mwModel,bstring,lstring):
        if mwModel == 'VernettoLipari':
            self.modelMW = "Gamma/gamma_"+bstring+"_"+lstring+".dat"
    
    #load absorption
    self.gamma_data=np.loadtxt(self.modelMW) #load data
    self.dist_dat=np.asarray(sorted(list(set(self.gamma_data[:,1]))))#extract distance data
    self.eng_dat=np.asarray(np.log10(sorted(list(set(self.gamma_data[:,0])))))#extract energy data. Convert to logarithmic form
    gamma_dat=np.asarray((gamma_data[:,2]).reshape(len(dist_dat),len(eng_dat)))
            gamma_int=interpolate.RectBivariateSpline(dist_dat,eng_dat,gamma_dat,kx=1,ky=1)#Checked correct implementation on 08/09/2018

def fgamma(d,omega):
    if d<=0.5:#We cut off at 0.5 since the first bin is special. If we interpolated, the local absorption rate would go to zero at d=0
        return gamma_int(0.5,np.log10(omega)+12) #omega is in TeV but the argument is in eV. 
    return gamma_int(d,np.log10(omega)+12)

#load delta_gamma_gamma
dgg_data=np.loadtxt("Chi/chi_"+bstring+"_"+lstring+".dat")
dist_datgg=np.asarray([dgg_data[61*ii,1] for ii in range(0,int(len(dgg_data[:,1])/61))])
eng_datgg=np.log10(dgg_data[:61,0])
dgg_dat=np.asarray((dgg_data[:,2]).reshape(len(dist_datgg),len(eng_datgg)))
dgg_int=interpolate.RectBivariateSpline(dist_datgg,eng_datgg,dgg_dat,kx=1,ky=1)
