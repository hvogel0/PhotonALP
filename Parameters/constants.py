#!/u/th/hvogel/.local/bin/python3.6

import numpy as np

"""
Constants

cv	Speed of light in cm/s


Conversions

SecInvTokpcInv	conversion of s-1 to kpc-1
pcTocm          conversion of parsec into cm
cmTokpc         conversion of cm into kiloparsec

Physics parameters

npl		Normalization of plasma frequency for ne in cm-3
na		Normalization of ALP mass parameter for ma in neV
nB		Normalization of magnetic birefringence for B in muG
nag		Normalization of ALP-photon mixing for B in muG
nGG		Normalization of photon-photon dispersion

"""

#Constants
cv=2.998*10**(10) #speed of light in cm/s

#Unit conversion
SecInvTokpcInv = (3.0857*10**21)/cv #s-1 to kpc-1
pcTocm = 3.0857*10**(18)
cmTokpc = 1/pcTocm*10**(-3)

#Physics parameters
npl = -1.07*10**(-7)	#Normalization of plasma frequency for ne in cm-3
na = -7.8*10**(-5)	#Normalization of ALP mass for ma in neV
nB = 4.1*10**(-6)	#Normalization of magnetic birefringence for B in muG
nag = 1.52*10**(-2)	#Normalization of ALP-photon mixing for B in muG
nGG = 8.02*10**(-5)	#Normalization of photon-photon dispersion


