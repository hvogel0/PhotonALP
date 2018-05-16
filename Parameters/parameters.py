# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
#User specified parameters and functions are defined here
import numpy as np

"""
Parameters
enList 		Grid of energy values in TeV for which the evolution is computed
galaxy_model    Model for the redshift evolution of the galaxies. Currently we have implemented 'Yuksel' and 'Schober'
flux_model      Model for the neutrino spectrum. Currently we have implemented 'Kopper' and 'Nieder' (Niederhausen analysis)
"""

enList = 10**np.arange(0,4.1,0.1)
galaxy_model = 'Schober'
flux_model = 'Nieder'


