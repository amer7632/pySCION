############################################################################################
########## pySCION - Spatial Continuous Integration ##########################################
########## Earth Evolution Model ###########################################################
############################################################################################
#### Coded by BJW Mills, AS Merdith
#### b.mills@leeds.ac.uk, andrew.merdith@adelaide.edu.au
####
#### model sensitivity analysis initialiser

import numpy as np
import pySCION_classes
import pySCION_initialise
import multiprocessing
from scipy.interpolate import interp1d
import pySCION_plot_sens
import time
import warnings

warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def pySCION_sens(sensruns, no_of_processes):

    sensruns = sensruns
    singlerun = 1
    #multi-core?
    output = []

    if __name__ == "__main__":
        p = multiprocessing.Pool(processes=no_of_processes)
    #should run SCION_initialise with an S=1
        for ind,iteration in enumerate(p.imap_unordered(pySCION_initialise.pySCION_initialise,
                                          [singlerun]*sensruns)):
            print('iteration:', ind)
            output.append(iteration)

        #make class to store results
        sens = pySCION_classes.Sens_class()
        #plots onto a grid of same spacing as first run
        tgrid = output[0].state.time

        #loop through output and put into class
        new_data_dict = {}
        for ind, i in enumerate(output):
            new_data_dict['interp_bas_area'] = interp1d(i.state.time,i.state.bas_area.ravel())(tgrid)
            new_data_dict['interp_gran_area'] = interp1d(i.state.time,i.state.gran_area.ravel())(tgrid)
            new_data_dict['interp_degass'] = interp1d(i.state.time,i.state.degass.ravel())(tgrid)
            new_data_dict['interp_delta_mccb'] = interp1d(i.state.time,i.state.delta_mccb.ravel())(tgrid)
            new_data_dict['interp_d34s_S'] = interp1d(i.state.time,i.state.d34s_S.ravel())(tgrid)
            new_data_dict['interp_delta_OSr'] = interp1d(i.state.time,i.state.delta_OSr.ravel())(tgrid)
            new_data_dict['interp_SmM'] = interp1d(i.state.time,i.state.SmM.ravel())(tgrid)
            new_data_dict['interp_co2ppm'] = interp1d(i.state.time,i.state.co2ppm.ravel())(tgrid)
            new_data_dict['interp_mro2'] = interp1d(i.state.time,i.state.mro2.ravel())(tgrid)
            new_data_dict['interp_iceline'] = interp1d(i.state.time,i.state.iceline, axis=0)(tgrid)
            new_data_dict['interp_T_gast'] = interp1d(i.state.time,i.state.T_gast.ravel())(tgrid)
            new_data_dict['interp_sat_tropical'] = interp1d(i.state.time,i.state.sat_tropical.ravel())(tgrid)
            new_data_dict['interp_sat_equator'] = interp1d(i.state.time,i.state.sat_equator.ravel())(tgrid)
            new_data_dict['interp_anox'] = interp1d(i.state.time,i.state.anox.ravel())(tgrid)
            new_data_dict['interp_P'] = interp1d(i.state.time,i.state.P.ravel())(tgrid)
            new_data_dict['interp_N'] = interp1d(i.state.time,i.state.N.ravel())(tgrid)
            new_data_dict['interp_time_myr'] = interp1d(i.state.time,i.state.time_myr.ravel())(tgrid)
            new_data_dict['interp_time'] = interp1d(i.state.time,i.state.time.ravel())(tgrid)

            #new_data = [new_data_dict]

            sens.add_states(new_data_dict)

        ###### plotting
        pySCION_plot_sens.pySCION_plot_sens(sens)

        return sens, output

##call pySCION sens
sens = pySCION_sens(24, 24)
#
##save
import pickle
filename = 'test_save'
t = time.localtime()
timestamp = time.strftime('%Y%b%d', t)
with open('./results/model_results/%s_%s.obj' % (filename, timestamp), 'wb') as file_:
    pickle.dump(sens, file_)
