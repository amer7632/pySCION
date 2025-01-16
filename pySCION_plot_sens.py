############################################################################################
########## pySCION - Pythonic Spatial Continuous Integration ###############################
########## Earth Evolution Model ###########################################################
############################################################################################
#### Coded by BJW Mills, AS Merdith
#### b.mills@leeds.ac.uk, andrew.merdith@adelaide.edu.au
####
#### plot sensitivity analysis

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp

def pySCION_plot_sens(sens):
    ###### define colours
    c_mean = np.asarray([255, 132, 34])/255
    c_std = np.asarray([255, 225, 192])/255
    c_range = np.asarray([1, 1, 1])/255

    #### Proxy color chart
    pc1 = np.asarray([65, 195, 199]) / 255
    pc2 = np.asarray([73, 167, 187]) / 255
    pc3 = np.asarray([82, 144, 170]) / 255
    pc4 = np.asarray([88, 119, 149]) / 255
    pc5 = np.asarray([89, 96, 125]) / 255
    pc6 = np.asarray([82, 56, 100]) / 255

    #### output to screen
    print('running sens plotting script... \t')

    #### load geochem data
    file_to_open_GEOCHEM = './data/geochem_data_2020.mat'
    geochem_data = sp.loadmat(file_to_open_GEOCHEM)
    file_to_open_SCOTESE = './data/Scotese_GAT_2021.mat'
    scotese_data = sp.loadmat(file_to_open_SCOTESE)

    #some standardised plotting params
    linewidth_mean = 2
    linewidth_outter = 2

    ####### make figure
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(24,12))
    for ax in axs.reshape(-1):
        ax.set_xlim([sens.time_myr[0][0], sens.time_myr[0][-1]])
        ax.set_xlabel('Time (Ma)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
    #
    #### Forcings (subplot 1)
    #
    #degass
    axs[0,0].plot(sens.time_myr[0],np.mean(sens.degass, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,0].plot(sens.time_myr[0],np.max(sens.degass, axis=0),  linewidth=linewidth_outter, c=c_range)
    axs[0,0].plot(sens.time_myr[0],np.min(sens.degass, axis=0),  linewidth=linewidth_outter, c=c_range)
    #### plot gran_area
    axs[0,0].plot(sens.time_myr[0],np.mean(sens.gran_area, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,0].plot(sens.time_myr[0],np.max(sens.gran_area, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,0].plot(sens.time_myr[0],np.min(sens.gran_area, axis=0), linewidth=linewidth_outter, c=c_range)
    #### plot bas_area
    axs[0,0].plot(sens.time_myr[0],np.mean(sens.bas_area, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,0].plot(sens.time_myr[0],np.max(sens.bas_area, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,0].plot(sens.time_myr[0],np.min(sens.bas_area, axis=0), linewidth=linewidth_outter, c=c_range)
    # Title and ylabel
    axs[0,0].set_title('Forcings', fontsize=14)
    axs[0,0].set_ylabel(r'$\delta^{13}C_{carb}$', fontsize=14)

    #### d13C record
    #### plot data comparison
    axs[0,1].plot(geochem_data['d13c_x'][0], geochem_data['d13c_y'][0], c=pc2, ls='dashed')
    #### plot this model
    #carb13
    axs[0,1].plot(sens.time_myr[0],np.mean(sens.delta_mccb, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,1].plot(sens.time_myr[0],np.max(sens.delta_mccb, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,1].plot(sens.time_myr[0],np.min(sens.delta_mccb, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,1].set_ylabel(r'$\delta^{13}C_{carb}$', fontsize=14)
    axs[0,1].set_title(r'$\delta^{13}C_{carb}$', fontsize=14)

    #### d34S record
    # plot data comparison
    axs[0,2].plot(geochem_data['d34s_x'][0],geochem_data['d34s_y'][0], c=pc2, ls='dashed')
    # plot this model
    axs[0,2].plot(sens.time_myr[0],np.mean(sens.d34s_S, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,2].plot(sens.time_myr[0],np.max(sens.d34s_S, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,2].plot(sens.time_myr[0],np.min(sens.d34s_S, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,2].set_ylabel(r'$\delta^{34}S_{sw}$', fontsize=14)
    axs[0,2].set_title(r'$\delta^{34}S_{sw}$', fontsize=14)

    #### Ocean 87Sr/86Sr
    # plot data comparison
    axs[0,3].plot(geochem_data['sr_x'][0],geochem_data['sr_y'][0], c=pc2)
    # plot this model
    axs[0,3].plot(sens.time_myr[0],np.mean(sens.delta_OSr, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,3].plot(sens.time_myr[0],np.max(sens.delta_OSr, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,3].plot(sens.time_myr[0],np.min(sens.delta_OSr, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,3].set_ylabel(r'$^{87}Sr/^{86}Sr seawater$', fontsize=14)
    axs[0,3].set_title(r'$^{87}Sr/^{86}Sr seawater$', fontsize=14)
    axs[0,3].set_ylim([0.706, 0.71])

    #### SO4
    # plot algeo data window comparison
    axs[0,4].plot(geochem_data['sconc_max_x'][0],geochem_data['sconc_max_y'][0], c=pc1)
    axs[0,4].plot(geochem_data['sconc_min_x'][0],geochem_data['sconc_min_y'][0], c=pc1)
    axs[0,4].plot(geochem_data['sconc_mid_x'][0],geochem_data['sconc_mid_y'][0], c=pc2)
    # plot fluid inclusion data comparison
    for i in np.arange(0, len(geochem_data['SO4_x']), 2):
        axs[0,4].plot([geochem_data['SO4_x'][i], geochem_data['SO4_x'][i]],
                      [geochem_data['SO4_y'][i], geochem_data['SO4_y'][i+1]], c=pc3)
    # plot this model
    axs[0,4].plot(sens.time_myr[0],np.mean(sens.SmM, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[0,4].plot(sens.time_myr[0],np.max(sens.SmM, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,4].plot(sens.time_myr[0],np.min(sens.SmM, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[0,4].set_ylabel(r'$Marine SO_{4} (mM)$')
    axs[0,4].set_title(r'$Marine\ SO_{4}$')

    #### O2 (%)
    # plot data comparison
    for i in np.arange(0, len(geochem_data['O2_x'])-1, 2):
        axs[1,0].plot([geochem_data['O2_x'][i], geochem_data['O2_x'][i]],
                      [geochem_data['O2_y'][i], geochem_data['O2_y'][i+1]], c=pc2)
    # plot this model
    axs[1,0].plot(sens.time_myr[0],np.mean(sens.mro2, axis=0) * 100, linewidth=linewidth_mean, c=c_mean)
    axs[1,0].plot(sens.time_myr[0],np.max(sens.mro2, axis=0) * 100, linewidth=linewidth_outter, c=c_range)
    axs[1,0].plot(sens.time_myr[0],np.min(sens.mro2, axis=0) * 100, linewidth=linewidth_outter, c=c_range)
    axs[1,0].set_title(r'$Atmospheric\ O_{2}\ (\%)$')
    axs[1,0].set_ylabel(r'$Atmospheric\ O_{2}\ (\%)$')

    #### CO2ppm
    ### plot data comparison
    # paleosol
    #axs[3,1].errorbar(geochem_data['paleosol_age'],geochem_data['paleosol_co2'],
    #                  geochem_data['paleosol_low'],geochem_data['paleosol_high'],
    #                   c=[0.4, 0.7, 0.7], ls=None)
    axs[1,1].scatter(geochem_data['paleosol_age'],
                     geochem_data['paleosol_co2'], color=pc1, edgecolors=pc1)
    #alkenone
    #axs[3,1].errorbar(geochem_data['alkenone_age'],geochem_data['alkenone_co2'],
    #                  geochem_data['alkenone_low'],geochem_data['alkenone_high'],
    #                   c=[0.4, 0.7, 0.4], ls=None)
    axs[1,1].scatter(geochem_data['alkenone_age'],
                     geochem_data['alkenone_co2'], color=pc2, edgecolors=pc2)
    #boron
    #axs[3,1].errorbar(geochem_data['boron_age'],geochem_data['boron_co2'],
    #                  geochem_data['boron_low'],geochem_data['boron_high'],
    #                   c=[0.4, 0.4, 0.4], ls=None)
    axs[1,1].scatter(geochem_data['boron_age'],
                     geochem_data['boron_co2'], color=pc3, edgecolors=pc3)
    #stomata
    #axs[3,1].errorbar(geochem_data['stomata_age'],geochem_data['stomata_co2'],
    #                  geochem_data['stomata_low'],geochem_data['stomata_high'],
    #                   c=[0.7, 0.7, 0.4], ls=None)
    axs[1,1].scatter(geochem_data['stomata_age'],
                     geochem_data['stomata_co2'], color=pc4, edgecolors=pc4)

    #liverwort
    #axs[3,1].errorbar(geochem_data['liverwort_age'],geochem_data['liverwort_co2'],
    #                  geochem_data['liverwort_low'],geochem_data['liverwort_high'],
    #                   c=[0.7, 0.7, 0.4], ls=None)
    axs[1,1].scatter(geochem_data['liverwort_age'],
                     geochem_data['liverwort_co2'], color=pc5, edgecolors=pc5)
    #phytane
    #axs[3,1].errorbar(geochem_data['phytane_age'],geochem_data['phytane_co2'],
    #                  geochem_data['phytane_low'],geochem_data['phytane_high'],
    #                   c=[0.7, 0.7, 0.4], ls=None)
    axs[1,1].scatter(geochem_data['phytane_age'],
                     geochem_data['phytane_co2'], color=pc6, edgecolors=pc6)
    # plot this model
    axs[1,1].plot(sens.time_myr[0],np.mean(sens.co2ppm, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[1,1].plot(sens.time_myr[0],np.max(sens.co2ppm, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,1].plot(sens.time_myr[0],np.min(sens.co2ppm, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,1].set_ylabel(r'$Atmospheric\ CO_{2}\ (ppm)$')
    axs[1,1].set_title(r'$Atmospheric\ CO_{2}$')
    axs[1,1].set_yscale('log')
    axs[1,1].set_ylim([100, 10000])

    ### TEMP
    # plot data comparison
    axs[1,2].plot(scotese_data['Scotese_2021_age'],scotese_data['Scotese_2021_GAT'], c=pc1)
    # plot this model
    axs[1,2].plot(sens.time_myr[0],np.mean(sens.T_gast, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[1,2].plot(sens.time_myr[0],np.max(sens.T_gast, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,2].plot(sens.time_myr[0],np.min(sens.T_gast, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,2].set_ylim([5, 40])
    axs[1,2].set_ylabel('GAST (C)')
    axs[1,2].set_title('GAST')

    ### ICE LINE
    # plot iceline proxy
    axs[1,3].plot(geochem_data['paleolat_x'],geochem_data['paleolat_y'], c=pc1)
    # plot this model
    axs[1,3].plot(sens.time_myr[0],np.mean(sens.iceline, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[1,3].plot(sens.time_myr[0],np.max(sens.iceline, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,3].plot(sens.time_myr[0],np.min(sens.iceline, axis=0), linewidth=linewidth_outter, c=c_range)
    #set title
    axs[1,3].set_title('Ice line')
    axs[1,3].set_ylabel('Ice line')

    #### P and N
    # plot this model
    axs[1,4].plot(sens.time_myr[0],np.mean(sens.P, axis=0), linewidth=linewidth_mean, c=c_mean)
    axs[1,4].plot(sens.time_myr[0],np.max(sens.P, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,4].plot(sens.time_myr[0],np.min(sens.P, axis=0), linewidth=linewidth_outter, c=c_range)
    axs[1,4].plot(sens.time_myr[0],np.mean(sens.N, axis=0), linewidth=linewidth_mean, ls='--',c=c_mean)
    axs[1,4].plot(sens.time_myr[0],np.max(sens.N, axis=0), linewidth=linewidth_outter, ls='--', c=c_range)
    axs[1,4].plot(sens.time_myr[0],np.min(sens.N, axis=0), linewidth=linewidth_outter, ls='--', c=c_range)
    axs[1,4].set_ylabel('P (-), N (--)')
    axs[1,4].set_title('P and N')

    fig.tight_layout()
    plt.show()
