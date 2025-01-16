import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp

def pySCION_plot_fluxes(state, model_pars, pars):
    #### Proxy color chart
    pc1 = np.asarray([65, 195, 199]) / 255
    pc2 = np.asarray([73, 167, 187]) / 255
    pc3 = np.asarray([82, 144, 170]) / 255
    pc4 = np.asarray([88, 119, 149]) / 255
    pc5 = np.asarray([89, 96, 125]) / 255
    pc6 = np.asarray([82, 56, 100]) / 255

    #####################################################################
    #################   Plot global variables   #########################
    #####################################################################

    #### load geochem data
    file_to_open_GEOCHEM = './data/geochem_data_2020.mat'
    geochem_data = sp.loadmat(file_to_open_GEOCHEM)
    file_to_open_SCOTESE = './data/Scotese_GAT_2021.mat'
    scotese_data = sp.loadmat(file_to_open_SCOTESE)

    ####### make figure
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(36,24))
    for ax in axs.reshape(-1):
        #ax.set_xlim([model_pars.whenstart/1e6, model_pars.whenend/1e6])
        ax.set_xlim([-600, -0])
        ax.set_xlabel('Time (Ma)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid()
    #
    #### Forcings (subplot 1)
    #
    axs[0,0].plot(state.time_myr,state.degass,'r')
    axs[0,0].plot(state.time_myr,state.bas_area,'k')
    axs[0,0].plot(state.time_myr,state.evo,'g')
    axs[0,0].plot(state.time_myr,state.W,'b')
    axs[0,0].plot(state.time_myr,state.Bforcing,'m')
    axs[0,0].plot(state.time_myr,state.gran_area, c=[0.8, 0.8, 0.8])
    # Legend
    axs[0,0].text(-590,2.4,'D', c='r', fontsize=12)
    axs[0,0].text(-590,2.2,'E', c='g', fontsize=12)
    axs[0,0].text(-590,2,'W', c='b', fontsize=12)
    axs[0,0].text(-590,1.8,'B', c='m', fontsize=12)
    axs[0,0].text(-590,1.6,'BA',c='k', fontsize=12)
    axs[0,0].text(-590,1.4,'GA',c=[0.8, 0.8, 0.8], fontsize=12)
    # Title and ylabel
    axs[0,0].set_title('Forcings', fontsize=14)
    axs[0,0].set_ylabel('Relative forcing', fontsize=14)
    axs[0,0].set_ylim([0, 2.5])
    #
    #### Corg fluxes (subplot 2)
    #
    axs[0,1].plot(state.time_myr,state.mocb,'b')
    axs[0,1].plot(state.time_myr,state.locb,'g')
    axs[0,1].plot(state.time_myr,state.oxidw,'r')
    axs[0,1].plot(state.time_myr,state.ocdeg,'k')
    #### Legend
    axs[0,1].text(-590,5e12,'mocb',c='b', fontsize=12)
    axs[0,1].text(-590,4e12,'locb',c='g', fontsize=12)
    axs[0,1].text(-590,3e12,'oxidw',c='r', fontsize=12)
    axs[0,1].text(-590,2e12,'ocdeg',c='k', fontsize=12)
    ##### Title
    axs[0,1].set_title(r'$C_{org}\ fluxes$', fontsize=14)
    axs[0,1].set_ylabel('Flux (mol/yr)', fontsize=14)
    #
    #### Ccarb fluxes (subplot 3)
    #
    axs[0,2].plot(state.time_myr,state.silw, c='r')
    axs[0,2].plot(state.time_myr,state.carbw, c='c')
    axs[0,2].plot(state.time_myr,state.sfw, c='b')
    axs[0,2].plot(state.time_myr,state.mccb, c='k')
    # Legend
    axs[0,2].text(-590,28e12,'silw', c='r', fontsize=12)
    axs[0,2].text(-590,24e12,'carbw', c='c', fontsize=12)
    axs[0,2].text(-590,20e12,'sfw', c='b', fontsize=12)
    axs[0,2].text(-590,16e12,'mccb', c='k', fontsize=12)
    # Title
    axs[0,2].set_title(r'$C_{carb}\ fluxes$', fontsize=14)
    axs[0,2].set_ylabel('Flux (mol/yr)', fontsize=14)
    axs[0,2].set_ylim([0, 3e13])
    #
    #### S fluxes (subplot 4)
    #
    axs[0,3].plot(state.time_myr,state.mpsb, c='k')
    axs[0,3].plot(state.time_myr,state.mgsb, c='c')
    axs[0,3].plot(state.time_myr,state.pyrw, c='r')
    axs[0,3].plot(state.time_myr,state.pyrdeg, c='m')
    axs[0,3].plot(state.time_myr,state.gypw, c='b')
    axs[0,3].plot(state.time_myr,state.gypdeg, c='g')
    # Legend
    axs[0,3].text(-590,1.9e12,'mpsb', c='k', fontsize=12)
    axs[0,3].text(-590,1.7e12,'mgsb', c='c', fontsize=12)
    axs[0,3].text(-590,1.5e12,'pyrw', c='r', fontsize=12)
    axs[0,3].text(-590,1.2e12,'pyrdeg', c='m', fontsize=12)
    axs[0,3].text(-590,1e12,'gypw', c='b', fontsize=12)
    axs[0,3].text(-590,0.8e12,'gypdeg', c='g', fontsize=12)
    # Title
    axs[0,3].set_title('S fluxes', fontsize=14)
    axs[0,3].set_ylabel('Flux (mol/yr)', fontsize=14)
    axs[0,3].set_ylim([0, 2e12])
    #
    #### C SPECIES (subplot 5)
    #
    axs[1,0].plot(state.time_myr,state.G/pars.G0, c='k')
    axs[1,0].plot(state.time_myr,state.C/pars.C0, c='c')
    axs[1,0].plot(state.time_myr,state.veg, c='g', ls='--')
    # Legend
    axs[1,0].text(-590,1.5,'VEG', c='g', fontsize=12)
    axs[1,0].text(-590,1.25,'G', c='k', fontsize=12)
    axs[1,0].text(-590,1,'C', c='c', fontsize=12)
    # Title
    axs[1,0].set_title('C reservoirs', fontsize=14)
    axs[1,0].set_ylabel('Relative size', fontsize=14)
    axs[1,0].set_ylim([0, 2])
    #
    #### S SPECIES (subplot 6)
    axs[1,1].plot(state.time_myr,state.pyr/pars.pyr0, c='k')
    axs[1,1].plot(state.time_myr,state.gyp/pars.gyp0, c='c')
    # Legend
    axs[1,1].text(-590,1,'pyrite', c='k', fontsize=12)
    axs[1,1].text(-590,0.9,'gypsum', c='c', fontsize=12)
    # Title
    axs[1,1].set_title('S reservoirs', fontsize=14)
    axs[1,1].set_ylabel('Relative size', fontsize=14)
    axs[1,1].set_ylim([0.7, 1.4])
    #
    #### NUTRIENTS P N (subplot 7)
    #
    axs[1,2].plot(state.time_myr,state.P/pars.P0, c='b')
    axs[1,2].plot(state.time_myr,state.N/pars.N0, c='g')
    # Legend
    axs[1,2].text(-590,1.5,'P', c='b', fontsize=12)
    axs[1,2].text(-590,1,'N', c='g', fontsize=12)
    # Title
    axs[1,2].set_title('Nutrient reservoirs', fontsize=14)
    axs[1,2].set_ylabel('Relative size', fontsize=14)
    axs[1,2].set_ylim([0,3])
    #
    #### Forg and Fpy ratios (subplot 8)
    #
    axs[1,3].plot(state.time_myr,state.mocb/(state.mocb + state.mccb),'k')
    axs[1,3].plot(state.time_myr, state.mpsb/(state.mpsb + state.mgsb),'m')
    # Legend
    axs[1,3].text(-590,0.8,'mpsb', c='m', fontsize=12)
    axs[1,3].text(-590,0.6,'mocb', c='k', fontsize=12)
    # Title
    axs[1,3].set_ylabel(r'$f_{org}, f_{py}$', fontsize=14)
    axs[1,3].set_title(r'$f_{org}\ and\ f_{py} ratios$', fontsize=14)
    #
    #### d13C record (subplot 9)
    #
    # plot data comparison
    axs[2,0].plot(geochem_data['d13c_x'][0], geochem_data['d13c_y'][0], c=pc2, ls='-.')
    # plot this model
    axs[2,0].plot(state.time_myr, state.delta_mccb,c='k')
    #title
    axs[2,0].set_ylabel(r'$\delta^{13}C_{carb}$', fontsize=14)
    axs[2,0].set_title(r'$\delta^{13}C_{carb}$', fontsize=14)
    axs[2,0].set_ylim([-10,10])
    #
    #### d34S record (subplot 10)
    #
    # plot data comparison
    axs[2,1].plot(geochem_data['d34s_x'][0],geochem_data['d34s_y'][0], c=pc2, ls='-.')
    # plot this model
    axs[2,1].plot(state.time_myr,state.d34s_S,'k')
    #title
    axs[2,1].set_ylabel(r'$\delta^{34}S_{sw}$', fontsize=14)
    axs[2,1].set_title(r'$\delta^{34}S_{sw}$', fontsize=14)
    #
    #### Ocean 87Sr/86Sr  (subplot 11)
    #
    # plot data comparison
    axs[2,2].plot(geochem_data['sr_x'][0],geochem_data['sr_y'][0], c=pc2)
    # plot this model
    axs[2,2].plot(state.time_myr,state.delta_OSr,'k')
    #title
    axs[2,2].set_ylim([0.706, 0.71])
    axs[2,2].set_ylabel(r'$^{87}Sr/^{86}Sr\ seawater$')
    axs[2,2].set_title(r'$^{87}Sr/^{86}Sr\ seawater$')
    #
    ####SO4 (subplot 12)
    #
    # plot algeo data window comparison
    axs[2,3].plot(geochem_data['sconc_max_x'][0],geochem_data['sconc_max_y'][0], c=pc1)
    axs[2,3].plot(geochem_data['sconc_min_x'][0],geochem_data['sconc_min_y'][0], c=pc1)
    axs[2,3].plot(geochem_data['sconc_mid_x'][0],geochem_data['sconc_mid_y'][0], c=pc2)
    # plot fluid inclusion data comparison
    for i in np.arange(0, len(geochem_data['SO4_x']), 2):
        axs[2,3].plot([geochem_data['SO4_x'][i], geochem_data['SO4_x'][i]],
                      [geochem_data['SO4_y'][i], geochem_data['SO4_y'][i+1]], c=pc3)
    #plot this model
    axs[2,3].plot(state.time_myr,(state.S/pars.S0)*28,'k')
    #title
    axs[2,3].set_ylabel(r'$Marine SO_{4} (mM)$')
    axs[2,3].set_title(r'$Marine\ SO_{4}$')
    #
    ####O2 (%) (subplot 13)
    #
    # plot data comparison
    for i in np.arange(0, len(geochem_data['O2_x'])-1, 2):
        axs[3,0].plot([geochem_data['O2_x'][i], geochem_data['O2_x'][i]],
                      [geochem_data['O2_y'][i], geochem_data['O2_y'][i+1]], c=pc2)
    # plot this model
    axs[3,0].plot(state.time_myr,state.mro2*100,'k')
    #title
    axs[3,0].set_title(r'$Atmospheric\ O_{2}\ (\%)$')
    axs[3,0].set_ylabel(r'$Atmospheric\ O_{2}\ (\%)$')
    #
    #### CO2ppm (subplot 14)
    #
    ### plot data comparison
    # paleosol
    #axs[3,1].errorbar(geochem_data['paleosol_age'],geochem_data['paleosol_co2'],
    #                  geochem_data['paleosol_low'],geochem_data['paleosol_high'],
    #                   c=[0.4, 0.7, 0.7], ls=None)
    axs[3,1].scatter(geochem_data['paleosol_age'],
                     geochem_data['paleosol_co2'], color=pc1, edgecolors=pc1)
    #alkenone
    #axs[3,1].errorbar(geochem_data['alkenone_age'],geochem_data['alkenone_co2'],
    #                  geochem_data['alkenone_low'],geochem_data['alkenone_high'],
    #                   c=[0.4, 0.7, 0.4], ls=None)
    axs[3,1].scatter(geochem_data['alkenone_age'],
                     geochem_data['alkenone_co2'], color=pc2, edgecolors=pc2)
    #boron
    #axs[3,1].errorbar(geochem_data['boron_age'],geochem_data['boron_co2'],
    #                  geochem_data['boron_low'],geochem_data['boron_high'],
    #                   c=[0.4, 0.4, 0.4], ls=None)
    axs[3,1].scatter(geochem_data['boron_age'],
                     geochem_data['boron_co2'], color=pc3, edgecolors=pc3)
    #stomata
    #axs[3,1].errorbar(geochem_data['stomata_age'],geochem_data['stomata_co2'],
    #                  geochem_data['stomata_low'],geochem_data['stomata_high'],
    #                   c=[0.7, 0.7, 0.4], ls=None)
    axs[3,1].scatter(geochem_data['stomata_age'],
                     geochem_data['stomata_co2'], color=pc4, edgecolors=pc4)

    #liverwort
    #axs[3,1].errorbar(geochem_data['liverwort_age'],geochem_data['liverwort_co2'],
    #                  geochem_data['liverwort_low'],geochem_data['liverwort_high'],
    #                   c=[0.7, 0.7, 0.4], ls=None)
    axs[3,1].scatter(geochem_data['liverwort_age'],
                     geochem_data['liverwort_co2'], color=pc5, edgecolors=pc5)
    #phytane
    #axs[3,1].errorbar(geochem_data['phytane_age'],geochem_data['phytane_co2'],
    #                  geochem_data['phytane_low'],geochem_data['phytane_high'],
    #                   c=[0.7, 0.7, 0.4], ls=None)
    axs[3,1].scatter(geochem_data['phytane_age'],
                     geochem_data['phytane_co2'], color=pc6, edgecolors=pc6)
    # plot this model
    axs[3,1].plot(state.time_myr,state.rco2*280,'k')
    #title
    axs[3,1].set_yscale('log')
    axs[3,1].set_ylim([100, 10000])
    axs[3,1].set_xlabel('Time (Ma)')
    axs[3,1].set_ylabel(r'$Atmospheric\ CO_{2}\ (ppm)$')
    axs[3,1].set_title(r'$Atmospheric\ CO_{2}$')
    #
    ####TEMP (subplot 15)
    #
    #plot data comparison
    axs[3,2].plot(scotese_data['Scotese_2021_age'],scotese_data['Scotese_2021_GAT'], c=pc1)
    #plot this model
    axs[3,2].plot(state.time_myr,state.tempC,'k')
    #title
    axs[3,2].set_ylim([5, 40])
    axs[3,2].set_ylabel('GAST (C)')
    axs[3,2].set_title('GAST')
    #
    ####ICE LINE (subplot 16)
    #
    #plot iceline proxy
    axs[3,3].plot(geochem_data['paleolat_x'],geochem_data['paleolat_y'], c=pc1)
    #plot this model
    axs[3,3].plot(state.time_myr,state.iceline,'k')
    #set title
    axs[3,3].set_title('Ice line')
    axs[3,3].set_ylabel('Ice line')
    axs[3,3].set_ylim([0, 90])

    fig.tight_layout()
