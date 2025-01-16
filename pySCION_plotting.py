import matplotlib.pyplot as plt
import pandas as pd
from cmcrameri import cm
import pickle
import numpy as np
import matplotlib as mpl

#scripts for plotting outputs from pySCION




def get_mean_distance(distance):

    # get fit in geological time periods

    Phanerozoic_time = {'Cambrian': [541.0, 485.4],
                    'Ordovician': [485.4, 443.4],
                    'Silurian': [443.4, 419.2],
                    'Devonian': [419.2, 358.9],
                    'Carboniferous': [358.9, 298.9],
                    'Permian': [298.9, 252.17],
                    'Triassic': [252.17, 201.3],
                    'Jurassic': [201.3, 145.0],
                    'Cretaceous': [145.0, 66.0],
                    'Palaeogene': [66.0, 23.03],
                    'Neogene': [23.03, 2.58]}

    period_mean_distance = np.zeros(len(Phanerozoic_time))
    for ind, (key, value) in enumerate(Phanerozoic_time.items()):

        start_ind = np.ceil(value[1])
        if start_ind > 540:
            start_ind = 540
        end_ind = np.ceil(value[0])
        start_ind = start_ind
        end_ind = end_ind
        distance_test = distance[int(start_ind): int(end_ind)]
        Phanerozoic_time[key].append([start_ind, end_ind])
        Phanerozoic_time[key].append(np.mean(distance_test))
        period_mean_distance[ind] = np.mean(distance_test)

    return period_mean_distance

# old summary cell to describe mean misfit in geological periods
#period_mean_dist = []
#summarised_results_dict = {}
#keys = list(Phanerozoic_time.keys())
#for ind, (key, value) in enumerate(results_dict.items()):
#    # print(key, value)
#    period_mean_dist.append(value[2])
#    summarised_results_dict[keys[ind]] = value[1]
## tmp_results = []
## for
## summarised_results_dict[keys[ind]] = np.mean(distance), period_mean_distance, distance


plot_time_start = -420
figsize = (12,4)
def get_results(filename):
    '''
    Load results from pySCION sensitivity runs.
    
    filename (string): name of file to load
    '''
    #load sensitivity results
    file = filename

    with open('./results/model_results/%s.obj' % file, 'rb') as file_:
        results = pickle.load(file_)
    if type(results) ==tuple:
        results = results[0]
    return results

def organise_results(results, vdMEER_GAT):

    x1_standard = vdMEER_GAT['GAT_degC'][1:422].values
    x2_results = np.asarray(results.T_gast)
    #reverse columns
    x2_results = np.flip(x2_results, axis=1)
    x2_trim_results = np.zeros((x2_results.shape[0], x1_standard.shape[0]))
    for ind, i in enumerate(x2_results):
        # x2_trim_results[ind,:] = i[:541]
        x2_trim_results[ind,:] = i[:421]

    return x1_standard, x2_trim_results

def make_global_temp_plot(results, gast_proxies, gast_distance, color, filename, 
                   default_results=False, plot_mean=False, save=False):
    '''
    temp data 0 = time
    temp data 1 = mean
    temp data 2 = min
    temp data 3 = max
    '''
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    gast_colours = ['#839DCA',
                    'indigo']

    for ind, gast_proxy in enumerate(gast_proxies):
        ax1.plot(gast_proxy[3].astype(float),
                gast_proxy[0].astype(float), lw=3, c=gast_colours[ind])

        ax1.fill_between(gast_proxy[3].astype(float),
                        gast_proxy[1].astype(float),
                        gast_proxy[2].astype(float), color=gast_colours[ind], alpha=0.25)
    
    #ax1.plot(temp_data['Ma'][1:542].values.astype(float)*-1,
    #         temp_data['GAT_degC'][1:542].values.astype(float), lw=3, c='#839DCA')
    #
    #ax1.fill_between(temp_data['Ma'][1:542].values.astype(float)*-1,
    #                 temp_data['GAT_degC.1'][1:542].values.astype(float),
    #                 temp_data['GAT_degC.2'][1:542].values.astype(float), color='#839DCA', alpha=0.25)

    #plot default model for comparison (just for temp)
    if default_results:
        ax1.plot(np.mean(default_results.time_myr, axis=0),
                 np.mean(default_results.T_gast, axis=0)+np.std(default_results.T_gast, axis=0),
                 c='k', alpha=0.2
                )
        ax1.plot(np.mean(default_results.time_myr, axis=0),
                 np.mean(default_results.T_gast, axis=0)-np.std(default_results.T_gast, axis=0),
                 c='k', alpha=0.2
                )
        ax1.fill_between(np.mean(default_results.time_myr, axis=0),
                         np.mean(default_results.T_gast, axis=0)+np.std(default_results.T_gast, axis=0),
                         np.mean(default_results.T_gast, axis=0)-np.std(default_results.T_gast, axis=0),
                         color='k', alpha=0.1
                        )

    #plot individual model runs    
    for ind, model_run in enumerate(results.time_myr):
        #print('here')
        if ind % 10 == 0:
            ax1.plot(results.time_myr[ind], results.T_gast[ind], c=color, alpha=0.2)

    if plot_mean:
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.T_gast, axis=0), c='forestgreen')
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.T_gast, axis=0)+np.std(results.T_gast, axis=0), c='forestgreen', alpha=0.75)
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.T_gast, axis=0)-np.std(results.T_gast, axis=0), c='forestgreen', alpha=0.75)
        #ax1.fill_between(np.mean(results.time_myr, axis=0),
        #                 np.mean(results.T_gast, axis=0)+np.std(results.T_gast, axis=0),
        #                 np.mean(results.T_gast, axis=0)-np.std(results.T_gast, axis=0),
        #                 color=color, alpha=0.1
        #                )

    ax1.plot(gast_proxies[0][3].astype(float), gast_distance, c='k')
    ax1.set_xlim([plot_time_start,0])
    ax1.set_ylim([0,40])
    ax1.set_ylabel('GAST (°C)', fontsize=16)
    ax1.set_xlabel('Time (Ma)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title(r'Global Average Surface Temperature', fontsize=16)
    ax1.grid()
    fig.suptitle(f'(a) {filename}', fontsize=16)
    fig.tight_layout()
    if save:
        fig.savefig('./results/figures/Phanerozoic_%s_TEMP_fill.pdf' % filename)
    return

def make_equtorial_temp_plot(results, o2_dict, sat_proxy, sat_distance, color, filename,
                    plot_mean=False, save=False):

    # o2_dict is the heatmap, not cuve of best fit
    times = []
    vals = []
    for i in o2_dict:
        vals.append(o2_dict[i][0].values)
        times.append(o2_dict[i][1].values)

    vals = np.concatenate(vals).ravel()
    times = np.concatenate(times).ravel()
    times = times[~np.isnan(vals)]*-1
    vals = vals[~np.isnan(vals)]

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    hist = ax1.hist2d(times, vals, bins=(200, 100), norm=mpl.colors.LogNorm(),
                cmap=cm.bilbao, alpha=0.2)#bamako_r)#, cmin = 1)

    # plot line of best fit (faintly)
    ax1.plot(sat_proxy[3].astype(float),
             sat_proxy[0].astype(float), lw=3, c='#839DCA')

    ax1.fill_between(sat_proxy[3].astype(float),
                     sat_proxy[1].astype(float),
                     sat_proxy[2].astype(float), color='#839DCA', alpha=0.25)

    #plot individual model runs    
    for ind, model_run in enumerate(results.time_myr):
        #print('here')
        if ind % 10 == 0:
            ax1.plot(results.time_myr[ind], results.sat_tropical[ind], c=color, alpha=0.2)
    if plot_mean:
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.sat_tropical, axis=0), c='forestgreen')
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.sat_tropical, axis=0)+np.std(results.sat_tropical, axis=0), c='forestgreen', alpha=0.75)
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.sat_tropical, axis=0)-np.std(results.sat_tropical, axis=0), c='forestgreen', alpha=0.75)
        #ax1.fill_between(np.mean(results.time_myr, axis=0),#
        #                 np.mean(results.sat_tropical, axis=0)+np.std(results.sat_tropical, axis=0),
        #                 np.mean(results.sat_tropical, axis=0)-np.std(results.sat_tropical, axis=0),
        #                 color=color, alpha=0.1
        #                )
    
    ax1.plot(sat_proxy[3].astype(float), sat_distance, c='k')
    ax1.set_xlim([plot_time_start,0])
    ax1.set_ylim([0,60])
    ax1.set_ylabel('EQAST (C)', fontsize=16)
    ax1.set_xlabel('Time (Ma)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title(r'Equatorial Average Surface Temperature', fontsize=16)
    ax1.grid()
    fig.suptitle(f'(a) {filename}', fontsize=16)
    cax = fig.add_axes([0.80,
                    0.72,
                    0.15,
                    0.05])
    fig.colorbar(hist[3],cax=cax, orientation='horizontal', label='Count')
    fig.tight_layout()
    if save:
        fig.savefig('./results/figures/Phanerozoic_%s_EQ_TEMP_fill.pdf' % filename)
    return

def make_ice_plot(results, ice_data, ice_proxy, ice_distance, color, filename,
                   plot_mean=False, save=False):
    
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    #mat = ax1.matshow(ice_data, aspect=1.25, cmap=cm.lapaz_r, extent=[0, -540, 0, 90],
    #                 vmax=10)
    mat = ax1.matshow(ice_data, aspect=1.25, cmap=cm.bilbao, alpha=0.4, extent=[0, -540, 0, 90],
                      vmax=10)


    # plot line of maximum ice extent (assuming >3 ice deposits)
    ax1.plot(ice_proxy[1].astype(float),
             ice_proxy[0].astype(float), lw=3, c='#839DCA')


    for ind, model_run in enumerate(results.time_myr):
        if ind % 40 == 0:
            ax1.plot(results.time_myr[ind], results.iceline[ind][:, 1], color=color, alpha=0.1)
    #plot mean stds
    ice_upper_std = np.mean(results.iceline, axis=0)+np.std(results.iceline, axis=0)
    ice_lower_std = np.mean(results.iceline, axis=0)-np.std(results.iceline, axis=0)
    ice_mean = np.mean(results.iceline, axis=0)
#
    ice_upper_std[ice_upper_std > 90] = 90
    ice_lower_std[ice_lower_std > 90] = 90
    ice_mean[ice_mean > 90] = 90


    
    #try:
    if plot_mean:
        ax1.plot(np.mean(results.time_myr, axis=0), ice_mean[:, 1], c='forestgreen')
        ax1.plot(np.mean(results.time_myr, axis=0),
                 ice_upper_std[:, 1], c='forestgreen', alpha=0.75)
        ax1.plot(np.mean(results.time_myr, axis=0),
                 ice_lower_std[:, 1], c='forestgreen', alpha=0.75)

        lss = [':', '-', '-.']
        for ind, i in enumerate(np.arange(np.shape(ice_mean)[1])):
            ax1.plot(np.mean(results.time_myr, axis=0),
                     ice_mean[:, i], ls=lss[ind], c='forestgreen')
            #ax1.plot(np.mean(results.time_myr, axis=0), ice_upper_std[:,1], c=color)
            #ax1.plot(np.mean(results.time_myr, axis=0), ice_lower_std[:,1], c=color)
            #ax1.fill_between(np.mean(results.time_myr, axis=0),
            #            ice_upper_std[:,1],
            #            ice_lower_std[:,1],
            #            color=color, alpha=0.1
            #            )
        #except:
        #    ax1.plot(np.mean(results.time_myr, axis=0), ice_mean, c=color)
        #    ax1.plot(np.mean(results.time_myr, axis=0), ice_upper_std, c=color)
        #    ax1.plot(np.mean(results.time_myr, axis=0), ice_lower_std, c=color)
        #    
        #    ax1.fill_between(np.mean(results.time_myr, axis=0),
        #                    ice_upper_std,
        #                    ice_lower_std,
        #                    color=color, alpha=0.1
        #                    )
        
    ax1.plot(ice_proxy[1].astype(float), ice_distance, c='k')
    ax1.set_xlim([plot_time_start,0])
    ax1.set_ylim([0,90])
    ax1.set_ylabel('Latitude (°)', fontsize=16)
    ax1.set_xlabel('Time (Ma)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title(r'Iceline', fontsize=16)
    ax1.grid()
    fig.suptitle(f'(b) {filename}', fontsize=16)
    cax = fig.add_axes([0.815,
                        0.3,
                        0.15,
                        0.05])
    fig.colorbar(mat, cax=cax, orientation='horizontal', label='Count')
    fig.tight_layout()

    if save:
        fig.savefig('./results/figures/Phanerozoic_%s_ICE.svg' % filename,
                    transparent=True)
    return

def make_CO2_plot(results, co2, co2_proxy, co2_distance, color, filename,
                  plot_mean=False, save=False):

    #cmap = cm.tokyo
    #colors = cmap(np.linspace(0, 1, 8))
    proxy_colour = '#450903'
    proxy_alpha = 0.2
    pc1 = proxy_colour#colors[0]
    pc2 = proxy_colour#colors[1]
    pc3 = proxy_colour#colors[3]
    pc4 = proxy_colour#colors[4]
    pc5 = proxy_colour#colors[5]
    pc6 = proxy_colour#colors[6]
    pc7 = proxy_colour#colors[7]

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    ax1.errorbar(co2['paleosol_age'].ravel(),
                 co2['paleosol_co2'].ravel(),
                 yerr=np.abs(co2['paleosol_high'].ravel() - co2['paleosol_co2'].ravel(),
                             co2['paleosol_co2'].ravel() - co2['paleosol_low'].ravel()),
                 fmt='o',  color=pc1, alpha=proxy_alpha, label='Palaeosol', zorder=1)

    ax1.errorbar(co2['alkenone_age'].ravel(),
                 co2['alkenone_co2'].ravel(),
                 yerr=(co2['alkenone_high'].ravel() - co2['alkenone_co2'].ravel(),
                       co2['alkenone_co2'].ravel() - co2['alkenone_low'].ravel()),
                 fmt='v',  color=pc2, alpha=proxy_alpha, label='Alkenone', zorder=1)

    ax1.errorbar(co2['boron_age'].ravel(),
                 co2['boron_co2'].ravel(),
                 yerr=(co2['boron_high'].ravel() - co2['boron_co2'].ravel(),
                       co2['boron_co2'].ravel() - co2['boron_low'].ravel()),
                 fmt='s',  color=pc3, alpha=proxy_alpha, label='Boron', zorder=1)

    ax1.errorbar(co2['stomata_age'].ravel(),
                 co2['stomata_co2'].ravel(),
                 yerr=(co2['stomata_high'].ravel() - co2['stomata_co2'].ravel(),
                      co2['stomata_co2'].ravel() - co2['stomata_low'].ravel()),
                 fmt='P',  color=pc4, alpha=proxy_alpha, label='Stomata', zorder=1)

    ax1.errorbar(co2['liverwort_age'].ravel(),
                 co2['liverwort_co2'].ravel(),
                 yerr=(co2['liverwort_high'].ravel() - co2['liverwort_co2'].ravel(),
                      co2['liverwort_co2'].ravel() - co2['liverwort_low'].ravel()),
                 fmt='x',  color=pc5, alpha=proxy_alpha, label='Liverwort', zorder=1)

    ax1.errorbar(co2['phytane_age'].ravel(),
                 co2['phytane_co2'].ravel(),
                 yerr=(co2['phytane_high'].ravel() - co2['phytane_co2'].ravel(),
                      co2['phytane_co2'].ravel() - co2['phytane_low'].ravel()),
                 fmt='D',  color=pc6, alpha=proxy_alpha, label='Phytane', zorder=1)

    ax1.errorbar(co2['phytoplankton_age'].ravel(),
                 co2['phytoplankton_co2'].ravel(),
                 yerr=(co2['phytoplankton_high'].ravel() - co2['phytoplankton_co2'].ravel(),
                       co2['phytoplankton_co2'].ravel() - co2['phytoplankton_low'].ravel()),
                 fmt='^',  color=pc7, alpha=proxy_alpha, label='Phytoplankton', zorder=1)

    # plot line of best fit (faintly)
    ax1.plot(co2_proxy[3].astype(float),
             co2_proxy[0].astype(float), lw=3, c='#839DCA')

    ax1.fill_between(co2_proxy[3].astype(float),
                     co2_proxy[1].astype(float),
                     co2_proxy[2].astype(float), color='#839DCA', alpha=0.25)

    for ind, model_run in enumerate(results.time_myr):
        if ind % 10 == 0:
            ax1.plot(results.time_myr[ind], results.co2ppm[ind], color=color, alpha=0.1)
    if plot_mean:
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.co2ppm, axis=0),  c = 'forestgreen')
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.co2ppm, axis=0)+np.std(results.co2ppm, axis=0), c = 'forestgreen', alpha = 0.75)
        ax1.plot(np.mean(results.time_myr, axis=0), np.mean(results.co2ppm, axis=0)-np.std(results.co2ppm, axis=0), c = 'forestgreen', alpha = 0.75)
        #ax1.fill_between(np.mean(results.time_myr, axis=0),
        #                 np.mean(results.co2ppm, axis=0)+np.std(results.co2ppm, axis=0),
        #                 np.mean(results.co2ppm, axis=0)-np.std(results.co2ppm, axis=0),
        #                 color=color, alpha=0.1)
                        
    ax1.plot(co2_proxy[3].astype(float), co2_distance, c='k')

    ax1.set_yscale('log')
    ax1.set_ylim([100, 10000])
    ax1.set_xlim([plot_time_start,0])
    ax1.legend(loc='lower left')
    ax1.set_xlabel('Time (Ma)', fontsize=16)
    ax1.set_ylabel(r'$Atmospheric\ CO_{2}\ (ppm)$', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title(r'$Atmospheric\ CO_{2}$', fontsize=16)
    ax1.grid()
    fig.suptitle(f'(c) {filename}', fontsize=16)
    fig.tight_layout()

    if save:
        fig.savefig('./results/figures/Phanerozoic_%s_CO2.pdf' % filename)
    return



