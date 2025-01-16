import numpy as np
import scipy.io as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def pySCION_plot_worldgraphic(gridstate, INTERPSTACK):
    #### IPCC precip colorbar modified
    IPCC_pre = np.asarray([[223, 194, 125],
                           [246, 232, 195],
                           [245, 245, 245],
                           [199, 234, 229],
                           [128, 205, 193],
                           [53, 151, 143],
                           [1, 102, 94],
                           [0, 60, 48]]) / 255

    IPCC_temp = np.flipud(np.asarray( [[103, 0, 31],
                                       [178, 24, 43],
                                       [214, 96, 77],
                                       [244, 165, 130],
                                       [253, 219, 199],
                                       [247, 247, 247],
                                       [209, 229, 240],
                                       [146, 197, 222],
                                       [67, 147, 195],
                                       [33, 102, 172],
                                       [5, 48, 97 ]]) / 255)

    #### IPCC sequential
    IPCC_seq = np.asarray([[255, 255, 204],
                            [161, 218, 180],
                            [65, 182, 196],
                            [44, 127, 184],
                            [37, 52, 148]]) / 255

    #### IPCC sequential 2
    IPCC_seq_2 = np.asarray([[237, 248, 251],
                             [179, 205, 227],
                             [140, 150, 198],
                             [136, 86, 167],
                             [129, 15, 124]]) / 255

    #turn into cmaps for plotting
    cmap_IPCC_pre = mpl.colors.ListedColormap(IPCC_pre)
    cmap_IPCC_temp = mpl.colors.ListedColormap(IPCC_temp)
    cmap_IPCC_seq = mpl.colors.ListedColormap(IPCC_seq)
    cmap_IPCC_seq_2 = mpl.colors.ListedColormap(IPCC_seq_2)

    #### Proxy color chart
    pc1 = np.asarray([65, 195, 199]) / 255
    pc2 = np.asarray([73, 167, 187]) / 255
    pc3 = np.asarray([82, 144, 170]) / 255
    pc4 = np.asarray([88, 119, 149]) / 255
    pc5 = np.asarray([89, 96, 125]) / 255
    pc6 = np.asarray([82, 56, 100]) / 255

    #### make land and sea colormap with ice
    c_topo = (1/255) * np.asarray([[189, 231, 255],
                                   [79, 124, 0],
                                   [189, 155, 79],
                                   [166, 100, 78],
                                   [255, 255, 255],
                                   [113, 49, 63]])
    cmap_topo = mpl.colors.ListedColormap(c_topo)
    norm_topo = mpl.colors.BoundaryNorm([-1,0,1,2,3,4,5], cmap_topo.N)

    #####################################################################
    #################   Plot global variables   #########################
    #####################################################################


    for i in np.arange(4):
        f=i
        #### use multiple figures to plot all time slices
        if f == 0:
            choose_gridsubs = np.arange(0, 6, 1)
        elif f == 1:
            choose_gridsubs = np.arange(6, 11, 1)
        elif f == 2:
            choose_gridsubs = np.arange(11, 16, 1)
        else:
            choose_gridsubs = np.arange(16, 22, 1)


        data_crs = ccrs.Geodetic()
        poly_data_crs = ccrs.PlateCarree()
        nrows=5
        ncols=6
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24,12),
                            subplot_kw={'projection': ccrs.Robinson(0)})



        for ind, gridsub in enumerate(choose_gridsubs):

            ###topography
            this_TOPO = np.copy(gridstate.topo[:,:,gridsub].real)
            this_TOPO[this_TOPO<1000] = 0
            this_TOPO[this_TOPO>=3000] = 2
            this_TOPO[this_TOPO>=1000] = 1
            this_TOPO[np.isnan(this_TOPO)==1] = -1
            # make approximate ice mask and set to number 3
            approx_ice = np.copy(gridstate.Tair[:,:,gridsub].real)
            #if t=0 load present day config
            if gridsub == 21:
                approx_ice = np.copy(INTERPSTACK.Tair[:,:,8,21])
            #get ice at latitude
            approx_ice[approx_ice >= -10] = 0
            approx_ice[approx_ice < -10] = 1
            #add ice to topography
            this_TOPO[approx_ice == 1] = 3

            #### add sutures
            #this_suture = np.copy(gridstate.suture[:,:,gridsub])
            #this_TOPO[this_suture > 0] = 4 ;
            ##plot topo and title on first col
            axs[0,ind].imshow(this_TOPO, cmap=cmap_topo, norm=norm_topo, interpolation='none', transform=poly_data_crs)
            gl = axs[0,ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                        linewidth=2, color='gray', alpha=0.5, linestyle='--')
            #if ind == 0:
            axs[0, ind].set_title((str(gridstate.time_myr[0,0,gridsub].real * -1) + ' Ma'))


            ###Tair
            #Nan out ocean
            thisfield = np.copy(gridstate.Tair[:,:,gridsub].real)
            thisfield[INTERPSTACK.land[:,:,gridsub] == 0] = np.nan
            #plot Tair and title on first col
            axs[1, ind].imshow(thisfield, cmap=cmap_IPCC_temp, transform=poly_data_crs)
            gl = axs[1,ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                         linewidth=2, color='gray', alpha=0.5, linestyle='--')
            if ind == 0:
                axs[1, ind].set_title('Air Temp (C)')

            ###runoff (Q)
            #Nan out ocean
            thisfield = np.copy(gridstate.Q[:,:,gridsub].real)
            thisfield[INTERPSTACK.land[:,:,gridsub] == 0] = np.nan
            #plot runoff (Q) and title on first column
            axs[2, ind].imshow(thisfield, cmap=cmap_IPCC_pre, transform=poly_data_crs)
            gl = axs[2,ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=2, color='gray',
                                      alpha=0.5, linestyle='--')
            if ind == 0:
                axs[2, ind].set_title('Runoff (log mm/yr)')

            ###epsilon (erosion)
            #Nan out ocean
            thisfield = np.copy(gridstate.epsilon[:, :, gridsub].real)
            thisfield[INTERPSTACK.land[:,:,gridsub] == 0] = np.nan
            ###plot runoff (erosion) and title on first column
            axs[3, ind].imshow(thisfield, cmap=cmap_IPCC_seq_2, transform=poly_data_crs)
            gl = axs[3,ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                         linewidth=2, color='gray', alpha=0.5, linestyle='--')
            if ind == 0:
                axs[3, ind].set_title('Erosion (log t/km2/yr)')

            #silw (silicate weathering)
            #Nan out ocean
            thisfield = np.copy(gridstate.cw[:,:,gridsub].real)
            thisfield[INTERPSTACK.land[:,:,gridsub] == 0] = np.nan
            ###plot runoff (silw) and title on first colum n
            axs[4, ind].imshow(thisfield, cmap=cmap_IPCC_seq, transform=poly_data_crs)
            gl = axs[4,ind].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                         linewidth=2, color='gray', alpha=0.5, linestyle='--')
            if ind == 0:
                axs[4, ind].set_title('Silw (log t/km2/yr)')

            fig.tight_layout()

            fig.savefig('./results/worldgraphics_%s.pdf' % f)
