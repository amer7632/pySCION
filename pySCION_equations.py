import numpy as np
from scipy.stats import norm

def pySCION_equations(t, y, pars, forcings, sensanal, tuning, interpstack, lipstack,
                         model_pars, workingstate, stepnumber, gridstate,
                         sensparams, erosion_pars):

    #################################################################################################
    #                                                                                               #
    #              110111010                                                                        #
    #           111010-1-----101                                                                    #
    #        1011111---------101111                                                                 #
    #      11011------------------101         pySCION: Spatial Continuous Integration               #
    #     111-----------------10011011        Earth Evolution Model (for python!)                   #
    #    1--10---------------1111011111                                                             #
    #    1---1011011---------1010110111       Coded by Benjaw J. W. Mills                         #
    #    1---1011000111----------010011       email: b.mills@leeds.ac.uk                            #
    #    1----1111011101----------10101       Translated into python by Andrew Merdith              #
    #     1----1001111------------0111        Model equations file                                  #
    #      1----1101-------------1101         contains flux and reservoir equations                 #
    #        1--111----------------1                                                                #
    #           1---------------1                                                                   #
    #               111011011                                                                       #
    #################################################################################################

    # Numpy divides for all elements in an array, even the ones that aren't
    # selected by the where (which are 0 and throwing the errow).
    np.seterr(divide = 'ignore')
    # Setup dict to store results
    new_data_dict = {}
    new_data_dict['t'] = t
    # Setup dy array
    dy = np.zeros(21)
    # Get variables from Y to make working easier
    new_data_dict['P'] = y[0]
    new_data_dict['O'] = y[1]
    new_data_dict['A'] = y[2]
    new_data_dict['S'] = y[3]
    new_data_dict['G'] = y[4]
    new_data_dict['C'] = y[5]
    new_data_dict['pyr'] = y[6]
    new_data_dict['gyp'] = y[7]
    # TEMP = y[8]
    # CAL = y[9]
    new_data_dict['N'] = y[10]
    new_data_dict['OSr'] = y[17]
    new_data_dict['SSr'] = y[19]
    new_data_dict['dSSr'] = y[20]/y[19]

    # Geological time in Ma
    new_data_dict['t_geol'] = t*(1e-6)
    
    # lip degasssing
    # still playing with this, but two lip options
    # default is siberian traps and is 'always' on
    # other is to have all lips included
    # if all lips are included it overwrites teh siberian traps calculation

    if pars.lip_degass == 0:
        lip_degass = 0
        for ind, lip_age in enumerate(forcings.D_lip_ages):
            tmp_mid_time = forcings.D_lip_ages[ind] + forcings.D_lip_durations[ind]/2

            tmp_lip_co2 = forcings.D_lip_fluxes[ind]*norm.pdf(new_data_dict['t_geol'], tmp_mid_time, forcings.D_lip_durations[ind]) \
                / norm.pdf(tmp_mid_time, tmp_mid_time, forcings.D_lip_durations[ind])

            lip_degass += tmp_lip_co2
    
    else:
        # LIP degassing
        time_siberian = -252
        flux_siberian = 1e13

        
        lip_degass = flux_siberian*norm.pdf(new_data_dict['t_geol'], time_siberian, 1) \
            / norm.pdf(time_siberian, time_siberian, 1)
        
    # this should possibly get moved to a dict
    lip_co2_d13c = -5
    # Calculate isotopic fractionation of reservoirs
    new_data_dict['delta_G'] = y[11]/y[4]
    new_data_dict['delta_C'] = y[12]/y[5]
    new_data_dict['delta_gyp']  = y[14]/y[7]
    new_data_dict['delta_pyr']  = y[13]/y[6]

    # Atmospheric fraction of total CO2, atfrac(A)
    atfrac0 = 0.01614
    # Xonstant
    # atfrac = 0.01614
    # Variable
    atfrac = atfrac0 * (new_data_dict['A']/pars.A0)

    # Calculations for pCO2, pO2
    new_data_dict['rco2'] = (new_data_dict['A']/pars.A0)*(atfrac/atfrac0)
    co2atm = new_data_dict['rco2']*(280e-6)
    co2ppm = new_data_dict['rco2']*280

    # Mixing ratio of oxygen (not proportional to O reservoir)
    new_data_dict['mro2'] = (new_data_dict['O']/pars.O0)/((new_data_dict['O']/pars.O0) + pars.copsek16)
    # Relative moles of oxygen
    new_data_dict['ro2'] =  new_data_dict['O']/pars.O0

    #####################################################################
    ############   Interpolate forcings for this timestep   #############
    #####################################################################

    # COPSE Reloaded forcing set
    # Interp1d in the format: E_reloaded = interp1d(1e6 * forcings.t, 
    #                                               forcings.E)(t)
    # and is equivalent to:
    # E_interp1d = interp1d(1e6 * forcings.t, forcings.E)
    # E_reloaded = E_interp1d(t)
    E_reloaded = forcings.E_reloaded_INTERP(t)
    W_reloaded = forcings.W_reloaded_INTERP(t)
    # Additional forcings, basalt and granite fractions
    Ba = forcings.Ba_reloaded_INTERP(t)
    Ga = forcings.Ga_reloaded_INTERP(t)

    # Select degassing here
    if pars.degass_test == 0:
        D_combined_min = forcings.D_complete_min_INTERP(new_data_dict['t_geol'])#0.9
        D_combined_mean = forcings.D_complete_mean_INTERP(new_data_dict['t_geol'])#1
        D_combined_max = forcings.D_complete_max_INTERP(new_data_dict['t_geol'])#1.1
    else:
        D_combined_min = 1
        D_combined_mean = 1
        D_combined_max = 1


    ######################################################################
    ####################  Choose forcing functions  ######################
    ######################################################################
    
    # degassing
    new_data_dict['degass'] = D_combined_mean
    new_data_dict['lip_degass'] = lip_degass
    # W?
    new_data_dict['W'] = W_reloaded
    #EVO?
    new_data_dict['evo'] = E_reloaded
    
    #CPLand?
    new_data_dict['cpland'] = 1
    # BForcing is gone now!
    #Bforcing = interp1d([-1000, -150, 0],[0.75, 0.75, 1])(t_geol)
    #Bforcing = interp1d([-1000, -120,-100,-50, 0],[0.82, 0.82, 0.85, 0.90, 1])(t_geol)
    new_data_dict['Bforcing'] = 1
    # basalt and granite area
    new_data_dict['bas_area'] = Ba
    new_data_dict['gran_area'] = Ga

    # Contribution to CO2 prior to plants
    new_data_dict['preplant'] = 1/7
    capdelS = 27
    capdelC_land = 27
    capdelC_marine = 35
    
    # Shoreline
    shoreline = forcings.shoreline_INTERP(new_data_dict['t_geol'])

    # bioturbation forcing
        
    f_biot = forcings.f_biot_INTERP(t)
    cb = forcings.cb_INTERP(f_biot)
    
    ######################################################################
    ######################   Sensitivity analysis  #######################
    ######################################################################

    # All sensparams vary between [-1 +1]
    if sensanal.key == 1:
        # Vary degassing between upper and lower bounds
        if sensparams.randminusplus1 > 0:
            new_data_dict['degass'] = (1 - sensparams.randminusplus1)*new_data_dict['degass'] + sensparams.randminusplus1*D_combined_max
        else:
            new_data_dict['degass'] = (1 + sensparams.randminusplus1)*new_data_dict['degass'] - sensparams.randminusplus1*D_combined_min

        # Simple +/- 20% variation
        new_data_dict['bas_area'] = new_data_dict['bas_area'] * (1 + 0.2*sensparams.randminusplus2)
        new_data_dict['gran_area'] = new_data_dict['gran_area'] * (1 + 0.2*sensparams.randminusplus3)

        # Preplant varies from 1/4 to 1/7, but leave fixed for now
        new_data_dict['preplant'] = 1/7 #1 / ( 4 + 3*sensparams.randminusplus4)
        capdelS = 30 + 10*sensparams.randminusplus5
        capdelC_land = 25 + 5*sensparams.randminusplus6
        capdelC_marine = 30 + 5*sensparams.randminusplus7
        #lip_co2_d13c = -5 + 5*sensparams.randminusplus8    

    ######################################################################
    ##################### Spatial fields from stack   ####################
    ######################################################################

    # Find past and future keyframes
    key_future_time = np.min(interpstack.time[(interpstack.time - new_data_dict['t_geol']) >=0])
    
    # Because 600 Ma is older than our times, it can return an empty array
    # which breaks np.max, so we check
    temp_key_past_time = interpstack.time[interpstack.time-new_data_dict['t_geol'] <=0]
    if not temp_key_past_time.size > 0:
        # If empty, i.e. between 600 and 540 Ma
        key_past_time = key_future_time
    else:
        key_past_time = np.max(temp_key_past_time)

    # Find keyframe indexes and fractional contribution
    key_past_index = np.argwhere(interpstack.time == key_past_time )[0][0]
    key_future_index = np.argwhere(interpstack.time == key_future_time )[0][0]
    dist_to_past = np.abs(key_past_time - new_data_dict['t_geol'])
    dist_to_future = np.abs(key_future_time - new_data_dict['t_geol'])

    # Fractional contribution of each keyframe
    # only if time = 0?
    if dist_to_past + dist_to_future == 0:
        contribution_past = 1
        contribution_future = 0
    else:
        contribution_past = dist_to_future / ( dist_to_past + dist_to_future )
        contribution_future = dist_to_past / ( dist_to_past + dist_to_future )

    # Intrepolate keyframe CO2 concentrations to generate keyframe fields
    # Find INTERPSTACK keyframes using model CO2
    if co2ppm > 112000:
        print('too bloody hot clive', co2ppm)
        co2ppm = 111999

    key_upper_co2 = np.min(interpstack.co2[(interpstack.co2 - co2ppm) >= 0])
    
    # If CO2 is between 0 and 10 it throws a value error. So this makes it grab
    # the smallest co2 value
    try:
        key_lower_co2 = np.max(interpstack.co2[(interpstack.co2 - co2ppm) <= 0])
    except ValueError:
        key_lower_co2 = 10
    
    # Find keyframe indexes and fractional contribution
    key_upper_co2_index = np.argwhere(interpstack.co2 == key_upper_co2 )[0][0]
    key_lower_co2_index = np.argwhere(interpstack.co2 == key_lower_co2 )[0][0]
    dist_to_upper = np.abs(key_upper_co2 - co2ppm)
    dist_to_lower = np.abs(key_lower_co2 - co2ppm)

    # Fractional contribution of each keyframe
    if dist_to_upper + dist_to_lower == 0:
        contribution_lower = 1
        contribution_upper = 0
    else:
        contribution_upper = dist_to_lower/(dist_to_upper + dist_to_lower)
        contribution_lower = dist_to_upper/(dist_to_upper + dist_to_lower)

    ######################################################################
    ###### Create time keyframes using CO2 keyfield contributions   ######
    ######################################################################

    # Runoff
    runoff_past = (contribution_upper*np.copy(interpstack.runoff[:,:,key_upper_co2_index,key_past_index]) + \
        contribution_lower*np.copy(interpstack.runoff[:,:,key_lower_co2_index,key_past_index]))
    runoff_future = (contribution_upper*np.copy(interpstack.runoff[:,:,key_upper_co2_index,key_future_index]) + \
        contribution_lower*np.copy(interpstack.runoff[:,:,key_lower_co2_index,key_future_index]))

    # Tair
    Tair_past = (contribution_upper * np.copy(interpstack.Tair[:, :, key_upper_co2_index, key_past_index]) + \
                     contribution_lower * np.copy(interpstack.Tair[:, :, key_lower_co2_index, key_past_index]))
    Tair_future = (contribution_upper * np.copy(interpstack.Tair[:, :, key_upper_co2_index, key_future_index]) + \
                   contribution_lower * np.copy(interpstack.Tair[:, :, key_lower_co2_index, key_future_index]))

    # Time keyframes that don't depend on CO2
    # Topography
    topo_past = np.copy(interpstack.topo[:,:,key_past_index])

    # Slope
    tslope_past = np.copy(interpstack.slope[:,:,key_past_index])
    tslope_future = np.copy(interpstack.slope[:,:,key_future_index])

    # Arcs
    arc_past = np.copy(interpstack.arc[:,:,key_past_index]) #just for the output
    arc_enhancement_past = np.copy(interpstack.arc_enhancement[:,:,key_past_index])
    arc_enhancement_future = np.copy(interpstack.arc_enhancement[:,:,key_future_index])

    # Sutures
    suture_past = np.copy(interpstack.suture[:,:,key_past_index])
    suture_enhancement_past = np.copy(interpstack.suture_enhancement[:,:,key_past_index])
    suture_enhancement_future = np.copy(interpstack.suture_enhancement[:,:,key_future_index])

    # Relict arcs
    relict_past = np.copy(interpstack.relict_arc[:,:,key_past_index])
    relict_arc_enhancement_past = np.copy(interpstack.relict_arc_enhancement[:,:,key_past_index])
    relict_arc_enhancement_future = np.copy(interpstack.relict_arc_enhancement[:,:,key_future_index])


    # lip past maps for outputting grids
    afar_past = lipstack.afar[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.afar_start) 
    bunbury_past = lipstack.bunbury[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.bunbury_start)
    camp_past = lipstack.camp[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.camp_start)
    caribbean_colombian_past = lipstack.caribbean_colombian[:, :,key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.caribbean_colombian_start)
    cimp_a_past = lipstack.cimp_a[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.cimp_a_start)
    cimp_b_past = lipstack.cimp_b[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.cimp_b_start)
    columbia_river_past = lipstack.columbia_river[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.columbia_river_start)
    comei_past = lipstack.comei[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.comei_start)
    deccan_past = lipstack.deccan[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.deccan_start)
    emeishan_past = lipstack.emeishan[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.emeishan_start)
    equamp_past = lipstack.equamp[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.equamp_start)
    ferrar_past = lipstack.ferrar[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.ferrar_start)
    franklin_past = lipstack.franklin[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.franklin_start)
    gunbarrel_past = lipstack.gunbarrel[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.gunbarrel_start)
    halip_past = lipstack.halip[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.halip_start)
    irkutsk_past = lipstack.irkutsk[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.irkutsk_start)
    kalkarindji_past = lipstack.kalkarindji[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.kalkarindji_start)
    karoo_past = lipstack.karoo[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.karoo_start)
    kola_dnieper_past = lipstack.kola_dnieper[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.kola_dnieper_start)
    madagascar_past = lipstack.madagascar[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.madagascar_start)
    magdalen_past = lipstack.magdalen[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.magdalen_start)
    mundine_well_past = lipstack.mundine_well[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.mundine_well_start)
    naip_past = lipstack.naip[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.naip_start)
    nw_australia_margin_past = lipstack.nw_australia_margin[:, :,key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.nw_australia_margin_start)
    panjal_past = lipstack.panjal[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.panjal_start)
    parana_etendeka_past = lipstack.parana_etendeka[:, :,key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.parana_etendeka_start)
    qiangtang_past = lipstack.qiangtang[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.qiangtang_start)
    seychelles_past = lipstack.seychelles[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.seychelles_start)
    siberia_past = lipstack.siberia[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.siberia_start)
    suordakh_past = lipstack.suordakh[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.suordakh_start)
    swcuc_past = lipstack.swcuc[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.swcuc_start)
    tarim_past = lipstack.tarim[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.tarim_start)
    trap_past = lipstack.trap[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.trap_start)
    vilyui_past = lipstack.vilyui[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.vilyui_start)
    wichita_past = lipstack.wichita[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.wichita_start)

    lip_past = np.sum((
        afar_past, bunbury_past, camp_past,
        caribbean_colombian_past, cimp_a_past,
        cimp_b_past, columbia_river_past,
        comei_past, deccan_past,
        emeishan_past, equamp_past,
        ferrar_past, franklin_past,
        gunbarrel_past, halip_past,
        irkutsk_past, kalkarindji_past,
        karoo_past, kola_dnieper_past,
        madagascar_past, magdalen_past,
        mundine_well_past, naip_past,
        nw_australia_margin_past, panjal_past,
        parana_etendeka_past, qiangtang_past,
        seychelles_past, siberia_past,
        suordakh_past, swcuc_past,
        tarim_past, trap_past,
        vilyui_past, wichita_past),
        axis=0)

    # lip enhancement past
    afar_past_enhancement = lipstack.afar_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.afar_start) 
    bunbury_past_enhancement = lipstack.bunbury_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.bunbury_start)
    camp_past_enhancement = lipstack.camp_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.camp_start)
    caribbean_colombian_past_enhancement = lipstack.caribbean_colombian_enhancement[:, :,key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.caribbean_colombian_start)
    cimp_a_past_enhancement = lipstack.cimp_a_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.cimp_a_start)
    cimp_b_past_enhancement = lipstack.cimp_b_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.cimp_b_start)
    columbia_river_past_enhancement = lipstack.columbia_river_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.columbia_river_start)
    comei_past_enhancement = lipstack.comei_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.comei_start)
    deccan_past_enhancement = lipstack.deccan_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.deccan_start)
    emeishan_past_enhancement = lipstack.emeishan_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.emeishan_start)
    equamp_past_enhancement = lipstack.equamp_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.equamp_start)
    ferrar_past_enhancement = lipstack.ferrar_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.ferrar_start)
    franklin_past_enhancement = lipstack.franklin_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.franklin_start)
    gunbarrel_past_enhancement = lipstack.gunbarrel_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.gunbarrel_start)
    halip_past_enhancement = lipstack.halip_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.halip_start)
    irkutsk_past_enhancement = lipstack.irkutsk_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.irkutsk_start)
    kalkarindji_past_enhancement = lipstack.kalkarindji_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.kalkarindji_start)
    karoo_past_enhancement = lipstack.karoo_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.karoo_start)
    kola_dnieper_past_enhancement = lipstack.kola_dnieper_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.kola_dnieper_start)
    madagascar_past_enhancement = lipstack.madagascar_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.madagascar_start)
    magdalen_past_enhancement = lipstack.magdalen_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.magdalen_start)
    mundine_well_past_enhancement = lipstack.mundine_well_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.mundine_well_start)
    naip_past_enhancement = lipstack.naip_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.naip_start)
    nw_australia_margin_past_enhancement = lipstack.nw_australia_margin_enhancement[:, :,key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.nw_australia_margin_start)
    panjal_past_enhancement = lipstack.panjal_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.panjal_start)
    parana_etendeka_past_enhancement = lipstack.parana_etendeka_enhancement[:, :,key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.parana_etendeka_start)
    qiangtang_past_enhancement = lipstack.qiangtang_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.qiangtang_start)
    seychelles_past_enhancement = lipstack.seychelles_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.seychelles_start)
    siberia_past_enhancement = lipstack.siberia_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.siberia_start)
    suordakh_past_enhancement = lipstack.suordakh_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.suordakh_start)
    swcuc_past_enhancement = lipstack.swcuc_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.swcuc_start)
    tarim_past_enhancement = lipstack.tarim_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.tarim_start)
    trap_past_enhancement = lipstack.trap_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.trap_start)
    vilyui_past_enhancement = lipstack.vilyui_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.vilyui_start)
    wichita_past_enhancement = lipstack.wichita_enhancement[:, :, key_past_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.wichita_start)

    # lip enhancement future
    afar_future_enhancement = lipstack.afar_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.afar_start) 
    bunbury_future_enhancement = lipstack.bunbury_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.bunbury_start)
    camp_future_enhancement = lipstack.camp_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.camp_start)
    caribbean_colombian_future_enhancement = lipstack.caribbean_colombian_enhancement[:, :,key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.caribbean_colombian_start)
    cimp_a_future_enhancement = lipstack.cimp_a_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.cimp_a_start)
    cimp_b_future_enhancement = lipstack.cimp_b_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.cimp_b_start)
    columbia_river_future_enhancement = lipstack.columbia_river_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.columbia_river_start)
    comei_future_enhancement = lipstack.comei_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.comei_start)
    deccan_future_enhancement = lipstack.deccan_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.deccan_start)
    emeishan_future_enhancement = lipstack.emeishan_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.emeishan_start)
    equamp_future_enhancement = lipstack.equamp_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.equamp_start)
    ferrar_future_enhancement = lipstack.ferrar_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.ferrar_start)
    franklin_future_enhancement = lipstack.franklin_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.franklin_start)
    gunbarrel_future_enhancement = lipstack.gunbarrel_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.gunbarrel_start)
    halip_future_enhancement = lipstack.halip_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.halip_start)
    irkutsk_future_enhancement = lipstack.irkutsk_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.irkutsk_start)
    kalkarindji_future_enhancement = lipstack.kalkarindji_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.kalkarindji_start)
    karoo_future_enhancement = lipstack.karoo_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.karoo_start)
    kola_dnieper_future_enhancement = lipstack.kola_dnieper_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.kola_dnieper_start)
    madagascar_future_enhancement = lipstack.madagascar_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.madagascar_start)
    magdalen_future_enhancement = lipstack.magdalen_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.magdalen_start)
    mundine_well_future_enhancement = lipstack.mundine_well_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.mundine_well_start)
    naip_future_enhancement = lipstack.naip_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.naip_start)
    nw_australia_margin_future_enhancement = lipstack.nw_australia_margin_enhancement[:, :,key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.nw_australia_margin_start)
    panjal_future_enhancement = lipstack.panjal_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.panjal_start)
    parana_etendeka_future_enhancement = lipstack.parana_etendeka_enhancement[:, :,key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.parana_etendeka_start)
    qiangtang_future_enhancement = lipstack.qiangtang_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.qiangtang_start)
    seychelles_future_enhancement = lipstack.seychelles_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.seychelles_start)
    siberia_future_enhancement = lipstack.siberia_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.siberia_start)
    suordakh_future_enhancement = lipstack.suordakh_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.suordakh_start)
    swcuc_future_enhancement = lipstack.swcuc_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.swcuc_start)
    tarim_future_enhancement = lipstack.tarim_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.tarim_start)
    trap_future_enhancement = lipstack.trap_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.trap_start)
    vilyui_future_enhancement = lipstack.vilyui_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.vilyui_start)
    wichita_future_enhancement = lipstack.wichita_enhancement[:, :, key_future_index]*lipstack.sigmf(new_data_dict['t_geol'], 10, lipstack.wichita_start)

    # lip enhancement past
    lip_enhancement_past = np.sum((
        afar_past_enhancement, bunbury_past_enhancement, camp_past_enhancement,
        caribbean_colombian_past_enhancement, cimp_a_past_enhancement,
        cimp_b_past_enhancement, columbia_river_past_enhancement,
        comei_past_enhancement, deccan_past_enhancement,
        emeishan_past_enhancement, equamp_past_enhancement,
        ferrar_past_enhancement, franklin_past_enhancement,
        gunbarrel_past_enhancement, halip_past_enhancement,
        irkutsk_past_enhancement, kalkarindji_past_enhancement,
        karoo_past_enhancement, kola_dnieper_past_enhancement,
        madagascar_past_enhancement, magdalen_past_enhancement,
        mundine_well_past_enhancement, naip_past_enhancement,
        nw_australia_margin_past_enhancement, panjal_past_enhancement,
        parana_etendeka_past_enhancement, qiangtang_past_enhancement,
        seychelles_past_enhancement, siberia_past_enhancement,
        suordakh_past_enhancement, swcuc_past_enhancement,
        tarim_past_enhancement, trap_past_enhancement,
        vilyui_past_enhancement, wichita_past_enhancement),
        axis=0)
    
    # lip enhancement future
    lip_enhancement_future = np.sum((
        afar_future_enhancement, bunbury_future_enhancement, camp_future_enhancement,
        caribbean_colombian_future_enhancement, cimp_a_future_enhancement,
        cimp_b_future_enhancement, columbia_river_future_enhancement,
        comei_future_enhancement, deccan_future_enhancement,
        emeishan_future_enhancement, equamp_future_enhancement,
        ferrar_future_enhancement, franklin_future_enhancement,
        gunbarrel_future_enhancement, halip_future_enhancement,
        irkutsk_future_enhancement, kalkarindji_future_enhancement,
        karoo_future_enhancement, kola_dnieper_future_enhancement,
        madagascar_future_enhancement, magdalen_future_enhancement,
        mundine_well_future_enhancement, naip_future_enhancement,
        nw_australia_margin_future_enhancement, panjal_future_enhancement, 
        parana_etendeka_future_enhancement, qiangtang_future_enhancement,
        seychelles_future_enhancement, siberia_future_enhancement, 
        suordakh_future_enhancement, swcuc_future_enhancement,
        tarim_future_enhancement, trap_future_enhancement,
        vilyui_future_enhancement, wichita_future_enhancement),
          axis=0)

    #because somes lips overlap
    #if pars.lip_factor == 1:
    #    lip_enhancement_past = np.minimum(lip_enhancement_past, 1)
    #    lip_enhancement_future = np.minimum(lip_enhancement_future, 1)


    # Land
    land_past = np.copy(interpstack.land[:,:,key_past_index])
    land_future = np.copy(interpstack.land[:,:,key_future_index])

    # Gridbox area
    grid_area_km2 = np.copy(interpstack.gridarea)

    ######################################################################
    ##################   Spatial silicate weathering   ###################
    ######################################################################

    # West / Maffre weathering approximation

    # Runoff in mm/yr, have to np.copy otherwise original gets altered
    Q_past = np.copy(runoff_past)
    Q_past[Q_past<0] = 0
    Q_future = np.copy(runoff_future)
    Q_future[Q_future<0] = 0

    # Temp in kelvin
    T_past = Tair_past + 273
    T_future = Tair_future + 273

    # Maffre erosion calculation, t/m2/yr
    k_erosion = 3.3e-3 #### for 16Gt present day erosion in FOAM
    epsilon_past = k_erosion * (Q_past**0.31) * tslope_past * np.maximum(Tair_past,2)
    epsilon_future = k_erosion * (Q_future**0.31) * tslope_future * np.maximum(Tair_future,2)
    
    # check total tonnes of erosion - should be ~16Gt
    epsilon_per_gridbox_past = epsilon_past * grid_area_km2 * 1e6 #t/m2/yr*m2
    epsilon_per_gridbox_future = epsilon_future * grid_area_km2 * 1e6 #t/m2/yr*m2
    erosion_tot_past = np.sum(epsilon_per_gridbox_past)
    erosion_tot_future = np.sum(epsilon_per_gridbox_future)

    # Maffre weathering equation params
    Xm = erosion_pars.Xm
    K = erosion_pars.K
    kw = erosion_pars.kw
    Ea = erosion_pars.Ea
    z = erosion_pars.z
    sigplus1 = erosion_pars.sigplus1
    T0 = erosion_pars.T0
    R = erosion_pars.R

    #### equations
    r_t_past = np.exp((Ea/(R*T0)) - (Ea/(R*T_past)))
    r_t_future = np.exp((Ea/(R*T0)) - (Ea/(R*T_future)))
    r_q_past = 1 - np.exp(-1*kw*Q_past)
    r_q_future = 1 - np.exp(-1*kw*Q_future)
    r_reg_past = ((z/epsilon_past)**sigplus1 )/sigplus1
    r_reg_future = ((z/epsilon_future)**sigplus1 )/sigplus1

    # Equation for cw per km2 in each box
    # Consider this the 'base' weathering
    cw_per_km2_past_raw = 1e6*epsilon_past*Xm*(1 - np.exp(-1*K*r_q_past*r_t_past*r_reg_past))
    cw_per_km2_future_raw = 1e6*epsilon_future*Xm*(1 - np.exp(-1*K*r_q_future*r_t_future*r_reg_future))

    # save these out as useful for troubleshooting
    new_data_dict['lip_past_enhancement'] = lip_enhancement_past
    new_data_dict['lip_future_enhancement'] = lip_enhancement_future
    new_data_dict['cw_per_km2_past_raw'] = cw_per_km2_past_raw
    new_data_dict['cw_per_km2_future_raw'] = cw_per_km2_future_raw
    new_data_dict['erosion_tot'] = erosion_tot_past * \
        contribution_past + erosion_tot_future*contribution_future

    # calculate chemical weathering
    # the '1' refers to 'base' chemical weathering
    cw_per_km2_past = cw_per_km2_past_raw * \
        (1 + arc_enhancement_past + suture_enhancement_past +
         relict_arc_enhancement_past + lip_enhancement_past)
    
    cw_per_km2_future = cw_per_km2_future_raw * \
        (1 + arc_enhancement_future + suture_enhancement_future +
         relict_arc_enhancement_future + lip_enhancement_future)

    # Get present day weathering to calculate silw_scale
    co2ppm_present_day = 280

    if stepnumber.step == 1:
        
        pars.get_cw_present(co2ppm_present_day, interpstack, lipstack, k_erosion,
                             Xm, K, kw, Ea, z, sigplus1, T0, R,grid_area_km2)
        
    # cw total
    cw_past = cw_per_km2_past*grid_area_km2
    cw_future = cw_per_km2_future*grid_area_km2
    # world cw
    cw_past[np.isnan(cw_past)==1] = 0
    cw_future[np.isnan(cw_future)==1] = 0
    cw_sum_past = np.sum(cw_past)
    cw_sum_future = np.sum(cw_future)
    cw_tot = cw_sum_past*contribution_past + cw_sum_future*contribution_future

    # Carbonate weathering spatial approximation, linear with runoff
    if pars.pgeog_test == 1:
        k_carb_scale = 200
    else:
        k_carb_scale = 200 #scaling parameter to recover present day rate
    cwcarb_per_km2_past = k_carb_scale*Q_past
    cwcarb_per_km2_future = k_carb_scale*Q_future
    # cw total
    cwcarb_past = cwcarb_per_km2_past*grid_area_km2
    cwcarb_future = cwcarb_per_km2_future*grid_area_km2
    # world cwcarb
    cwcarb_past[np.isnan(cwcarb_past)==1] = 0
    cwcarb_future[np.isnan(cwcarb_future)==1] = 0
    cwcarb_sum_past = np.sum(cwcarb_past)
    cwcarb_sum_future = np.sum(cwcarb_future)

    ######################################################################
    ##################   Grid interpolated variables   ###################
    ######################################################################

    # Silicate weathering scale factor by present day rate in cation tonnes
    silw_scale = pars.cw_present #default run 4.2e8 at suturefactor = 1; preplant =1/4; for k erosion 3.3e-3.
    # Overall spatial weathering
    silw_spatial = cw_tot*((pars.k_basw + pars.k_granw)/silw_scale)
    carbw_spatial = (cwcarb_sum_past*contribution_past + cwcarb_sum_future*contribution_future)

    # Global average surface temperature
    gast = np.mean(Tair_past*model_pars.rel_contrib)*contribution_past \
                + np.mean(Tair_future*model_pars.rel_contrib)*contribution_future
    # Tropical surface temperature (24 S to 24 N gridcells)
    new_data_dict['sat_tropical'] = np.mean(Tair_past[14:26,:]*model_pars.rel_contrib[14:26,:]*0.67)*contribution_past \
        + np.mean(Tair_future[14:26,:]*model_pars.rel_contrib[14:26,:]*0.67)*contribution_future
    # equatorial surface temperature (equal contribution from 2S and 2N lat bands)
    new_data_dict['sat_equator'] = np.mean(Tair_past[19:21,:]*model_pars.rel_contrib[19:21,:])*contribution_past \
        + np.mean(Tair_future[19:21,:]*model_pars.rel_contrib[19:21,:])*contribution_future #set assumed ice temperature
    Tcrits = [-8, -10, -12]
    #Tcrits = [-10]
    new_data_dict['iceline'] = []

    # ice line calculations
    for Tcrit in Tcrits:    
        Tair_past_ice = np.copy(Tair_past)
        Tair_past_ice[Tair_past_ice >= Tcrit] = 0
        Tair_past_ice[Tair_past_ice < Tcrit] = 1
        Tair_future_ice = np.copy(Tair_future)
        Tair_future_ice[Tair_future_ice >= Tcrit] = 0
        Tair_future_ice[Tair_future_ice < Tcrit] = 1
        # Count only continental ice
        Tair_past_ice = Tair_past_ice*land_past
        Tair_future_ice = Tair_future_ice*land_future
        # Sum into lat bands, must be nansum
        latbands_past = np.nansum(Tair_past_ice, axis=1)
        latbands_future = np.nansum(Tair_future_ice, axis=1)
        latbands_past[latbands_past>0] = 1
        latbands_future[latbands_future>0] = 1
        # Find appropiate lat
        latresults_past =  interpstack.lat*latbands_past
        latresults_future =  interpstack.lat*latbands_future
        latresults_past[latresults_past == 0] = 90
        latresults_future[latresults_future == 0] = 90
        # Lowest glacial latitude
        iceline_past = np.min(np.abs(latresults_past))
        iceline_future = np.min(np.abs(latresults_future))
        tmp_iceline = iceline_past*contribution_past +  iceline_future*contribution_future
        
        new_data_dict['iceline'].append(tmp_iceline)

    ######################################################################
    ########################   Global variables   ########################
    ######################################################################

    # Effect of temp on VEG
    v_t = 1 - (((gast - 25)/25)**2)

    # Effect of CO2 on VEG
    P_atm = co2atm*1e6
    P_half = 183.6
    P_min = 10
    v_co2 = (P_atm - P_min)/(P_half + P_atm - P_min)

    # effect of O2 on VEG
    v_o2 = 1.5 - 0.5*(new_data_dict['O']/pars.O0)

    # full VEG limitation
    v_npp = 2*new_data_dict['evo']*v_t*v_o2*v_co2

    # COPSE reloaded fire feedback
    ignit = min(max(48*new_data_dict['mro2'] - 9.08, 0), 5 )
    firef = pars.kfire/(pars.kfire - 1 + ignit)

    # Mass of terrestrial biosphere
    new_data_dict['veg'] = v_npp * firef

    # basalt and granite temp dependency - direct and runoff
    Tsurf = gast + 273
    temp_gast = Tsurf
    new_data_dict['temp_gast'] = temp_gast
    new_data_dict['tempC'] = gast

    # COPSE reloaded fbiota
    V = new_data_dict['veg']
    f_biota = (1 - min(V*new_data_dict['W'], 1))*new_data_dict['preplant']*(new_data_dict['rco2']**0.5) + (V*new_data_dict['W'])

    # version using gran area and conserving total silw
    #basw+ granw should equal total weathering rate, irrispective of basw/granw fractions

    new_data_dict['basw'] = silw_spatial*(pars.basfrac*new_data_dict['bas_area'] \
                                          /(pars.basfrac*new_data_dict['bas_area'] \
                                            + (1 - pars.basfrac)*new_data_dict['gran_area']))
    new_data_dict['granw'] = silw_spatial*((1 - pars.basfrac)*new_data_dict['gran_area'] \
                                           /(pars.basfrac*new_data_dict['bas_area'] \
                                             + (1 - pars.basfrac)*new_data_dict['gran_area']))

    # Add fbiota
    new_data_dict['basw'] = new_data_dict['basw']*f_biota
    new_data_dict['granw'] = new_data_dict['granw']*f_biota
    new_data_dict['carbw'] = carbw_spatial*f_biota

    # Overall weathering
    new_data_dict['silw'] = new_data_dict['basw'] + new_data_dict['granw']
    carbw_relative = (new_data_dict['carbw']/pars.k_carbw)

    # Oxidative weathering
    new_data_dict['oxidw'] = pars.k_oxidw*carbw_relative*(new_data_dict['G']/pars.G0)*((new_data_dict['O']/pars.O0)**pars.a)

    # pyrite weathering
    new_data_dict['pyrw'] = pars.k_pyrw*carbw_relative*(new_data_dict['pyr']/pars.pyr0)

    # gypsum weathering
    new_data_dict['gypw'] = pars.k_gypw*(new_data_dict['gyp']/pars.gyp0)*carbw_relative

    # Seafloor weathering, revised following Brady and Gislason but not directly linking to CO2
    f_T_sfw = np.exp(0.0608*(Tsurf-288))
    new_data_dict['sfw'] = pars.k_sfw*f_T_sfw*new_data_dict['degass'] #assume spreading rate follows degassing here
    # Degassing
    new_data_dict['ocdeg'] = pars.k_ocdeg*new_data_dict['degass']*(new_data_dict['G']/pars.G0)
    new_data_dict['ccdeg'] = pars.k_ccdeg*new_data_dict['degass']*(new_data_dict['C']/pars.C0)*new_data_dict['Bforcing']
    new_data_dict['pyrdeg'] = pars.k_pyrdeg*(new_data_dict['pyr']/pars.pyr0)*new_data_dict['degass']
    new_data_dict['gypdeg'] = pars.k_gypdeg*(new_data_dict['gyp']/pars.gyp0)*new_data_dict['degass']

    # gypsum burial
    #mgsb = pars.k_mgsb*(S/pars.S0)
    new_data_dict['mgsb'] = pars.k_mgsb*(new_data_dict['S']/pars.S0)*(1/shoreline)

    # carbonate burial
    new_data_dict['mccb'] = new_data_dict['carbw'] + new_data_dict['silw']

    # COPSE reloaded Phospherous weathering
    pfrac_silw = 0.8
    pfrac_carbw = 0.14
    pfrac_oxidw = 0.06

    # Extra phospherous weathering from Longman et al. 2021 Nat Geo
    # 95th percentile + 95th percentile 2 Myr weathering, x 5 for recycling
    P_gice = 5*(6.49e15 + 2*1.23e15)
    P_hice = 5*(8.24e15 + 2*1.23e15)
    extra_P = P_gice*1e-6*norm.pdf(new_data_dict['t_geol'],-453.45,0.4)/norm.pdf(-453.45,-453.45,0.4) \
        + P_hice*1e-6*norm.pdf(new_data_dict['t_geol'],-444,0.4)/norm.pdf(-444,-444,0.4)
    new_data_dict['phosw'] = extra_P + pars.k_phosw*((pfrac_silw)*(new_data_dict['silw']/pars.k_silw) \
                                    + (pfrac_carbw)*(new_data_dict['carbw']/pars.k_carbw) \
                                        + (pfrac_oxidw)*(new_data_dict['oxidw']/pars.k_oxidw))

    # COPSE reloaded
    pland = pars.k_landfrac*new_data_dict['veg']*new_data_dict['phosw']
    pland0 = pars.k_landfrac*pars.k_phosw
    new_data_dict['psea'] = new_data_dict['phosw'] - pland

    # Convert total reservoir moles to micromoles/kg concentration
    Pconc = (new_data_dict['P']/pars.P0)*2.2
    Nconc = (new_data_dict['N']/pars.N0)*30.9
    newp = 117*min(Nconc/16,Pconc)
    new_data_dict['relativenewp'] = newp/pars.newp0

    # Carbon burial
    new_data_dict['mocb'] = pars.k_mocb*((newp/pars.newp0)**pars.b)*cb
    new_data_dict['locb'] = pars.k_locb*(pland/pland0)*new_data_dict['cpland']

    # pyr(ite) burial function (COPSE)
    fox = 1/(new_data_dict['O']/pars.O0)
    # mpsb scales with mocb so no extra uplift dependence
    new_data_dict['mpsb'] = pars.k_mpsb*(new_data_dict['S']/pars.S0)*fox*(new_data_dict['mocb']/pars.k_mocb)

    # OCEAN ANOXIC FRACTION
    k_anox = 12
    k_u = 0.5
    new_data_dict['anox'] = 1/(1 + np.exp(-1*k_anox*(k_u*(newp/pars.newp0) - (new_data_dict['O']/pars.O0))))

    # Nutrient burial
    CNsea = 37.5
    monb = new_data_dict['mocb']/CNsea

    # P burial with bioturbation on
    CPbiot = 250
    CPlam = 1000
    mopb = new_data_dict['mocb']*((f_biot/CPbiot) + ((1-f_biot)/CPlam))
    capb = pars.k_capb*(new_data_dict['mocb']/pars.k_mocb)

    # Reloaded(?)
    fepb = (pars.k_fepb/pars.k_oxfrac)*(1-new_data_dict['anox'])*(new_data_dict['P']/pars.P0)

    # nitrogen cycle, COPSE reloaded
    if (new_data_dict['N']/16) < new_data_dict['P']:
        new_data_dict['nfix'] = pars.k_nfix*(((new_data_dict['P'] - (new_data_dict['N']/16)) / (pars.P0 - (pars.N0/16)))**2)
    else:
        new_data_dict['nfix'] = 0

    new_data_dict['denit'] = pars.k_denit*(1 + (new_data_dict['anox']/(1-pars.k_oxfrac)))*(new_data_dict['N']/pars.N0)

    # Reductant input
    reductant_input = pars.k_reductant_input*new_data_dict['degass']

    ######################################################################
    ######################   Reservoir calculations  #####################
    ######################################################################

    # Phosphate
    dy[0] = new_data_dict['psea'] - mopb - capb - fepb

    # Oxygen
    dy[1] = new_data_dict['locb'] + new_data_dict['mocb'] - new_data_dict['oxidw']  - new_data_dict['ocdeg']  + 2*(new_data_dict['mpsb'] - new_data_dict['pyrw']  - new_data_dict['pyrdeg']) - reductant_input

    # Carbon dioxide
    dy[2] = -new_data_dict['locb'] - new_data_dict['mocb'] + new_data_dict['oxidw'] + new_data_dict['ocdeg'] + \
        new_data_dict['ccdeg'] + new_data_dict['carbw'] - new_data_dict['mccb'] - \
        new_data_dict['sfw'] + reductant_input + new_data_dict['lip_degass']

    # Sulphate
    dy[3] = new_data_dict['gypw'] + new_data_dict['pyrw'] - new_data_dict['mgsb'] - new_data_dict['mpsb'] + new_data_dict['gypdeg'] + new_data_dict['pyrdeg']

    # Buried organic C
    dy[4] = new_data_dict['locb'] + new_data_dict['mocb'] - new_data_dict['oxidw'] - new_data_dict['ocdeg']

    # Buried carb C
    dy[5] = new_data_dict['mccb'] + new_data_dict['sfw'] - new_data_dict['carbw'] - new_data_dict['ccdeg']

    # Buried pyrite S
    dy[6] = new_data_dict['mpsb'] - new_data_dict['pyrw'] - new_data_dict['pyrdeg']

    # Buried gypsum S
    dy[7] = new_data_dict['mgsb'] - new_data_dict['gypw'] - new_data_dict['gypdeg']

    # Nitrate
    dy[10] = new_data_dict['nfix'] - new_data_dict['denit'] - monb

    # Isotope reservoirs
    # d13c and d34s for forwards model
    new_data_dict['d13c_A'] = y[15]/y[2]
    new_data_dict['d34s_S'] = y[16]/y[3]

    # carbonate fractionation
    delta_locb = new_data_dict['d13c_A'] - capdelC_land
    delta_mocb = new_data_dict['d13c_A'] - capdelC_marine
    new_data_dict['delta_mccb'] = new_data_dict['d13c_A']
    # S isotopes (copse)
    new_data_dict['delta_mpsb'] = new_data_dict['d34s_S'] - capdelS

    # deltaORG_C*ORG_C
    dy[11] =  new_data_dict['locb']*(delta_locb) + new_data_dict['mocb']*(delta_mocb) - new_data_dict['oxidw']*new_data_dict['delta_G'] - new_data_dict['ocdeg']*new_data_dict['delta_G']

    # deltaCARB_C*CARB_C
    dy[12] =  new_data_dict['mccb']*new_data_dict['delta_mccb'] + new_data_dict['sfw']*new_data_dict['delta_mccb'] - new_data_dict['carbw']*new_data_dict['delta_C'] - new_data_dict['ccdeg']*new_data_dict['delta_C']

    # deltaPYR_S*PYR_S (young)
    dy[13] =  new_data_dict['mpsb']*new_data_dict['delta_mpsb'] - new_data_dict['pyrw']*new_data_dict['delta_pyr'] - new_data_dict['pyrdeg']*new_data_dict['delta_pyr']

    # deltaGYP_S*GYP_S (young)
    dy[14] =  new_data_dict['mgsb']*new_data_dict['d34s_S'] - new_data_dict['gypw']*new_data_dict['delta_gyp'] - new_data_dict['gypdeg']*new_data_dict['delta_gyp']

    # delta_A * A
    dy[15] = -new_data_dict['locb']*(delta_locb) -new_data_dict['mocb']*(delta_mocb) + new_data_dict['oxidw']*new_data_dict['delta_G'] \
        + new_data_dict['ocdeg']*new_data_dict['delta_G'] + new_data_dict['ccdeg']*new_data_dict['delta_C'] + new_data_dict['carbw']*new_data_dict['delta_C'] - new_data_dict['mccb']*new_data_dict['delta_mccb'] \
            - new_data_dict['sfw']*new_data_dict['delta_mccb'] + reductant_input*-5

    # delta_S * S
    dy[16] = new_data_dict['gypw']*new_data_dict['delta_gyp'] + new_data_dict['pyrw']*new_data_dict['delta_pyr'] -new_data_dict['mgsb']*new_data_dict['d34s_S'] \
        - new_data_dict['mpsb']*new_data_dict['delta_mpsb'] + new_data_dict['gypdeg']*new_data_dict['delta_gyp'] + new_data_dict['pyrdeg']*new_data_dict['delta_pyr'] \
        + new_data_dict['lip_degass']*lip_co2_d13c

    ######################################################################
    ########################   Strontium system   ########################
    ######################################################################

    # Fluxes
    new_data_dict['Sr_granw'] = pars.k_Sr_granw*(new_data_dict['granw']/pars.k_granw)
    new_data_dict['Sr_basw'] = pars.k_Sr_basw*(new_data_dict['basw']/pars.k_basw)
    new_data_dict['Sr_sedw'] = pars.k_Sr_sedw*(new_data_dict['carbw']/pars.k_carbw)*(new_data_dict['SSr']/pars.SSr0)
    new_data_dict['Sr_mantle'] = pars.k_Sr_mantle*new_data_dict['degass']
    Sr_sfw = pars.k_Sr_sfw*(new_data_dict['sfw']/pars.k_sfw)*(new_data_dict['OSr']/pars.OSr0)
    Sr_metam = pars.k_Sr_metam*new_data_dict['degass']*(new_data_dict['SSr']/pars.SSr0)
    Sr_sedb = pars.k_Sr_sedb*(new_data_dict['mccb']/pars.k_mccb)*(new_data_dict['OSr']/pars.OSr0)

    # fractionation calculations
    new_data_dict['delta_OSr'] = y[18]/y[17] ;
    delta_SSr = y[20]/y[19] ;

    # original frac
    RbSr_bas = 0.1
    RbSr_gran = 0.26
    RbSr_mantle = 0.066
    RbSr_carbonate = 0.5

    # frac calcs
    dSr0 = 0.69898
    tforwards = 4.5e9 + t
    lambda_val = 1.4e-11
    dSr_bas = dSr0 + RbSr_bas*(1 - np.exp(-1*lambda_val*tforwards))
    dSr_gran = dSr0 + RbSr_gran*(1 - np.exp(-1*lambda_val*tforwards))
    dSr_mantle = dSr0 + RbSr_mantle*(1 - np.exp(-1*lambda_val*tforwards))

    # Ocean [Sr]
    dy[17] = new_data_dict['Sr_granw'] + new_data_dict['Sr_basw'] + new_data_dict['Sr_sedw'] + new_data_dict['Sr_mantle'] - Sr_sedb - Sr_sfw

    # Ocean [Sr]*87/86Sr
    dy[18] = new_data_dict['Sr_granw']*dSr_gran + new_data_dict['Sr_basw']*dSr_bas + new_data_dict['Sr_sedw']*delta_SSr \
        + new_data_dict['Sr_mantle']*dSr_mantle - Sr_sedb*new_data_dict['delta_OSr'] - Sr_sfw*new_data_dict['delta_OSr']

    # Sediment [Sr]
    dy[19] = Sr_sedb - new_data_dict['Sr_sedw'] - Sr_metam

    # Sediment [Sr]*87/86Sr
    dy[20] = Sr_sedb*new_data_dict['delta_OSr'] - new_data_dict['Sr_sedw']*delta_SSr - Sr_metam*delta_SSr \
        + new_data_dict['SSr']*lambda_val*RbSr_carbonate*np.exp(lambda_val*tforwards)

    ######################################################################
    #####################   Mass conservation check   ####################
    ######################################################################

    new_data_dict['res_C'] = new_data_dict['A'] + new_data_dict['G'] + new_data_dict['C']
    new_data_dict['res_S'] = new_data_dict['S'] + new_data_dict['pyr'] + new_data_dict['gyp']
    new_data_dict['iso_res_C'] = new_data_dict['A']*new_data_dict['d13c_A'] + new_data_dict['G']*new_data_dict['delta_G'] + new_data_dict['C']*new_data_dict['delta_C']
    new_data_dict['iso_res_S'] = new_data_dict['S']*new_data_dict['d34s_S'] + new_data_dict['pyr']*new_data_dict['delta_pyr'] + new_data_dict['gyp']*new_data_dict['delta_gyp']

    ######################################################################
    ################   Record states for single run   ################
    ######################################################################

    if sensanal.key == 0:
        workingstate.add_workingstates(new_data_dict)
        ### print a gridstate when each keytime threshold is crossed, or at model end
        next_stamp = model_pars.next_gridstamp
        if model_pars.finishgrid == 0:
            if new_data_dict['t_geol'] > next_stamp or new_data_dict['t_geol'] == 0:
                #### write gridstates
                gridstate.time_myr[:,:,model_pars.gridstamp_number] = np.copy(next_stamp)
                gridstate.land[:,:,model_pars.gridstamp_number] = np.copy(land_past)
                gridstate.base_chem_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw)
                gridstate.suture[:,:,model_pars.gridstamp_number] = np.copy(suture_past)
                gridstate.suture_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*suture_enhancement_past)
                gridstate.arc[:,:,model_pars.gridstamp_number] = np.copy(arc_past)
                gridstate.arc_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*arc_enhancement_past)
                gridstate.relict_arc[:,:,model_pars.gridstamp_number] = np.copy(relict_past)
                gridstate.relict_arc_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*relict_arc_enhancement_past)
                gridstate.lip[:, :, model_pars.gridstamp_number] = np.copy(lip_past)
                gridstate.lip_weathering[:, :, model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*lip_enhancement_past)
                gridstate.Q[:,:,model_pars.gridstamp_number] = np.copy(Q_past)
                gridstate.Tair[:,:,model_pars.gridstamp_number] = np.copy(Tair_past)
                gridstate.topo[:,:,model_pars.gridstamp_number] = np.copy(topo_past)
                gridstate.cw[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past) #t/km2/yr
                gridstate.cwcarb[:,:,model_pars.gridstamp_number] = np.copy(cwcarb_past)
                gridstate.epsilon[:,:,model_pars.gridstamp_number] = np.copy(epsilon_past)*1e6 #t/km2/yr
                gridstate.preplant[:,:,model_pars.gridstamp_number] = np.copy(new_data_dict['preplant'])

                #### set next boundary
                if new_data_dict['t_geol'] < 0:
                    model_pars.gridstamp_number = model_pars.gridstamp_number + 1
                    model_pars.next_gridstamp = model_pars.runstamps[model_pars.gridstamp_number]
                else:
                    model_pars.finishgrid = 1

    ######################################################################
    #######   Record plotting states only in sensanal and tuning  ########
    ######################################################################

    if sensanal.key == 1:

        workingstate.bas_area.append(new_data_dict['bas_area'])
        workingstate.gran_area.append(new_data_dict['gran_area'])
        workingstate.degass.append(new_data_dict['degass'])
        workingstate.lip_degass.append(new_data_dict['lip_degass'])
        workingstate.delta_mccb.append(new_data_dict['delta_mccb'])
        workingstate.d34s_S.append(new_data_dict['d34s_S'])
        workingstate.delta_OSr.append(new_data_dict['delta_OSr'])
        workingstate.SmM.append(28*new_data_dict['S']/pars.S0)
        workingstate.co2ppm.append(new_data_dict['rco2']*280)
        workingstate.mro2.append(new_data_dict['mro2'])
        workingstate.iceline.append(new_data_dict['iceline'])
        workingstate.T_gast.append(temp_gast - 273)
        workingstate.sat_tropical.append(new_data_dict['sat_tropical'])
        workingstate.sat_equator.append(new_data_dict['sat_equator'])
        workingstate.anox.append(new_data_dict['anox'])
        workingstate.P.append(new_data_dict['P']/pars.P0)
        workingstate.N.append(new_data_dict['N']/pars.N0)
        workingstate.time_myr.append(new_data_dict['t_geol'])
        workingstate.time.append(t)
        workingstate.preplant.append(new_data_dict['preplant'])

        #tuning states
        if tuning.key == 1:

            workingstate.Orel.append(new_data_dict['O']/pars.O0)
            workingstate.Srel.append(new_data_dict['S']/pars.S0)
            workingstate.Arel.append(new_data_dict['A']/pars.A0)
            workingstate.Crel.append(new_data_dict['C']/pars.C0)
            workingstate.Grel.append(new_data_dict['G']/pars.G0)
            workingstate.pyrrel.append(new_data_dict['pyr']/pars.pyr0)
            workingstate.gyprel.append(new_data_dict['gyp']/pars.gyp0)
                    

        next_stamp = model_pars.next_gridstamp
        if model_pars.finishgrid == 0:
            if new_data_dict['t_geol'] > next_stamp or new_data_dict['t_geol'] == 0:
                #### write gridstates
                gridstate.time_myr[:,:,model_pars.gridstamp_number] = np.copy(next_stamp)
                gridstate.land[:,:,model_pars.gridstamp_number] = np.copy(land_past)
                gridstate.base_chem_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw)
                gridstate.suture[:,:,model_pars.gridstamp_number] = np.copy(suture_past)
                gridstate.suture_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*suture_enhancement_past)
                gridstate.arc[:,:,model_pars.gridstamp_number] = np.copy(arc_past)
                gridstate.arc_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*arc_enhancement_past)
                gridstate.relict_arc[:,:,model_pars.gridstamp_number] = np.copy(relict_past)
                gridstate.relict_arc_weathering[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*relict_arc_enhancement_past)
                gridstate.lip[:, :, model_pars.gridstamp_number] = np.copy(lip_past)
                gridstate.lip_weathering[:, :, model_pars.gridstamp_number] = np.copy(cw_per_km2_past_raw*lip_enhancement_past)
                gridstate.Q[:, :, model_pars.gridstamp_number] = np.copy(Q_past)
                gridstate.Tair[:,:,model_pars.gridstamp_number] = np.copy(Tair_past)
                gridstate.topo[:,:,model_pars.gridstamp_number] = np.copy(topo_past)
                gridstate.cw[:,:,model_pars.gridstamp_number] = np.copy(cw_per_km2_past) #t/km2/yr
                gridstate.cwcarb[:,:,model_pars.gridstamp_number] = np.copy(cwcarb_past)
                gridstate.epsilon[:,:,model_pars.gridstamp_number] = np.copy(epsilon_past)*1e6 #t/km2/yr
                gridstate.preplant[:,:,model_pars.gridstamp_number] = np.copy(new_data_dict['preplant'])

                #### set next boundary
                if new_data_dict['t_geol'] < 0:
                    model_pars.gridstamp_number = model_pars.gridstamp_number + 1
                    model_pars.next_gridstamp = model_pars.runstamps[model_pars.gridstamp_number]
                else:
                    model_pars.finishgrid = 1

    ######################################################################
    #########################   Final actions   ##########################
    ######################################################################

    ##### output timestep if specified
    if sensanal.key == 0:
        if pars.telltime == 1:
            if np.mod(stepnumber.step, model_pars.display_resolution) == 0:
                ### print model state to screen
                print('Model step: %d \t time: %s \t next keyframe: %d \n' % (stepnumber.step, new_data_dict['t_geol'], next_stamp))
    #### record current model step
    stepnumber.step = stepnumber.step + 1


    #### option to bail out if model is running aground
    if stepnumber.step > model_pars.bailnumber:
        return dy

    return dy
