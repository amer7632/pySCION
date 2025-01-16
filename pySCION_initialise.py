import numpy as np
import time
import pySCION_classes
import pySCION_equations
import pySCION_plot_fluxes
import pySCION_plot_worldgraphic
import pandas as pd
import scipy.io as sp
from pathlib import Path
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

def intersect_mtlb(a, b):
    # Same functionality as intersect in matlab taken from:
    # https://stackoverflow.com/questions/45637778/how-to-find-intersect-indexes-and-values-in-python
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

def pySCION_initialise(runcontrol, tuning_start_vals= np.asarray([])):
    start_time = time.time()

    ######################################################################
    #################   Check for sensitivity analysis   #################
    ######################################################################
    if runcontrol == 1:
        # For sensitivity analysis (slightly lighter run with less printed outputs)
        sensanal = pySCION_classes.Sensanal_class(sensanal_key = 1)
        tuning = pySCION_classes.Tuning_class(tuning_key = 0)
        plotrun = pySCION_classes.Plotrun_class(plotrun_key = 0)
        telltime = 0
    
    elif runcontrol == 0:
        # for tuning, must be run from call_pySCION_tune.ipynb
        print('Tuning...')
        print('\n')
        sensanal = pySCION_classes.Sensanal_class(sensanal_key = 1)
        tuning = pySCION_classes.Tuning_class(tuning_key = 1)
        plotrun = pySCION_classes.Plotrun_class(plotrun_key = 0)
        telltime = 0
    elif runcontrol == -1:
        # For other runs
        sensanal = pySCION_classes.Sensanal_class(sensanal_key = 0)
        tuning = pySCION_classes.Tuning_class(tuning_key = 0)
        plotrun = pySCION_classes.Plotrun_class(plotrun_key = 1)
        telltime = 1
    elif runcontrol == -2:
        # For other runs
        print('Worldgraphic plots, takes awhile at the end')
        sensanal = pySCION_classes.Sensanal_class(sensanal_key = 0)
        tuning = pySCION_classes.Tuning_class(tuning_key = 0)
        plotrun = pySCION_classes.Plotrun_class(plotrun_key = 1)
        telltime = 1

    else:
        print('Runcontrol not set properly:'
                    '\n'
                    '-2, one run and world plots'
                    '\n'
                    '-1, one run and flux plots (default)'
                    '\n'
                    '0 for tuning (use tuning notebook)'
                    '\n'
                    '1 sensitivity analysis, no plots')

        return

    # Set flags for testing drivers
    # 0 = default, driver is on
    # 1 = driver is equal to present-day (i.e. 1)

    # different tests
    arc_test = 0
    suture_test = 0
    palaeogeog_test = 0
    degassing_test = 0
    bio_test = 0
    lip_weathering_test = 0
    lip_degass_test = 0
    
    # to do:
    # integrate this more systematically to make it less dangerous
    if tuning.key == 0:
        # ignore if tuning
        if arc_test == 1:
            print('Arc weathering is turned OFF!')
        else:
            print('Arc weathering is turned ON!')
        if suture_test == 1:
            print('Suture weathering is turned OFF')
        else:
            print('Suture weathering is turned ON!')
        if degassing_test == 1:
            print('Degassing is turned OFF!')
        else:
            print('Degassing is turned ON!')
        if palaeogeog_test == 1:
            print('Palaeogeography is turned OFF!')
        else:
            print('Palaeogeography is turned ON!')
        if bio_test == 1:
            print('Plant forcings are turned OFF!')
        else:
            print('Plant forcings are turned ON!')
        if lip_weathering_test == 1:
            print('LIP weathering is turned OFF!')
        else:
            print('LIP weathering is turned ON!')
        if lip_degass_test == 1:
            print('LIP degassing is turned OFF!')
        else:
            print('LIP degassing is turned ON!')            
    # Set parameters
    if sensanal.key == 0:
        print('setting parameters... %s' % (time.time() - start_time))


    ####################################################################
    ####################   Flux values at present   ####################
    ####################################################################

    # variable dictionary
    var_dict = {}
    var_dict['telltime'] = telltime
    # tests
    var_dict['palaeogeog_test'] = palaeogeog_test
    var_dict['suture_test'] = suture_test
    var_dict['arc_test'] = arc_test
    var_dict['degassing_test'] = degassing_test
    var_dict['biology_test'] = bio_test
    var_dict['lip_weathering_test'] = lip_weathering_test
    var_dict['lip_degass_test'] = lip_degass_test

    # reductant input
    var_dict['k_reductant_input'] = 0.4e12 ### schopf and klein 1992

    # org C cycle
    var_dict['k_locb'] = 2.5e12
    var_dict['k_mocb'] = 2.5e12
    var_dict['k_ocdeg'] = 1.25e12

    # carb C cycle
    var_dict['k_ccdeg'] = 12e12
    var_dict['k_carbw'] = 8e12
    var_dict['k_sfw'] = 1.75e12
    var_dict['basfrac'] = 0.3

    # S cycle
    var_dict['k_mpsb'] = 0.7e12
    var_dict['k_mgsb'] = 1e12
    var_dict['k_pyrw'] = 7e11
    var_dict['k_gypw'] = 1e12
    var_dict['k_pyrdeg'] = 0
    var_dict['k_gypdeg'] = 0

    # P cycle
    var_dict['k_capb'] = 2e10
    var_dict['k_fepb'] = 1e10
    var_dict['k_mopb'] = 1e10
    var_dict['k_phosw'] = 4.25e10
    var_dict['k_landfrac'] = 0.0588
    
    # N cycle
    var_dict['k_nfix'] = 8.67e12
    var_dict['k_denit'] = 4.3e12

    # Sr cycle
    var_dict['k_Sr_sedw'] = 17e9
    var_dict['k_Sr_mantle'] = 7.3e9
    var_dict['k_Sr_silw'] = 13e9
    var_dict['k_Sr_metam'] = 13e9

    # other
    var_dict['k_oxfrac'] = 0.9975
    var_dict['Pconc0'] = 2.2
    var_dict['Nconc0'] = 30.9
    var_dict['newp0'] = 117 * min(var_dict['Nconc0']/16,
                                  var_dict['Pconc0'])
    
    # COPSE constant for calculating pO2 from normalised O2
    var_dict['copsek16'] = 3.762
    # Oxidative weathering dependency on O2 concentration
    var_dict['a'] = 0.5
    # Marine organic carbon burial dependency on new production
    var_dict['b'] = 2
    # Fire feedback
    var_dict['kfire'] = 3

    # Reservoir present day sizes (mol)
    var_dict['P0'] = 3.1*10**15
    var_dict['O0'] = 3.7*10**19
    var_dict['A0'] = 3.193*10**18
    var_dict['G0'] = 1.25*10**21#1.25*10**21
    var_dict['C0'] = 5*10**21  # 5*10**21
    var_dict['pyr0'] = 1.8*10**20
    var_dict['gyp0'] = 2*10**20
    var_dict['S0'] = 4*10**19
    var_dict['cal0'] = 1.397e19
    var_dict['N0'] = 4.35e16
    var_dict['OSr0'] = 1.2e17 #francois and walker 1992
    var_dict['SSr0'] = 5e18

    # Arc and suture weathering enhancement factor
    var_dict['arc_factor'] = 7
    var_dict['suture_factor'] = 20
    var_dict['relict_arc_factor'] = 7
    var_dict['lip_factor'] = 7
    
    # If setting them to constant
    if arc_test == 1:
        var_dict['arc_factor'] = 1
        var_dict['relict_arc_factor'] = 1
    if suture_test == 1:
        var_dict['suture_factor'] = 1
    if lip_weathering_test == 1:
        var_dict['lip_factor'] = 1

    # Finished loading params
    if sensanal.key == 0:
        print('Done')
        endtime = time.time() - start_time
        print('time:', endtime)

    #####################################################################
    #################   Load Forcings   #################################
    #####################################################################

    # Starting to load forcings
    if sensanal.key == 0:
        print('loading forcings... %s' % (time.time() - start_time))

    # Load INTERPSTACK and put into dict
    file_to_open_INTERPSTACK ='./forcings/INTERPSTACK_sep2021_v5.mat'
    mat_contents = sp.loadmat(file_to_open_INTERPSTACK)
    interpstack_dict = {}
    interpstack_dict['co2'] = mat_contents['INTERPSTACK'][0][0][0][0]
    interpstack_dict['interp_time'] = mat_contents['INTERPSTACK'][0][0][1][0]
    interpstack_dict['Tair'] = mat_contents['INTERPSTACK'][0][0][2]
    interpstack_dict['runoff'] = mat_contents['INTERPSTACK'][0][0][3]
    interpstack_dict['land'] = mat_contents['INTERPSTACK'][0][0][4]
    interpstack_dict['lat'] = mat_contents['INTERPSTACK'][0][0][5][0]
    interpstack_dict['lon'] = mat_contents['INTERPSTACK'][0][0][6][0]
    interpstack_dict['topo'] = mat_contents['INTERPSTACK'][0][0][7]
    interpstack_dict['aire'] = mat_contents['INTERPSTACK'][0][0][8]
    interpstack_dict['gridarea'] = mat_contents['INTERPSTACK'][0][0][9]
    interpstack_dict['suture'] = mat_contents['INTERPSTACK'][0][0][10]
    interpstack_dict['arc'] = mat_contents['INTERPSTACK'][0][0][14] #raw plate model arcs
    interpstack_dict['slope'] = mat_contents['INTERPSTACK'][0][0][16]
    interpstack_dict['relict_arc'] = mat_contents['INTERPSTACK'][0][0][17] #plate model relict arcs
    
    # Maffre and West params, pySCION default are greyed out
    erosion_dict = {}
    erosion_dict['Xm'] = 0.1
    erosion_dict['K'] =  1e-5 #6e-5 | 3.8e-5
    erosion_dict['kw'] = 1e-3 # 1e-3 |6.3e-1
    erosion_dict['Ea'] = 20
    erosion_dict['z'] = 10
    erosion_dict['sigplus1'] = 0.89 #0.9
    erosion_dict['T0'] = 286
    erosion_dict['R'] = 8.31e-3

    erosion_pars = pySCION_classes.Erosion_parameters_class(erosion_dict)

    # Load COPSE reloaded forcing set
    file_to_open_FORCINGS =  './forcings/COPSE_forcings_June2022_v2.mat'
    forcings_contents = sp.loadmat(file_to_open_FORCINGS)

    forcings_dict = {}
    forcings_dict['t'] = forcings_contents['forcings'][0][0][0][0]
    forcings_dict['B'] = forcings_contents['forcings'][0][0][1][2]
    #forcings_dict['BA'] = forcings_contents['forcings'][0][0][2][0]
    forcings_dict['Ca'] = forcings_contents['forcings'][0][0][3][0]
    forcings_dict['CP'] = forcings_contents['forcings'][0][0][4][0]
    forcings_dict['D'] = forcings_contents['forcings'][0][0][5][0]
    forcings_dict['E'] = forcings_contents['forcings'][0][0][6][0]
    #forcings_dict['GA'] = forcings_contents['forcings'][0][0][7][0]
    forcings_dict['PG'] = forcings_contents['forcings'][0][0][8][0]
    forcings_dict['U'] = forcings_contents['forcings'][0][0][9][0]
    forcings_dict['W'] = forcings_contents['forcings'][0][0][10][0]
    forcings_dict['coal'] = forcings_contents['forcings'][0][0][11][0]
    forcings_dict['epsilon'] = forcings_contents['forcings'][0][0][12][0]

    if bio_test == 1:
        forcings_dict['W'] = np.ones_like(forcings_contents['forcings'][0][0][10][0])
        forcings_dict['E'] = np.ones_like(forcings_contents['forcings'][0][0][6][0])

    # Granite/basalt fraction of silicate weathering
    file_to_open_Ba = './forcings/BA.xlsx'
    forcings_dict['Ba_df'] = pd.read_excel(file_to_open_Ba)
    # revised GA?
    file_to_open_Ga = './forcings/GA_revised.xlsx'
    forcings_dict['Ga_df'] = pd.read_excel(file_to_open_Ga)

    # Degassing rate
    file_to_open_DEGASSING = './forcings/combined_D_force_revised_Apr2024.mat'
    forcings_dict['degassing'] = sp.loadmat(file_to_open_DEGASSING)
    # Shoreline forcing
    file_to_open_SHORELINE = './forcings/shoreline.mat'
    forcings_dict['shoreline'] = sp.loadmat(file_to_open_SHORELINE)

    # add LIPs? (currently from Ben's work, will update to Jack's)
    file_to_open_LIP_DEGASSING = './forcings/CR_force_LIP_table_simple.csv'
    forcings_dict['lip_degassing'] = pd.read_csv(file_to_open_LIP_DEGASSING)

    # Make forcings class here
    forcings = pySCION_classes.Forcings_class(forcings_dict)
    # get interpolations
    forcings.get_interp_forcings()

    # Finished loading forcings
    if sensanal.key == 0:
        print('Done')
        endtime = time.time() - start_time
        print('time:', endtime)
        #dont need sensparams, so we can pass an empty array
        sensparams = np.asarray([])

    #####################################################################
    #################   Load LIPs   ##################
    #####################################################################

    file_to_open_LIPSTACK = './forcings/LIPSTACK.mat'
    lipstack_dict = sp.loadmat(file_to_open_LIPSTACK)


    file_to_open_lips_start_time = './data/lips_start_time.csv'
    lips_start_time = pd.read_csv(file_to_open_lips_start_time)

    del_keys = ['__header__',
                '__version__',
                '__globals__']
    for key in del_keys:
        del lipstack_dict[key]

    lipstack_dict['afar_start'] = lips_start_time['afar'].values[0]
    lipstack_dict['bunbury_start'] = lips_start_time['bunbury'].values[0]
    lipstack_dict['camp_start'] = lips_start_time['camp'].values[0]
    lipstack_dict['caribbean_colombian_start'] = lips_start_time['caribbean_colombian'].values[0]
    lipstack_dict['cimp_a_start'] = lips_start_time['cimp_a'].values[0]
    lipstack_dict['cimp_b_start'] = lips_start_time['cimp_b'].values[0]
    lipstack_dict['columbia_river_start'] = lips_start_time['columbia_river'].values[0]
    lipstack_dict['comei_start'] = lips_start_time['comei'].values[0]
    lipstack_dict['deccan_start'] = lips_start_time['deccan'].values[0]
    lipstack_dict['emeishan_start'] = lips_start_time['emeishan'].values[0]
    lipstack_dict['equamp_start'] = lips_start_time['equamp'].values[0]
    lipstack_dict['ferrar_start'] = lips_start_time['ferrar'].values[0]
    lipstack_dict['franklin_start'] = lips_start_time['franklin'].values[0]
    lipstack_dict['gunbarrel_start'] = lips_start_time['gunbarrel'].values[0]
    lipstack_dict['halip_start'] = lips_start_time['halip'].values[0]
    lipstack_dict['irkutsk_start'] = lips_start_time['irkutsk'].values[0]
    lipstack_dict['kalkarindji_start'] = lips_start_time['kalkarindji'].values[0]
    lipstack_dict['karoo_start'] = lips_start_time['karoo'].values[0]
    lipstack_dict['kola_dnieper_start'] = lips_start_time['kola_dnieper'].values[0]
    lipstack_dict['madagascar_start'] = lips_start_time['madagascar'].values[0]
    lipstack_dict['magdalen_start'] = lips_start_time['magdalen'].values[0]
    lipstack_dict['mundine_well_start'] = lips_start_time['mundine_well'].values[0]
    lipstack_dict['naip_start'] = lips_start_time['naip'].values[0]
    lipstack_dict['nw_australia_margin_start'] = lips_start_time['nw_australia_margin'].values[0]
    lipstack_dict['panjal_start'] = lips_start_time['panjal'].values[0]
    lipstack_dict['parana_etendeka_start'] = lips_start_time['parana_etendeka'].values[0]
    lipstack_dict['qiangtang_start'] = lips_start_time['qiangtang'].values[0]
    lipstack_dict['seychelles_start'] = lips_start_time['seychelles'].values[0]
    lipstack_dict['siberia_start'] = lips_start_time['siberia'].values[0]
    lipstack_dict['suordakh_start'] = lips_start_time['suordakh'].values[0]
    lipstack_dict['swcuc_start'] = lips_start_time['swcuc'].values[0]
    lipstack_dict['tarim_start'] = lips_start_time['tarim'].values[0]
    lipstack_dict['trap_start'] = lips_start_time['trap'].values[0]
    lipstack_dict['vilyui_start'] = lips_start_time['vilyui'].values[0]
    lipstack_dict['wichita_start'] = lips_start_time['wichita'].values[0]
    
    # Make LIP class here
    lipstack = pySCION_classes.Lipstack_class(lipstack_dict)

    #####################################################################
    #################   Generate sensitivity randoms   ##################
    #####################################################################

    if sensanal.key == 1:
        # Set the random seed so we get pseudo random each iteration and process
        np.random.seed()
        # Generate random number in [-1 +1]
        sensparams = pySCION_classes.Sensparams_class(
            randminusplus1 = 2*(0.5 - np.random.rand(1)),
            randminusplus2 = 2*(0.5 - np.random.rand(1)),
            randminusplus3 = 2*(0.5 - np.random.rand(1)),
            randminusplus4 = 2*(0.5 - np.random.rand(1)),
            randminusplus5 = 2*(0.5 - np.random.rand(1)),
            randminusplus6 = 2*(0.5 - np.random.rand(1)),
            randminusplus7 = 2*(0.5 - np.random.rand(1)),
            randminusplus8 = 2*(0.5 - np.random.rand(1))
            )

    #####################################################################
    #######################   Initialise solver   #######################
    #####################################################################

    # Run beginning
    if sensanal.key == 0:
        print('Beginning run: \n')

    # if no plot or sensitivity command set to single run
    if not sensanal.key:
        sensanal.key = 0
    if sensanal.key != 1:
        if not plotrun.key:
            plotrun.key = 1

    # Model parameters
    model_pars_dict = {}
    # Model timeframe in negative years (0 = present day)
    model_pars_dict['whenstart'] = -600e6
    model_pars_dict['whenend'] = -0

    # Set up grid stamp times
    model_pars_dict['gridstamp_number'] = 0
    model_pars_dict['finishgrid'] = 0

    # Set number of model steps to take before bailing out
    model_pars_dict['bailnumber'] = 1e5

    # Display every n model steps whilst running
    model_pars_dict['display_resolution'] = 200
    # Set output length to be 0 for now
    model_pars_dict['output_length'] = 0
    
    # Define interpstack class here
    interpstack = pySCION_classes.Interpstack_class(interpstack_dict)

    # Define model parameters class here
    model_pars = pySCION_classes.Model_parameters_class(
                    model_pars_dict, interpstack.time
                    )
    # Relative contribution from latitude bands
    model_pars.get_rel_contrib(interpstack.lat, interpstack.lon)

    # To test palaeogeography
    if palaeogeog_test == 1:

    #GAST = np.mean(Tair_past * model_pars.rel_contrib)*contribution_past  +  np.mean(Tair_future * model_pars.rel_contrib)*contribution_future
        #for time_step in np.arange(np.shape(Tair)[3]):
        #    for CO2_step in np.arange(np.shape(Tair)[2]):
        #        #get mean tair value (note, Tair is a full coverage grid, so don't need to use land to filter 1/0s)
        #        tmp_tair_mean = np.mean(Tair[:,:,CO2_step,time_step] * model_pars.rel_contrib)
        #        #get land mask
        #        tmp_land_tair = np.ones_like(land[:,:,time_step]) * tmp_tair_mean
        #        INTERPSTACK.Tair[:,:,CO2_step,time_step] = tmp_land_tair

        for time_step in np.arange(np.shape(interpstack_dict['runoff'])[3]):
            for co2_step in np.arange(np.shape(interpstack_dict['runoff'])[2]):
                # Get mean runoff value
                tmp_runoff_mean = interpstack_dict['runoff'][:,:,co2_step,time_step][np.nonzero(
                    interpstack_dict['runoff'][:,:,co2_step,time_step])].mean()
                # Here we don't have 0 values in our final calc, 
                # see CO2_vs_CWeathering_tot.ipynb
                if tmp_runoff_mean < 4 :
                    tmp_runoff_mean = 4
                # Get land mask
                tmp_land_runoff = np.copy(interpstack_dict['land'][:,:,time_step])*tmp_runoff_mean
                interpstack.runoff[:,:,co2_step,time_step] = tmp_land_runoff

        for time_step in np.arange(np.shape(interpstack_dict['slope'])[2]):

            tmp_slope_mean = interpstack_dict['slope'][:,:,time_step][np.nonzero(
                    interpstack_dict['slope'][:,:,time_step])].mean()
            tmp_land_slope = np.copy(interpstack_dict['land'][:,:,time_step])*tmp_slope_mean
            interpstack.slope[:,:,time_step] = tmp_land_slope

    # Define variable pars class here
    pars = pySCION_classes.Variable_parameters_class(var_dict)
    
    # Get weathering enhancements
    interpstack.get_enhancements(pars)
    lipstack.get_lip_enhancements(pars)
    
    # Define stepnumber class
    step = 1
    stepnumber = pySCION_classes.Stepnumber_class(step)

    # Set starting reservoir sizes
    start_pars_dict = {}
    start_pars_dict['pstart'] = pars.P0
    start_pars_dict['tempstart'] = 288
    start_pars_dict['cal_start'] = pars.cal0
    start_pars_dict['N_start'] = pars.N0
    start_pars_dict['OSr_start'] = pars.OSr0
    start_pars_dict['SSr_start'] = pars.SSr0
    start_pars_dict['delta_A_start'] = 0
    start_pars_dict['delta_S_start'] = 35
    start_pars_dict['delta_G_start'] = -27
    start_pars_dict['delta_C_start'] = -2
    start_pars_dict['delta_pyr_start'] = -5
    start_pars_dict['delta_gyp_start'] = 20
    start_pars_dict['delta_OSr_start'] = 0.708
    start_pars_dict['delta_SSr_start'] = 0.708

    #####################################################################
    ################   Initial parameter tuning option  #################
    #####################################################################

    if tuning.key == 0:
        # use pre tuned
        #tuned_vals = [0.1, 3, 0.05, 0.55, 0.95, 1.2, 1]
        tuned_vals = [0.10174495, 2.98351764, 0.05135419, 0.62902351, 0.98318291, 1.24076051, 0.77681714]
    else:
        # run call pySCION_tune and hope for the best
        if not tuning_start_vals.size:
            print('No tuning values provided, ending run. Use Runcontrol = 1 or Runcontrol = -1 for non tuning runs, or use call_pySCION_tune.ipynb')
            return None
        tuned_vals = tuning_start_vals
    
    start_pars_dict['ostart'] = pars.O0*tuned_vals[0]
    start_pars_dict['astart'] = pars.A0*tuned_vals[1]
    start_pars_dict['sstart'] = pars.S0*tuned_vals[2]
    start_pars_dict['gstart'] = pars.G0*tuned_vals[3]
    start_pars_dict['cstart'] = pars.C0*tuned_vals[4]
    start_pars_dict['pyrstart'] = pars.pyr0*tuned_vals[5]
    start_pars_dict['gypstart'] = pars.gyp0*tuned_vals[6]

    # Define starting parameters here ###
    start_pars = pySCION_classes.Starting_parameters_class(start_pars_dict)
    # note model start time
    model_time = time.time()
    # Make classes for results storage
    if sensanal.key == 0:
        workingstate = pySCION_classes.Workingstate_class()
        gridstate_array = np.zeros([40,48,22], dtype=complex)
        gridstate = pySCION_classes.Gridstate_class(gridstate_array)

        # Run the system
        rawoutput = solve_ivp(pySCION_equations.pySCION_equations, [model_pars.whenstart,
                                                                model_pars.whenend],
                                  start_pars.startstate, method='BDF',
                                  max_step=1e6, args=[pars, forcings, sensanal, tuning,
                                                      interpstack, lipstack, model_pars, workingstate,
                                                      stepnumber, gridstate, sensparams,
                                                      erosion_pars])
    else:
        workingstate = pySCION_classes.Workingstate_class_sensanal()
        gridstate_array = np.zeros([40,48,22], dtype=complex)
        gridstate = pySCION_classes.Gridstate_class(gridstate_array)
        # Run the system
        rawoutput = solve_ivp(pySCION_equations.pySCION_equations, [model_pars.whenstart,
                                                                model_pars.whenend],
                                  start_pars.startstate, method='BDF',
                                  max_step=1e6, args=[pars, forcings, sensanal, tuning,
                                                      interpstack, lipstack, model_pars, workingstate,
                                                      stepnumber, gridstate, sensparams,
                                                      erosion_pars])

    #####################################################################
    #################   Postprocessing   ################################
    #####################################################################
    # Size of output
    model_pars.output_length = len(rawoutput.t)

    if sensanal.key == 0:
        # Model finished output to screen
        print('Integration finished \t')
        print('Total steps: %d \t' % stepnumber.step)
        print('Output steps: %d \n' % model_pars.output_length)

    # Print final model states using final state for each timepoint during integration
    if sensanal.key == 0:
        print('assembling   vectors... \t')

    # Trecords is index of shared values between ode15s output T vector and
    # Model recorded workingstate t vector
    common_vals, workingstate_index, rawouput_index = intersect_mtlb(workingstate.time, rawoutput.t)

    # Get field names to make our final result class
    field_names = []
    for property, value in vars(workingstate).items():
        field_names.append(property)

    # Convert our workingstates to arrays for indexing and get our state class
    if sensanal.key == 0:
        workingstate.convert_to_array()
        state = pySCION_classes.State_class(workingstate, workingstate_index)
        run = pySCION_classes.Run_class(state, gridstate, pars, model_pars,
                                      start_pars, forcings, erosion_pars)

    else:
        workingstate.convert_to_array()
        state = pySCION_classes.State_class_sensanal(workingstate, workingstate_index, tuning)
        run = pySCION_classes.Run_class(state, gridstate, pars, model_pars,
                                      start_pars, forcings, erosion_pars)

    if sensanal.key == 0:
        # Done message
        print('Done')
        endtime = time.time() - start_time
        print('time:', endtime)

    #####################################################################
    ###########################   Plotting   ############################
    #####################################################################

    # Only plot if no tuning structure exists, only plot fluxes for quick runs
    
    if plotrun.key == 1: # == 1
        pySCION_plot_fluxes.pySCION_plot_fluxes(state, model_pars, pars)
        if runcontrol>-1:
            pySCION_plot_worldgraphic.pySCION_plot_worldgraphic(gridstate, interpstack)

    if sensanal.key == 1:
        return run
    else:
        return run, interpstack, lipstack
