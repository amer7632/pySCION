#define classes for pySCION
import numpy as np
from scipy.interpolate import interp1d

class Variable_parameters_class(object):
    '''
    Class to hold variable parameters used in the SCION calculation/.
    Each of the arguments should be a float or int.
    '''
    def __init__(self, var_dict):
        
        self.telltime = var_dict['telltime'] #time
        self.k_reductant_input  = var_dict['k_reductant_input'] #reductant
        
        # Organic C cycle
        self.k_locb = var_dict['k_locb']
        self.k_mocb = var_dict['k_mocb']
        self.k_ocdeg = var_dict['k_ocdeg']
        
        # Carb C cycle
        self.k_ccdeg = var_dict['k_ccdeg']
        self.k_carbw = var_dict['k_carbw']
        self.k_sfw = var_dict['k_sfw']
        self.basfrac = var_dict['basfrac']
        
        # Carb C cycle dependent
        self.k_mccb = self.k_carbw + self.k_ccdeg - self.k_sfw
        self.k_silw = self.k_mccb - self.k_carbw
        self.k_granw = self.k_silw*(1 - self.basfrac)
        self.k_basw = self.k_silw*self.basfrac
        #S cycle
        self.k_mpsb = var_dict['k_mpsb']
        self.k_mgsb = var_dict['k_mgsb']
        self.k_pyrw = var_dict['k_pyrw']
        self.k_gypw = var_dict['k_gypw']
        self.k_pyrdeg = var_dict['k_pyrdeg']
        self.k_gypdeg = var_dict['k_gypdeg']
        
        # P cycle
        self.k_capb = var_dict['k_capb']
        self.k_fepb = var_dict['k_fepb']
        self.k_mopb = var_dict['k_mopb']
        self.k_phosw = var_dict['k_phosw']
        self.k_landfrac = var_dict['k_landfrac']

        # N cycle
        self.k_nfix = var_dict['k_nfix']
        self.k_denit = var_dict['k_denit']
        
        # Flux for steady state
        self.k_oxidw = self.k_mocb + self.k_locb - self.k_ocdeg - self.k_reductant_input
        
        # Sr cycle
        self.k_Sr_sedw = var_dict['k_Sr_sedw']
        self.k_Sr_mantle = var_dict['k_Sr_mantle']
        self.k_Sr_silw = var_dict['k_Sr_silw']
        self.k_Sr_metam = var_dict['k_Sr_metam']

        # Sr cycle dependent
        self.k_Sr_granw = self.k_Sr_silw*(1 - self.basfrac)
        self.k_Sr_basw = self.k_Sr_silw*self.basfrac
        self.total_Sr_removal = self.k_Sr_granw + self.k_Sr_basw + self.k_Sr_sedw \
                                + self.k_Sr_mantle
        self.k_Sr_sfw = self.total_Sr_removal*(self.k_sfw \
                                /(self.k_sfw + self.k_mccb))
        self.k_Sr_sedb = self.total_Sr_removal*(self.k_mccb \
                                /(self.k_sfw + self.k_mccb))
        # Others?
        self.k_oxfrac = var_dict['k_oxfrac']
        self.newp0 = var_dict['newp0']
        
        # COPSE constant for calculating pO2 from normalised O2
        self.copsek16 = var_dict['copsek16']
        # Oxidative weathering dependency on O2 concentration
        self.a = var_dict['a']
        # Marine organic carbon burial dependency on new production
        self.b = var_dict['b']
        # Fire feedback
        self.kfire = var_dict['kfire']
        # Resevoir present_ day
        self.P0 = var_dict['P0']
        self.O0 = var_dict['O0']
        self.A0 = var_dict['A0']
        self.G0 = var_dict['G0']
        self.C0 = var_dict['C0']
        self.pyr0 = var_dict['pyr0']
        self.gyp0 = var_dict['gyp0']
        self.S0 = var_dict['S0']
        self.cal0 = var_dict['cal0']
        self.N0 = var_dict['N0']
        self.OSr0 = var_dict['OSr0']
        self.SSr0 = var_dict['SSr0']
        # Weathering factors
        self.suture_factor = var_dict['suture_factor']
        self.arc_factor = var_dict['arc_factor']
        self.relict_arc_factor = var_dict['relict_arc_factor']
        self.lip_factor = var_dict['lip_factor']

        # Mechanism tests
        self.pgeog_test = var_dict['palaeogeog_test']
        self.suture_test = var_dict['suture_test']
        self.arc_test = var_dict['arc_test']
        self.degass_test = var_dict['degassing_test']
        self.bio_test = var_dict['biology_test']
        self.lip_weathering_test = var_dict['lip_weathering_test']
        self.lip_degass_test = var_dict['lip_degass_test']

    def get_cw_present(self, co2ppm_present_day, interpstack, lipstack, k_erosion, Xm, K,
                        kw, Ea, z, sigplus1, T0, R, grid_area_km2):
        #### get present day co2
        key_upper_co2_present = np.min(interpstack.co2[(interpstack.co2 - co2ppm_present_day) >= 0])
        key_lower_co2_present = np.max(interpstack.co2[(interpstack.co2 - co2ppm_present_day) <= 0])

        #### find keyframe indexes and fractional contribution for present day
        key_upper_co2_index_present = np.argwhere(interpstack.co2 == key_upper_co2_present)[0][0]
        key_lower_co2_index_present = np.argwhere(interpstack.co2 == key_lower_co2_present)[0][0]

        #### fractional contribution of each keyframe at present day
        #if dist_to_upper_present + dist_to_lower_present == 0:
        contribution_lower_present = 1
        contribution_upper_present = 0

        #present day runoff
        runoff_present = contribution_upper_present*np.copy(interpstack.runoff[:,:,key_upper_co2_index_present,21]) \
                       + contribution_lower_present*np.copy(interpstack.runoff[:,:,key_lower_co2_index_present,21])

        # Air temperature
        Tair_present = contribution_upper_present*np.copy(interpstack.Tair[:,:,key_upper_co2_index_present,21]) \
                     + contribution_lower_present*np.copy(interpstack.Tair[:,:,key_lower_co2_index_present,21])
        T_present = Tair_present + 273
        # Get slope
        tslope_present = np.copy(interpstack.slope[:,:,21])#*25
        # Get arcs and arc mask
        arc_enhancement = np.copy(interpstack.arc_enhancement[:, :, 21])
        # Get sutures and suture mask
        suture_enhancement = np.copy(interpstack.suture_enhancement[:,:,21])
        # Get relict arcs and relict arc masks
        relict_enhancement = np.copy(
            interpstack.relict_arc_enhancement[:, :, 21])

        # get LIPs at present-day and sum (all are 'on' since it's present-day)
        lip_enhancement = np.sum((
        lipstack.afar_enhancement[:, :, 21],
        lipstack.bunbury_enhancement[:, :, 21],
        lipstack.camp_enhancement[:, :, 21],
        lipstack.caribbean_colombian_enhancement[:, :,21],
        lipstack.cimp_a_enhancement[:, :, 21],
        lipstack.cimp_b_enhancement[:, :, 21],
        lipstack.columbia_river_enhancement[:, :, 21],
        lipstack.comei_enhancement[:, :, 21],
        lipstack.deccan_enhancement[:, :, 21],
        lipstack.emeishan_enhancement[:, :, 21],
        lipstack.equamp_enhancement[:, :, 21],
        lipstack.ferrar_enhancement[:, :, 21],
        lipstack.franklin_enhancement[:, :, 21],
        lipstack.gunbarrel_enhancement[:, :, 21],
        lipstack.halip_enhancement[:, :, 21],
        lipstack.irkutsk_enhancement[:, :, 21],
        lipstack.kalkarindji_enhancement[:, :, 21],
        lipstack.karoo_enhancement[:, :, 21],
        lipstack.kola_dnieper_enhancement[:, :, 21],
        lipstack.madagascar_enhancement[:, :, 21],
        lipstack.magdalen_enhancement[:, :, 21],
        lipstack.mundine_well_enhancement[:, :, 21],
        lipstack.naip_enhancement[:, :, 21],
        lipstack.nw_australia_margin_enhancement[:, :,21],
        lipstack.panjal_enhancement[:, :, 21],
        lipstack.parana_etendeka_enhancement[:, :,21],
        lipstack.qiangtang_enhancement[:, :, 21],
        lipstack.seychelles_enhancement[:, :, 21],
        lipstack.siberia_enhancement[:, :, 21],
        lipstack.suordakh_enhancement[:, :, 21],
        lipstack.swcuc_enhancement[:, :, 21],
        lipstack.tarim_enhancement[:, :, 21],
        lipstack.trap_enhancement[:, :, 21],
        lipstack.vilyui_enhancement[:, :, 21],
        lipstack.wichita_enhancement[:, :, 21]), axis=0)

        # Runoff in mm/yr present day
        Q_present = np.copy(runoff_present)
        Q_present[Q_present<0] = 0

        epsilon_present = k_erosion*(Q_present**0.31)*tslope_present \
                         *np.maximum(Tair_present,2)

        # Equations
        r_t_present = np.exp((Ea/(R*T0)) - (Ea/(R*T_present)))
        r_q_present = 1 - np.exp(-1*kw*Q_present)
        R_reg_past = ((z/epsilon_present)**sigplus1)/sigplus1

        # base chemical weathering (cw)
        cw_per_km2_present_raw = 1e6*epsilon_present*Xm*(1 - np.exp(
                        -1*K*r_q_present*r_t_present*R_reg_past
                        ))

        # All weathering
        cw_per_km2_present = cw_per_km2_present_raw * \
            (1 + arc_enhancement + suture_enhancement + relict_enhancement + lip_enhancement)

        # cw total
        cw_present = cw_per_km2_present * grid_area_km2
        cw_present[np.isnan(cw_present)==1] = 0

        self.cw_present = sum(sum(cw_present)) #4.5e8 for palaeogeog=1

    def __bool__ (self):
        return bool(self.telltime)

class Erosion_parameters_class(object):
    '''
    Class to hold erosion parameters used in the SCION calculation/.
    Each of the arguments should be a float or int.
    '''
    def __init__(self, erosion_dict):
        self.Xm = erosion_dict['Xm']
        self.K = erosion_dict['K']
        self.kw = erosion_dict['kw']
        self.Ea = erosion_dict['Ea']
        self.z = erosion_dict['z']
        self.sigplus1 = erosion_dict['sigplus1']
        self.T0 = erosion_dict['T0']
        self.R = erosion_dict['R']

class Model_parameters_class(object):
    '''
    Class to store parameters pertaining to the boundary conditions of the model.
    '''
    def __init__(self, model_pars_dict, interpstack_time):

        self.whenstart = model_pars_dict['whenstart']
        self.whenend = model_pars_dict['whenend']
        self.gridstamp_number = model_pars_dict['gridstamp_number']
        self.finishgrid = model_pars_dict['finishgrid']
        self.bailnumber = model_pars_dict['bailnumber']
        self.display_resolution = model_pars_dict['display_resolution']
        self.runstamps = interpstack_time[interpstack_time > (self.whenstart*1e-6)]
        self.next_gridstamp = self.runstamps[0]
        self.output_length = 0 #will update at the end

    def get_rel_contrib(self, interpstack_lat, interpstack_lon):
        lat_areas = np.cos(interpstack_lat * (np.pi/180))
        self.rel_contrib = np.zeros((len(lat_areas),len(interpstack_lon)))
        for ind, lon in enumerate(interpstack_lon):
            self.rel_contrib[:,ind] = lat_areas / np.mean(lat_areas)

    def __bool__ (self):
        return bool(self.whenstart)

class Starting_parameters_class(object):
    '''
    Class to specfically store the starting conditions of the SCION model that are solved each step.
    These form an array of size (21,) and are passed directly to the ODE.
    '''

    def __init__(self, start_pars_dict):

        self.pstart = start_pars_dict['pstart']
        self.tempstart = start_pars_dict['tempstart']
        self.cal_start = start_pars_dict['cal_start']
        self.N_start = start_pars_dict['N_start']
        self.OSr_start = start_pars_dict['OSr_start']
        self.SSr_start = start_pars_dict['SSr_start']
        self.delta_A_start = start_pars_dict['delta_A_start']
        self.delta_S_start = start_pars_dict['delta_S_start']
        self.delta_G_start = start_pars_dict['delta_G_start']
        self.delta_C_start = start_pars_dict['delta_C_start']
        self.delta_pyr_start = start_pars_dict['delta_pyr_start']
        self.delta_gyp_start = start_pars_dict['delta_gyp_start']
        self.delta_OSr_start = start_pars_dict['delta_OSr_start']
        self.delta_SSr_start = start_pars_dict['delta_SSr_start']
        self.ostart = start_pars_dict['ostart']
        self.astart = start_pars_dict['astart']
        self.sstart = start_pars_dict['sstart']
        self.gstart = start_pars_dict['gstart']
        self.cstart = start_pars_dict['cstart']
        self.pyrstart = start_pars_dict['pyrstart']
        self.gypstart = start_pars_dict['gypstart']

        # Actual input parameters, will be [21,] matrix
        self.startstate = np.zeros(21,)
        self.startstate[0] = self.pstart
        self.startstate[1] = self.ostart
        self.startstate[2] = self.astart
        self.startstate[3] = self.sstart
        self.startstate[4] = self.gstart
        self.startstate[5] = self.cstart
        self.startstate[6] = self.pyrstart
        self.startstate[7] = self.gypstart
        self.startstate[8] = self.tempstart
        self.startstate[9] = self.cal_start
        self.startstate[10] = self.N_start
        self.startstate[11] = self.gstart*self.delta_G_start
        self.startstate[12] = self.cstart*self.delta_C_start
        self.startstate[13] = self.pyrstart*self.delta_pyr_start
        self.startstate[14] = self.gypstart*self.delta_gyp_start
        self.startstate[15] = self.astart*self.delta_A_start
        self.startstate[16] = self.sstart*self.delta_S_start
        self.startstate[17] = self.OSr_start
        self.startstate[18] = self.OSr_start*self.delta_OSr_start
        self.startstate[19] = self.SSr_start
        self.startstate[20] = self.SSr_start*self.delta_SSr_start

    def __bool__ (self):
        return bool(self.whenstart)

class State_class(object):
    '''
    Class for final states (i.e. results)
    '''
    def __init__(self, workingstate, correct_indices):
        #time
        self.time_myr = workingstate.time_myr[correct_indices]
        self.time = workingstate.time[correct_indices]
        #mass conservation (resevoirs)
        self.iso_res_C = workingstate.iso_res_C[correct_indices]
        self.iso_res_S = workingstate.iso_res_S[correct_indices]
        self.res_C = workingstate.res_C[correct_indices]
        self.res_S = workingstate.res_S[correct_indices]
        #basalt and granite temp dependency
        self.temperature = workingstate.temperature[correct_indices]
        self.tempC = workingstate.tempC[correct_indices]
        self.sat_tropical = workingstate.sat_tropical[correct_indices]
        self.sat_equator = workingstate.sat_equator[correct_indices]
        #element resevoirs
        self.P = workingstate.P[correct_indices]
        self.O = workingstate.O[correct_indices]
        self.A = workingstate.A[correct_indices]
        self.S = workingstate.S[correct_indices]
        self.G = workingstate.G[correct_indices]
        self.C = workingstate.C[correct_indices]
        self.N = workingstate.N[correct_indices]
        #mineral resevoirs
        self.pyr = workingstate.pyr[correct_indices]
        self.gyp = workingstate.gyp[correct_indices]
        self.OSr = workingstate.OSr[correct_indices]
        self.SSr = workingstate.SSr[correct_indices]
        #isotope resevoirs1
        self.d13c_A = workingstate.d13c_A[correct_indices]
        self.delta_mccb = workingstate.delta_mccb[correct_indices]
        self.d34s_S = workingstate.d34s_S[correct_indices]
        self.delta_G = workingstate.delta_G[correct_indices]
        self.delta_C = workingstate.delta_C[correct_indices]
        self.delta_pyr = workingstate.delta_pyr[correct_indices]
        self.delta_gyp = workingstate.delta_gyp[correct_indices]
        self.delta_OSr = workingstate.delta_OSr[correct_indices]
        #forcings1
        self.degass = workingstate.degass[correct_indices]
        self.lip_degass = workingstate.lip_degass[correct_indices]
        self.W = workingstate.W[correct_indices]
        self.evo = workingstate.evo[correct_indices]
        self.cpland = workingstate.cpland[correct_indices]
        self.Bforcing = workingstate.Bforcing[correct_indices]
        self.bas_area = workingstate.bas_area[correct_indices]
        self.gran_area = workingstate.gran_area[correct_indices]
        #variables
        self.rco2 = workingstate.rco2[correct_indices]
        self.ro2 = workingstate.ro2[correct_indices]
        self.mro2 = workingstate.mro2[correct_indices]
        self.veg = workingstate.veg[correct_indices]
        self.anox = workingstate.anox[correct_indices]
        self.iceline = workingstate.iceline[correct_indices]
        #fluxes1
        self.mocb = workingstate.mocb[correct_indices]
        self.locb = workingstate.locb[correct_indices]
        self.mccb = workingstate.mccb[correct_indices]
        self.mpsb = workingstate.mpsb[correct_indices]
        self.mgsb = workingstate.mgsb[correct_indices]
        self.silw = workingstate.silw[correct_indices]
        self.carbw = workingstate.carbw[correct_indices]
        self.oxidw = workingstate.oxidw[correct_indices]
        self.basw = workingstate.basw[correct_indices]
        self.granw = workingstate.granw[correct_indices]
        self.phosw = workingstate.phosw[correct_indices]
        self.psea = workingstate.psea[correct_indices]
        self.nfix = workingstate.nfix[correct_indices]
        self.denit = workingstate.denit[correct_indices]
        self.pyrw = workingstate.pyrw[correct_indices]
        self.gypw = workingstate.gypw[correct_indices]
        self.ocdeg = workingstate.ocdeg[correct_indices]
        self.ccdeg = workingstate.ccdeg[correct_indices]
        self.pyrdeg = workingstate.pyrdeg[correct_indices]
        self.gypdeg = workingstate.gypdeg[correct_indices]
        self.sfw = workingstate.sfw[correct_indices]
        self.Sr_granw = workingstate.Sr_granw[correct_indices]
        self.Sr_basw = workingstate.Sr_basw[correct_indices]
        self.Sr_sedw = workingstate.Sr_sedw[correct_indices]
        self.Sr_mantle = workingstate.Sr_mantle[correct_indices]
        self.dSSr = workingstate.dSSr[correct_indices]
        self.relativenewp = workingstate.relativenewp[correct_indices]
        self.lip_past_enhancement = workingstate.lip_past_enhancement[correct_indices]
        self.lip_future_enhancement = workingstate.lip_future_enhancement[correct_indices]
        self.cw_per_km2_past_raw = workingstate.cw_per_km2_past_raw[correct_indices]
        self.cw_per_km2_future_raw = workingstate.cw_per_km2_future_raw[correct_indices]

class State_class_sensanal(object):
    '''
    Class for final states (i.e. results) if doing sensitvity analysis
    '''
    def __init__(self, workingstate, correct_indices, tuning):
        self.bas_area = workingstate.bas_area[correct_indices]
        self.gran_area = workingstate.gran_area[correct_indices]
        self.degass = workingstate.degass[correct_indices]
        self.lip_degass = workingstate.lip_degass[correct_indices]
        self.delta_mccb = workingstate.delta_mccb[correct_indices]
        self.d34s_S = workingstate.d34s_S[correct_indices]
        self.delta_OSr = workingstate.delta_OSr[correct_indices]
        self.SmM = workingstate.SmM[correct_indices]
        self.co2ppm = workingstate.co2ppm[correct_indices]
        self.mro2 = workingstate.mro2[correct_indices]
        self.iceline = workingstate.iceline[correct_indices]
        self.T_gast = workingstate.T_gast[correct_indices]
        self.sat_tropical = workingstate.sat_tropical[correct_indices]
        self.sat_equator = workingstate.sat_equator[correct_indices]
        self.anox = workingstate.anox[correct_indices]
        self.P = workingstate.P[correct_indices]
        self.N = workingstate.N[correct_indices]
        self.time_myr = workingstate.time_myr[correct_indices]
        self.time = workingstate.time[correct_indices]
        self.preplant = workingstate.preplant[correct_indices]

        #tuning
        if tuning.key == 1:
            self.Orel = workingstate.Orel[correct_indices]
            self.Srel = workingstate.Srel[correct_indices]
            self.Arel = workingstate.Arel[correct_indices]
            self.Crel = workingstate.Crel[correct_indices]
            self.Grel = workingstate.Grel[correct_indices]
            self.pyrrel = workingstate.pyrrel[correct_indices]
            self.gyprel = workingstate.gyprel[correct_indices]

class Workingstate_class(object):
    '''
    Class for storing workingstates as the ODE progresses.
    '''
    def __init__(self):
        #time
        self.time_myr, self.time = [], []
        #mass conservation (resevoirs)
        self.iso_res_C, self.iso_res_S, self.res_C, self.res_S = [], [], [], []
        #basalt and granite temp dependency
        self.temperature, self.tempC = [], []
        self.sat_tropical, self.sat_equator = [], []
        #element resevoirs
        self.P, self.O, self.A, self.S, self.G, self.C, self.N = [], [], [], [], [], [], []
        #mineral resevoirs
        self.pyr, self.gyp, self.OSr, self.SSr = [], [], [], []
        #isotope resevoirs1
        self.d13c_A, self.delta_mccb, self.d34s_S, self.delta_G = [], [], [], []
        #isotope resevoirs2
        self.delta_C, self.delta_pyr, self.delta_gyp, self.delta_OSr = [], [], [], []
        #forcings1
        self.degass, self.lip_degass, self.W, self.evo, self.cpland = [], [], [], [], []
        #forcings2
        self.Bforcing, self.bas_area, self.gran_area = [], [], []
        #variables
        self.rco2, self.ro2, self.mro2, self.veg, self.anox, self.iceline = [], [], [], [], [], []
        #fluxes1
        self.mocb, self.locb, self.mccb, self.mpsb, self.mgsb, self.silw = [], [], [], [], [], []
        #fluxes2
        self.carbw, self.oxidw, self.basw, self.granw, self.phosw, self.psea = [], [], [], [], [], []
        #fluxes3
        self.nfix, self.denit, self.pyrw, self.gypw, self.ocdeg = [], [], [], [], []
        #fluxes4
        self.ccdeg, self.pyrdeg, self.gypdeg, self.sfw, self.Sr_granw, self.Sr_basw = [], [], [], [], [], []
        self.Sr_sedw, self.Sr_mantle, self.dSSr, self.relativenewp, self.erosion_tot = [], [], [], [], []
        self.preplant = []
        self.lip_past_enhancement, self.lip_future_enhancement = [], []
        self.cw_per_km2_past_raw, self.cw_per_km2_future_raw = [], []

    def add_workingstates(self, new_data_dict):
        #new data is a list of all the data generated in a solver run
        self.iso_res_C.append(new_data_dict['iso_res_C'])
        self.iso_res_S.append(new_data_dict['iso_res_S'])
        self.res_C.append(new_data_dict['res_C'])
        self.res_S.append(new_data_dict['res_S'])
        self.time.append(new_data_dict['t'])
        self.temperature.append(new_data_dict['temp_gast'])
        self.tempC.append(new_data_dict['tempC'])
        self.P.append(new_data_dict['P'])
        self.O.append(new_data_dict['O'])
        self.A.append(new_data_dict['A'])
        self.S.append(new_data_dict['S'])
        self.G.append(new_data_dict['G'])
        self.C.append(new_data_dict['C'])
        self.pyr.append(new_data_dict['pyr'])
        self.gyp.append(new_data_dict['gyp'])
        self.N.append(new_data_dict['N'])
        self.OSr.append(new_data_dict['OSr'])
        self.SSr.append(new_data_dict['SSr'])
        self.d13c_A.append(new_data_dict['d13c_A'])
        self.delta_mccb.append(new_data_dict['delta_mccb'])
        self.d34s_S.append(new_data_dict['d34s_S'])
        self.delta_G.append(new_data_dict['delta_G'])
        self.delta_C.append(new_data_dict['delta_C'])
        self.delta_pyr.append(new_data_dict['delta_pyr'])
        self.delta_gyp.append(new_data_dict['delta_gyp'])
        self.delta_OSr.append(new_data_dict['delta_OSr'])
        self.degass.append(new_data_dict['degass'])
        self.lip_degass.append(new_data_dict['lip_degass'])
        self.W.append(new_data_dict['W'])
        self.evo.append(new_data_dict['evo'])
        self.cpland.append(new_data_dict['cpland'])
        self.Bforcing.append(new_data_dict['Bforcing'])
        self.bas_area.append(new_data_dict['bas_area'])
        self.gran_area.append(new_data_dict['gran_area'])
        self.rco2.append(new_data_dict['rco2'])
        self.ro2.append(new_data_dict['ro2'])
        self.mro2.append(new_data_dict['mro2'])
        self.veg.append(new_data_dict['veg'])
        self.anox.append(new_data_dict['anox'])
        self.iceline.append(new_data_dict['iceline'])
        self.mocb.append(new_data_dict['mocb'])
        self.locb.append(new_data_dict['locb'])
        self.mccb.append(new_data_dict['mccb'])
        self.mpsb.append(new_data_dict['mpsb'])
        self.mgsb.append(new_data_dict['mgsb'])
        self.silw.append(new_data_dict['silw'])
        self.carbw.append(new_data_dict['carbw'])
        self.oxidw.append(new_data_dict['oxidw'])
        self.basw.append(new_data_dict['basw'])
        self.granw.append(new_data_dict['granw'])
        self.phosw.append(new_data_dict['phosw'])
        self.psea.append(new_data_dict['psea'])
        self.nfix.append(new_data_dict['nfix'])
        self.denit.append(new_data_dict['denit'])
        self.pyrw.append(new_data_dict['pyrw'])
        self.gypw.append(new_data_dict['gypw'])
        self.ocdeg.append(new_data_dict['ocdeg'])
        self.ccdeg.append(new_data_dict['ccdeg'])
        self.pyrdeg.append(new_data_dict['pyrdeg'])
        self.gypdeg.append(new_data_dict['gypdeg'])
        self.sfw.append(new_data_dict['sfw'])
        self.Sr_granw.append(new_data_dict['Sr_granw'])
        self.Sr_basw.append(new_data_dict['Sr_basw'])
        self.Sr_sedw.append(new_data_dict['Sr_sedw'])
        self.Sr_mantle.append(new_data_dict['Sr_mantle'])
        self.dSSr.append(new_data_dict['dSSr'])
        self.relativenewp.append(new_data_dict['relativenewp'])
        self.erosion_tot.append(new_data_dict['erosion_tot'])
        self.time_myr.append(new_data_dict['t_geol'])
        self.preplant.append(new_data_dict['preplant'])
        self.sat_tropical.append(new_data_dict['sat_tropical'])
        self.sat_equator.append(new_data_dict['sat_equator'])
        self.lip_past_enhancement.append(new_data_dict['lip_past_enhancement'])
        self.lip_future_enhancement.append(new_data_dict['lip_future_enhancement'])
        self.cw_per_km2_past_raw.append(new_data_dict['cw_per_km2_past_raw'])
        self.cw_per_km2_future_raw.append(new_data_dict['cw_per_km2_future_raw'])

    def convert_to_array(self):
       
        self.iso_res_C = np.asarray(self.iso_res_C) 
        self.iso_res_S = np.asarray(self.iso_res_S)
        self.res_C = np.asarray(self.res_C)
        self.res_S = np.asarray(self.res_S)
        self.time = np.asarray(self.time)
        self.temperature = np.asarray(self.temperature)
        self.tempC = np.asarray(self.tempC)
        self.P = np.asarray(self.P)
        self.O = np.asarray(self.O)
        self.A = np.asarray(self.A)
        self.S = np.asarray(self.S)
        self.G = np.asarray(self.G)
        self.C = np.asarray(self.C)
        self.pyr = np.asarray(self.pyr)
        self.gyp = np.asarray(self.gyp)
        self.N = np.asarray(self.N)
        self.OSr = np.asarray(self.OSr)
        self.SSr = np.asarray(self.SSr)
        self.d13c_A = np.asarray(self.d13c_A)
        self.delta_mccb = np.asarray(self.delta_mccb)
        self.d34s_S = np.asarray(self.d34s_S)
        self.delta_G = np.asarray(self.delta_G)
        self.delta_C = np.asarray(self.delta_C)
        self.delta_pyr = np.asarray(self.delta_pyr)
        self.delta_gyp = np.asarray(self.delta_gyp)
        self.delta_OSr = np.asarray(self.delta_OSr)
        self.degass = np.asarray(self.degass)
        self.lip_degass = np.asarray(self.lip_degass)
        self.W = np.asarray(self.W)
        self.evo = np.asarray(self.evo)
        self.cpland = np.asarray(self.cpland)
        self.Bforcing = np.asarray(self.Bforcing)
        self.bas_area = np.asarray(self.bas_area)
        self.gran_area = np.asarray(self.gran_area)
        self.rco2 = np.asarray(self.rco2)
        self.ro2 = np.asarray(self.ro2)
        self.mro2 = np.asarray(self.mro2)
        self.veg = np.asarray(self.veg)
        self.anox = np.asarray(self.anox)
        self.iceline = np.asarray(self.iceline)
        self.mocb = np.asarray(self.mocb)
        self.locb = np.asarray(self.locb)
        self.mccb = np.asarray(self.mccb)
        self.mpsb = np.asarray(self.mpsb)
        self.mgsb = np.asarray(self.mgsb)
        self.silw = np.asarray(self.silw)
        self.carbw = np.asarray(self.carbw)
        self.oxidw = np.asarray(self.oxidw)
        self.basw = np.asarray(self.basw)
        self.granw = np.asarray(self.granw)
        self.phosw = np.asarray(self.phosw)
        self.psea = np.asarray(self.psea)
        self.nfix = np.asarray(self.nfix)
        self.denit = np.asarray(self.denit)
        self.pyrw = np.asarray(self.pyrw)
        self.gypw = np.asarray(self.gypw)
        self.ocdeg = np.asarray(self.ocdeg)
        self.ccdeg = np.asarray(self.ccdeg)
        self.pyrdeg = np.asarray(self.pyrdeg)
        self.gypdeg = np.asarray(self.gypdeg)
        self.sfw = np.asarray(self.sfw)
        self.Sr_granw = np.asarray(self.Sr_granw)
        self.Sr_basw = np.asarray(self.Sr_basw)
        self.Sr_sedw = np.asarray(self.Sr_sedw)
        self.Sr_mantle = np.asarray(self.Sr_mantle)
        self.dSSr = np.asarray(self.dSSr)
        self.relativenewp = np.asarray(self.relativenewp)
        self.erosion_tot = np.asarray(self.erosion_tot)
        self.time_myr = np.asarray(self.time_myr)
        self.preplant = np.asarray(self.preplant)
        self.sat_tropical = np.asarray(self.sat_tropical)
        self.sat_equator = np.asarray(self.sat_equator)
        self.lip_past_enhancement = np.asarray(self.lip_past_enhancement)
        self.lip_future_enhancement = np.asarray(self.lip_future_enhancement)
        self.cw_per_km2_past_raw = np.asarray(self.cw_per_km2_past_raw)
        self.cw_per_km2_future_raw = np.asarray(self.cw_per_km2_future_raw)
class Workingstate_class_sensanal(object):
    '''
    Class for storing workingstates as the ODE progresses.
    '''
    def __init__(self):
        #time
        self.bas_area = []
        self.gran_area = []
        self.degass = []
        self.lip_degass = []
        self.delta_mccb = []
        self.d34s_S = []
        self.delta_OSr = []
        self.SmM = []
        self.co2ppm = []
        self.mro2 = []
        self.iceline = []
        self.T_gast = []
        self.sat_tropical = []
        self.sat_equator = []
        self.anox = []
        self.P = []
        self.N = []
        self.time_myr = []
        self.time = []
        self.preplant = []
        
        # tuning states
        self.Orel = []
        self.Srel = []
        self.Arel = []
        self.Crel = []
        self.Grel = []
        self.pyrrel = []
        self.gyprel = []

    def convert_to_array(self):
        self.bas_area = np.asarray(self.bas_area)
        self.gran_area = np.asarray(self.gran_area)
        self.degass = np.asarray(self.degass)
        self.lip_degass = np.asarray(self.lip_degass)
        self.delta_mccb = np.asarray(self.delta_mccb)
        self.d34s_S = np.asarray(self.d34s_S)
        self.delta_OSr = np.asarray(self.delta_OSr)
        self.SmM = np.asarray(self.SmM)
        self.co2ppm = np.asarray(self.co2ppm)
        self.mro2 = np.asarray(self.mro2)
        self.iceline = np.asarray(self.iceline)
        self.T_gast = np.asarray(self.T_gast)
        self.sat_tropical = np.asarray(self.sat_tropical)
        self.sat_equator = np.asarray(self.sat_equator)
        self.anox = np.asarray(self.anox)
        self.P = np.asarray(self.P)
        self.N = np.asarray(self.N)
        self.time_myr = np.asarray(self.time_myr)
        self.time = np.asarray(self.time)
        self.preplant = np.asarray(self.preplant)
        
        # tuning states
        self.Orel = np.asarray(self.Orel)
        self.Srel = np.asarray(self.Srel)
        self.Arel = np.asarray(self.Arel)
        self.Crel = np.asarray(self.Crel)
        self.Grel = np.asarray(self.Grel)
        self.pyrrel = np.asarray(self.pyrrel)
        self.gyprel = np.asarray(self.gyprel)

class Run_class(object):

    def __init__(self, state, gridstate, pars, model_pars, start_pars, forcings,
                 erosion_pars):
        self.state = state
        self.gridstate = gridstate
        self.pars = pars
        self.model_pars = model_pars
        self.start_pars = start_pars
        self.forcings = forcings
        self.erosion_pars = erosion_pars

class Run_class_sensanal(object):

    def __init__(self):
        self.state = []
        self.gridstate = []
        self.pars = []
        self.model_pars = []
        self.start_pars = []
        self.forcings = []

class Interpstack_class(object):

    def __init__(self, Interpstack_class):

        self.time = Interpstack_class['interp_time']
        self.co2 = Interpstack_class['co2'].astype(float)
        self.Tair = Interpstack_class['Tair'].astype(float)
        self.runoff = Interpstack_class['runoff'].astype(float)
        self.land = Interpstack_class['land']
        self.lat =  Interpstack_class['lat']
        self.lon =  Interpstack_class['lon']
        self.topo = Interpstack_class['topo'].astype(float)
        self.aire = Interpstack_class['aire'].astype(float)
        self.gridarea = Interpstack_class['gridarea']
        self.suture = Interpstack_class['suture']
        self.arc = Interpstack_class['arc']
        self.relict_arc = Interpstack_class['relict_arc']
        self.slope = Interpstack_class['slope']

    def get_enhancements(self, pars):
        # get enhancements
        self.arc_enhancement = (self.arc*(pars.arc_factor - 1))
        self.suture_enhancement = (self.suture*(pars.suture_factor - 1))
        self.relict_arc_enhancement = (self.relict_arc*(pars.relict_arc_factor - 1))

    def __bool__ (self):
        return bool(self.telltime)

class Lipstack_class(object):

    def __init__(self, lipstack_dict):

        # load in lips individually so we can track each one?
        self.afar = lipstack_dict['afar']
        self.bunbury = lipstack_dict['bunbury']
        self.camp = lipstack_dict['camp']
        self.caribbean_colombian = lipstack_dict['caribbean_colombian']
        self.cimp_a = lipstack_dict['cimp_a']
        self.cimp_b = lipstack_dict['cimp_b']
        self.columbia_river = lipstack_dict['columbia_river']
        self.comei = lipstack_dict['comei']
        self.deccan = lipstack_dict['deccan']
        self.emeishan = lipstack_dict['emeishan']
        self.equamp = lipstack_dict['equamp']
        self.ferrar = lipstack_dict['ferrar']
        self.franklin = lipstack_dict['franklin']
        self.gunbarrel = lipstack_dict['gunbarrel']
        self.halip = lipstack_dict['halip']
        self.irkutsk = lipstack_dict['irkutsk']
        self.kalkarindji = lipstack_dict['kalkarindji']
        self.karoo = lipstack_dict['karoo']
        self.kola_dnieper = lipstack_dict['kola_dnieper']
        self.madagascar = lipstack_dict['madagascar']
        self.magdalen = lipstack_dict['magdalen']
        self.mundine_well = lipstack_dict['mundine_well']
        self.naip = lipstack_dict['naip']
        self.nw_australia_margin = lipstack_dict['nw_australia_margin']
        self.panjal = lipstack_dict['panjal']
        self.parana_etendeka = lipstack_dict['parana_etendeka']
        self.qiangtang = lipstack_dict['qiangtang']
        self.seychelles = lipstack_dict['seychelles']
        self.siberia = lipstack_dict['siberia']
        self.suordakh = lipstack_dict['suordakh']
        self.swcuc = lipstack_dict['swcuc']
        self.tarim = lipstack_dict['tarim']
        self.trap = lipstack_dict['trap']
        self.vilyui = lipstack_dict['vilyui']
        self.wichita = lipstack_dict['wichita']
        
        # start times so we can turn them off/on
        self.afar_start = lipstack_dict['afar_start']
        self.bunbury_start = lipstack_dict['bunbury_start']
        self.camp_start = lipstack_dict['camp_start']
        self.caribbean_colombian_start = lipstack_dict['caribbean_colombian_start']
        self.cimp_a_start = lipstack_dict['cimp_a_start']
        self.cimp_b_start = lipstack_dict['cimp_b_start']
        self.columbia_river_start = lipstack_dict['columbia_river_start']
        self.comei_start = lipstack_dict['comei_start']
        self.deccan_start = lipstack_dict['deccan_start']
        self.emeishan_start = lipstack_dict['emeishan_start']
        self.equamp_start = lipstack_dict['equamp_start']
        self.ferrar_start = lipstack_dict['ferrar_start']
        self.franklin_start = lipstack_dict['franklin_start']
        self.gunbarrel_start = lipstack_dict['gunbarrel_start']
        self.halip_start = lipstack_dict['halip_start']
        self.irkutsk_start = lipstack_dict['irkutsk_start']
        self.kalkarindji_start = lipstack_dict['kalkarindji_start']
        self.karoo_start = lipstack_dict['karoo_start']
        self.kola_dnieper_start = lipstack_dict['kola_dnieper_start']
        self.madagascar_start = lipstack_dict['madagascar_start']
        self.magdalen_start = lipstack_dict['magdalen_start']
        self.mundine_well_start = lipstack_dict['mundine_well_start']
        self.naip_start = lipstack_dict['naip_start']
        self.nw_australia_margin_start = lipstack_dict['nw_australia_margin_start']
        self.panjal_start = lipstack_dict['panjal_start']
        self.parana_etendeka_start = lipstack_dict['parana_etendeka_start']
        self.qiangtang_start = lipstack_dict['qiangtang_start']
        self.seychelles_start = lipstack_dict['seychelles_start']
        self.siberia_start = lipstack_dict['siberia_start']
        self.suordakh_start = lipstack_dict['suordakh_start']
        self.swcuc_start = lipstack_dict['swcuc_start']
        self.tarim_start = lipstack_dict['tarim_start']
        self.trap_start = lipstack_dict['trap_start']
        self.vilyui_start = lipstack_dict['vilyui_start']
        self.wichita_start = lipstack_dict['wichita_start']


    def sigmf(self, x, a, c):
        '''
        sigmoid function to determine whether lips are on/off
        x: model time (t_geol)
        a: 10 (other side of 0)
        c: emplace age
        '''
        return 1/(1 + np.exp(-a*(x-c)))
    
    def get_lip_enhancements(self, pars):
        # get enhancements
        self.afar_enhancement = (self.afar*(pars.lip_factor - 1))
        self.bunbury_enhancement = (self.bunbury*(pars.lip_factor - 1))
        self.camp_enhancement = (self.camp*(pars.lip_factor - 1))
        self.caribbean_colombian_enhancement = (self.caribbean_colombian*(pars.lip_factor - 1))
        self.cimp_a_enhancement = (self.cimp_a*(pars.lip_factor - 1))
        self.cimp_b_enhancement = (self.cimp_b*(pars.lip_factor - 1))
        self.columbia_river_enhancement = (self.columbia_river*(pars.lip_factor - 1))
        self.comei_enhancement = (self.comei*(pars.lip_factor - 1))
        self.deccan_enhancement = (self.deccan*(pars.lip_factor - 1))
        self.emeishan_enhancement = (self.emeishan*(pars.lip_factor - 1))
        self.equamp_enhancement = (self.equamp*(pars.lip_factor - 1))
        self.ferrar_enhancement = (self.ferrar*(pars.lip_factor - 1))
        self.franklin_enhancement = (self.franklin*(pars.lip_factor - 1))
        self.gunbarrel_enhancement = (self.gunbarrel*(pars.lip_factor - 1))
        self.halip_enhancement = (self.halip*(pars.lip_factor - 1))
        self.irkutsk_enhancement = (self.irkutsk*(pars.lip_factor - 1))
        self.kalkarindji_enhancement = (self.kalkarindji*(pars.lip_factor - 1))
        self.karoo_enhancement = (self.karoo*(pars.lip_factor - 1))
        self.kola_dnieper_enhancement = (self.kola_dnieper*(pars.lip_factor - 1))
        self.madagascar_enhancement = (self.madagascar*(pars.lip_factor - 1))
        self.magdalen_enhancement = (self.magdalen*(pars.lip_factor - 1))
        self.mundine_well_enhancement = (self.mundine_well*(pars.lip_factor - 1))
        self.naip_enhancement = (self.naip*(pars.lip_factor - 1))
        self.nw_australia_margin_enhancement = (self.nw_australia_margin*(pars.lip_factor - 1))
        self.panjal_enhancement = (self.panjal*(pars.lip_factor - 1))
        self.parana_etendeka_enhancement = (self.parana_etendeka*(pars.lip_factor - 1))
        self.qiangtang_enhancement = (self.qiangtang*(pars.lip_factor - 1))
        self.seychelles_enhancement = (self.seychelles*(pars.lip_factor - 1))
        self.siberia_enhancement = (self.siberia*(pars.lip_factor - 1))
        self.suordakh_enhancement = (self.suordakh*(pars.lip_factor - 1))
        self.swcuc_enhancement = (self.swcuc*(pars.lip_factor - 1))
        self.tarim_enhancement = (self.tarim*(pars.lip_factor - 1))
        self.trap_enhancement = (self.trap*(pars.lip_factor - 1))
        self.vilyui_enhancement = (self.vilyui*(pars.lip_factor - 1))
        self.wichita_enhancement = (self.wichita*(pars.lip_factor - 1))

class Forcings_class(object):

    def __init__(self, forcings_dict):

        self.t = forcings_dict['t']
        self.B = forcings_dict['B']
        self.Ca = forcings_dict['Ca']
        self.CP = forcings_dict['CP']
        self.D = forcings_dict['D']
        self.E =  forcings_dict['E']
        #self.GA =  forcings_dict['GA']
        self.PG =  forcings_dict['PG']
        self.U =  forcings_dict['U']
        self.W  =  forcings_dict['W']
        self.coal = forcings_dict['coal']
        self.epsilon = forcings_dict['epsilon']

        self.Ba = np.asarray(
            [forcings_dict['Ba_df']['t'].to_numpy()*1e6,
             forcings_dict['Ba_df']['BA'].to_numpy()]
            )
        self.Ga = np.asarray(
            [forcings_dict['Ga_df']['t'].to_numpy()*1e6,
             forcings_dict['Ga_df']['GA'].to_numpy()]
             )

        #degassing
        # leave blank ('') for default
        degass_key = ''  # _constant_mors, _constant_arcs, _constant_rifts
        self.D_force_x = forcings_dict['degassing']['D_force_x'][0] # time
        # Reshape these next three to (601,)
        self.D_force_min = forcings_dict['degassing'][f'D_force_min{degass_key}'].reshape(
            len(forcings_dict['degassing'][f'D_force_min{degass_key}']),
            ) 
        self.D_force_mean = forcings_dict['degassing'][f'D_force_mean{degass_key}'].reshape(
            len(forcings_dict['degassing'][f'D_force_mean{degass_key}']),
            )
        self.D_force_max = forcings_dict['degassing'][f'D_force_max{degass_key}'].reshape(
            len(forcings_dict['degassing'][f'D_force_max{degass_key}']),
            )
        self.shoreline_time = forcings_dict['shoreline']['shoreline_time'][0]
        self.shoreline_relative = forcings_dict['shoreline']['shoreline_relative'][0]

        # LIP degassing
        self.D_lip_ages = forcings_dict['lip_degassing']['age_Ma'].values
        self.D_lip_durations = forcings_dict['lip_degassing']['duration_Ma'].values
        self.D_lip_fluxes = forcings_dict['lip_degassing']['mol_C_per_a'].values

    def get_interp_forcings(self):
        '''
          Make interpolations of these forcings
        '''
        self.E_reloaded_INTERP = interp1d(1e6 * self.t, self.E)
        self.W_reloaded_INTERP = interp1d(1e6 * self.t, self.W)
        self.Ba_reloaded_INTERP = interp1d(self.Ba[0], self.Ba[1])
        self.Ga_reloaded_INTERP = interp1d(self.Ga[0], self.Ga[1])
        self.D_complete_min_INTERP = interp1d(self.D_force_x, self.D_force_min)
        self.D_complete_mean_INTERP = interp1d(self.D_force_x, self.D_force_mean)
        self.D_complete_max_INTERP = interp1d(self.D_force_x, self.D_force_max)
        self.shoreline_INTERP = interp1d(self.shoreline_time, self.shoreline_relative)
        self.f_biot_INTERP = interp1d([-1000e6, -525e6, -520e6, 0],[0, 0, 1, 1])
        self.cb_INTERP = interp1d([0, 1], [1.2, 1])       

    def __bool__ (self):
        return bool(self.telltime)

class Stepnumber_class(object):

    def __init__(self, step):
        self.step = step

class Gridstate_class(object):
    def __init__(self, gridstate_array):
        self.time_myr = np.copy(gridstate_array)
        self.land = np.copy(gridstate_array)
        self.base_chem_weathering = np.copy(gridstate_array)
        self.suture = np.copy(gridstate_array)
        self.suture_weathering = np.copy(gridstate_array)
        self.arc = np.copy(gridstate_array)
        self.arc_weathering = np.copy(gridstate_array)
        self.relict_arc = np.copy(gridstate_array)
        self.relict_arc_weathering = np.copy(gridstate_array)
        self.lip = np.copy(gridstate_array)
        self.lip_weathering = np.copy(gridstate_array)
        self.Q = np.copy(gridstate_array)
        self.Tair = np.copy(gridstate_array)
        self.topo = np.copy(gridstate_array)
        self.cw = np.copy(gridstate_array)
        self.cwcarb = np.copy(gridstate_array)
        self.epsilon = np.copy(gridstate_array)
        self.preplant = np.copy(gridstate_array)

class Sensanal_class(object):
    def __init__(self, sensanal_key):
        self.key = sensanal_key

    def __bool__ (self):
        return bool(self.key)
    
class Tuning_class(object):
    def __init__(self, tuning_key):
        self.key = tuning_key

    def __bool__ (self):
        return bool(self.key)    

class Plotrun_class(object):
    def __init__(self, plotrun_key):
        self.key = plotrun_key

    def __bool__ (self):
        return bool(self.key)


class Sensparams_class(object):

    def __init__(self, randminusplus1, randminusplus2, randminusplus3,
                 randminusplus4, randminusplus5, randminusplus6, randminusplus7,
                 randminusplus8):
        self.randminusplus1 = randminusplus1
        self.randminusplus2 = randminusplus2
        self.randminusplus3 = randminusplus3
        self.randminusplus4 = randminusplus4
        self.randminusplus5 = randminusplus5
        self.randminusplus6 = randminusplus6
        self.randminusplus7 = randminusplus7
        self.randminusplus8 = randminusplus8

class Sens_class(object):
    '''
    for storing
    '''
    def __init__(self):
        self.bas_area = []
        self.gran_area = []
        self.degass = []
        self.delta_mccb = []
        self.d34s_S = []
        self.delta_OSr = []
        self.SmM = []
        self.co2ppm = []
        self.mro2 = []
        self.iceline = []
        self.T_gast = []
        self.sat_tropical = []
        self.sat_equator = []
        self.anox = []
        self.P = []
        self.N = []
        self.time_myr = []
        self.time = []

    def add_states(self, new_data_dict):
        self.bas_area.append(new_data_dict['interp_bas_area'])
        self.gran_area.append(new_data_dict['interp_gran_area'])
        self.degass.append(new_data_dict['interp_degass'])
        self.delta_mccb.append(new_data_dict['interp_delta_mccb'])
        self.d34s_S.append(new_data_dict['interp_d34s_S'])
        self.delta_OSr.append(new_data_dict['interp_delta_OSr'])
        self.SmM.append(new_data_dict['interp_SmM'])
        self.co2ppm.append(new_data_dict['interp_co2ppm'])
        self.mro2.append(new_data_dict['interp_mro2'])
        self.iceline.append(new_data_dict['interp_iceline'])
        self.T_gast.append(new_data_dict['interp_T_gast'])
        self.sat_tropical.append(new_data_dict['interp_sat_tropical'])
        self.sat_equator.append(new_data_dict['interp_sat_equator'])
        self.anox.append(new_data_dict['interp_anox'])
        self.P.append(new_data_dict['interp_P'])
        self.N.append(new_data_dict['interp_N'])
        self.time_myr.append(new_data_dict['interp_time_myr'])
        self.time.append(new_data_dict['interp_time'])

    def convert_to_array(self):
        self.bas_area = np.asarray(self.bas_area)
        self.gran_area = np.asarray(self.gran_area)
        self.degass = np.asarray(self.degass)
        self.delta_mccb = np.asarray(self.delta_mccb)
        self.d34s_S = np.asarray(self.d34s_S)
        self.delta_OSr = np.asarray(self.delta_OSr)
        self.SmM = np.asarray(self.SmM)
        self.co2ppm = np.asarray(self.co2ppm)
        self.mro2 = np.asarray(self.mro2)
        self.iceline = np.asarray(self.iceline)
        self.T_gast = np.asarray(self.T_gast)
        self.sat_tropical = np.asarray(self.sat_tropical)
        self.sat_equator = np.asarray(self.sat_equator)
        self.anox = np.asarray(self.anox)
        self.P = np.asarray(self.P)
        self.N = np.asarray(self.N)
        self.time_myr = np.asarray(self.time_myr)
        self.time = np.asarray(self.time)
