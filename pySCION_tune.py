import pySCION_initialise


def pySCION_tune(tune_dict):


    Otune = tune_dict['Otune']
    Atune = tune_dict['Atune']
    Stune = tune_dict['Stune']
    Gtune = tune_dict['Gtune']
    Ctune = tune_dict['Ctune']
    pyrtune = tune_dict['pyrtune']
    gyptune = tune_dict['gyptune']

    
    res = pySCION_initialise.pySCION_initialise(1)
    
    costfunc = (res.state.O[-1]/res.pars.O0 - 1)**2 \
             + (res.state.A[-1]/res.pars.A0 - 1)**2 \
             + (res.state.S[-1]/res.pars.S0 - 1)**2 \
             + (res.state.G[-1]/res.pars.G0 - 1)**2 \
             + (res.state.C[-1]/res.pars.C0 - 1)**2 \
             + (res.state.pyr[-1]/res.pars.pyr0 - 1)**2 \
             + (res.state.gyp[-1]/res.pars.gyp0 - 1)**2
        
    # print update
    print('Reservoirs: A S G C pyr gyp')
    print('\n')
    print(f'Parameters: {tune_dict}')
    print('\n')
    print('Final vals: ')

    print([res.state.O[-1]/res.pars.O0,
                    res.state.A[-1]/res.pars.A0,
                    res.state.S[-1]/res.pars.S0,
                    res.state.G[-1]/res.pars.G0,
                    res.state.C[-1]/res.pars.C0,
                    res.state.pyr[-1]/res.pars.pyr0,
                    res.state.gyp[-1]/res.pars.gyp0])
    
    print('\n')
    print(f'chisquared: {costfunc}')

    return costfunc