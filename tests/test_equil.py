import copy
from mimetypes import init

import numpy as np
import pandas as pd
from pytest import approx, fixture, mark, raises

import magmaforge
import thermoengine as thermo


GEOCOMP = {
    'Rhyolite': { 
        'SiO2':  77.5, 
        'TiO2':   0.08, 
        'Al2O3': 12.5, 
        'Fe2O3':  0.207,
        'FeO':    0.473, 
        'MgO':    0.03, 
        'CaO':    0.43, 
        'Na2O':   3.98, 
        'K2O':    4.88, 
        'H2O':    5.5
        },
    'MORB': { 
        'SiO2':  48.68, 
        'TiO2':   1.01, 
        'Al2O3': 17.64, 
        'Fe2O3':  0.89,
        'Cr2O3':  0.0425, 
        'FeO':    7.59, 
        'MgO':    9.10, 
        'CaO':   12.45, 
        'Na2O':   2.65, 
        'K2O':    0.03, 
        'P2O5':   0.08, 
        'H2O':    0.2},
    'Thingmuli': {
        'SiO2':49.91,
        'TiO2': 1.47,
        'Al2O3': 17.91,
        'Fe2O3': 2.45,
        'FeO':  7.02,
        'MnO':  0.0, #0.16
        'MgO':  6.62,
        'CaO':  10.02,
        'Na2O':  3.02,
        'K2O':  0.64,
        'P2O5':  0.2,
        'H2O':  0.01,},
}

@fixture
def comp_library():
    yield copy.deepcopy(GEOCOMP)
       

def allclose(a, b):
    return np.allclose(a, b)

class TestBasicSystem:
    def should_set_initial_comp(self, comp_library):
        # TODO: Add helper method that provides zero values for missing oxides
        comp = comp_library['MORB']
        comp.update({'CoO': 0.0, 'MnO': 0.0, 'NiO': 0.0})
        sys = magmaforge.System(comp=comp)
        assert sys.bulk_comp == comp
        
    def should_exclude_water_from_nonvolatile_phases(self, comp_library):
        comp = comp_library['MORB']
        comp['H2O'] = 5.0
        sys = magmaforge.System(comp=comp)
        assert 'Water' not in sys.nonvolatile_phase_names
        
    def should_start_at_molten_temperature(self):
        sys = magmaforge.System(comp={'SiO2':100})
        assert sys.T > 1000
        
    def should_start_fully_molten(self, comp_library):
        H20_free_MORB = comp_library['MORB']
        H20_free_MORB['H2O'] = 0

        sys = magmaforge.System(comp=H20_free_MORB)
        
        assert sys.melt_fraction == 1
     
    def should_ignore_H2O_when_reporting_melt_fraction(self, comp_library):
        comp = comp_library['MORB']
        comp['H2O'] = 5.0
        sys = magmaforge.System(comp=comp)
        
        assert sys.mass_of_nonvolatiles < sys.total_mass
        assert sys.melt_fraction == 1
        
    def should_throw_exception_if_fix_fO2_without_defined_buffer(self, comp_library):
        sys = magmaforge.System(comp=comp_library['MORB'])
        with raises(magmaforge.System.O2BufferNotDefined):
            sys.crystallize(fix_fO2=True)
            
    @mark.parametrize("comp_adj", [{'MnO': 0, 'NiO': .1},{'MnO': .1, 'NiO': 0}])
    def should_throw_exception_if_olivine_included_with_nonzero_Mn_or_Ni(
        self, comp_adj, comp_library):
        
        MORB = comp_library['MORB']
        MORB.update(comp_adj)
        
        with raises(magmaforge.System.InvalidMnNiCompWithOlivine):
            magmaforge.System(comp=MORB)
            
    def should_determine_liquid_density(self, comp_library):
        MORB = comp_library['MORB']
        sys = magmaforge.System(comp=MORB)
        rho_bnds = [1, 5]
        
        rho0 = sys.history.get_liquid_densities()[0]

        assert ((rho0 > rho_bnds[0]) and (rho0 < rho_bnds[1]))
        
    def should_calculate_chemical_potentials(self, comp_library):
        MORB = comp_library['MORB']
        
        sys = magmaforge.System(P=10e4, comp=MORB)
        chempot_init = sys.history.calc_chemical_potentials()[0]
        
        assert type(chempot_init) is pd.Series
    
    def should_allow_pMELTS_model(self, comp_library):
        MORB = comp_library['MORB']
        sys_pMELTS = magmaforge.System(P=10e4, comp=MORB, model_name='pMELTS')
        
        chempot_init_pMELTS = sys_pMELTS.history.calc_chemical_potentials()[0]
        
        assert sorted(chempot_init_pMELTS.index) == sorted([
            'Si4O8', 'TiO2', 'Al4O6', 'Fe2O3', 'MgCr2O4', 'Fe2SiO4',
            'MnSi0.5O2', 'Mg2SiO4', 'NiSi0.5O2', 'CoSi0.5O2', 'Ca2Si2O6',
            'NaSi0.5O1.5', 'KAlSiO4', 'Ca3(PO4)2', 'H2O'])
        
    def should_give_distinct_chempots_for_MELTS_and_pMELTS_models(self, comp_library):
        MORB = comp_library['MORB']
        sys_MELTS = magmaforge.System(P=10e4, comp=MORB, model_name='v1.0')
        sys_pMELTS = magmaforge.System(P=10e4, comp=MORB, model_name='pMELTS')
        
        chempot_init_MELTS = sys_MELTS.history.calc_chemical_potentials()[0]
        chempot_init_pMELTS = sys_pMELTS.history.calc_chemical_potentials()[0]
        
        assert not chempot_init_pMELTS.equals(chempot_init_MELTS)
        
        
    
        

            
    
class TestEquilCrystallization:
    MELT_FRAC_CUTOFF = 0.9
    
    @fixture(scope='class')
    def MORB_comp(self):
        yield GEOCOMP['MORB'].copy()
    
    @fixture(scope='class')
    def crystallized_MORB(self, MORB_comp, T0=1500, Tstep=5):
        sys = magmaforge.System(comp=MORB_comp, T0=T0, melt_frac_cutoff=self.MELT_FRAC_CUTOFF)
        yield sys.crystallize(Tstep=Tstep)
        
    def should_cool_by_a_stepdown_in_temp(self, MORB_comp, dT=5, Tliquidus=1499):
        sys = magmaforge.System(comp=MORB_comp, T0=Tliquidus) 
        init_melt_frac = sys.melt_fraction
        assert init_melt_frac < 1
        
        sys.cool(dT=dT)
        
        assert sys.T == Tliquidus - dT
        assert sys.melt_fraction < init_melt_frac
     
    def should_cool_by_a_stepdown_in_entropy(self, MORB_comp, S0=250, dS=1, Tliquidus=1499):
        sys = magmaforge.System(comp=MORB_comp, T0=Tliquidus, min_potential='H', S0=S0) 
        S0 = sys.total_entropy
        
        sys.cool(dS=dS)
      
        assert dS == approx(S0 - sys.total_entropy)
    
    def should_progressively_cool(self, crystallized_MORB):
        Temp_history = crystallized_MORB.history.get_temps()
        assert Temp_history[-1] < Temp_history[0]
        assert crystallized_MORB.T == Temp_history[-1]
        
    def should_yield_steady_cooling_history(self, crystallized_MORB):
        Temp_history = crystallized_MORB.history.get_temps()
        assert len(Temp_history) == len(np.unique(Temp_history))

    def should_reach_melt_frac_cutoff(self, crystallized_MORB):
        assert crystallized_MORB.melt_fraction < self.MELT_FRAC_CUTOFF
        
    def should_evolve_composition(self, crystallized_MORB):
        liq_comp = crystallized_MORB.history.get_liquid_comps()
        assert not liq_comp[0].equals(liq_comp[-1])
        
    def should_evolve_liquid_density(self, crystallized_MORB):
        rho_liq = crystallized_MORB.history.get_liquid_densities() 
        assert not rho_liq[0]==rho_liq[-1]
        
    def should_conserve_elemental_abundances(self, crystallized_MORB):
        sys = crystallized_MORB
        final_elem_comp = sys.elem_comp
        assert allclose(final_elem_comp, sys.init_elem_comp)

class TestCustomPhases:
    MELT_FRAC_CUTOFF = 0.9
    
    @fixture(scope='class')
    def MORB_comp(self):
        yield GEOCOMP['MORB'].copy()
    
    
    def should_crystallize_MORB_using_custom_Liq_phase(self, MORB_comp, T0=1500, Tstep=5):
        modelDB = thermo.model.Database(liq_mod='v1.2')
        Liq = modelDB.get_phase('Liq')
        sys = magmaforge.System(comp=MORB_comp, T0=T0, P=10e3, melt_frac_cutoff=self.MELT_FRAC_CUTOFF, 
                                has_CO2=True, Liq=Liq, Tfinal=T0-200)
        sys.crystallize(Tstep=Tstep, method='equil',)
        
        
class TestFracCrystallization:
    @fixture(scope='class')
    def MORB_comp(self):
        yield GEOCOMP['MORB'].copy()
    
    @fixture(scope='class')
    def crystallizing_MORB(self, MORB_comp, Tstep=5, T0=1500, Tfinal=1490):
        sys = magmaforge.System(comp=MORB_comp, T0=T0, Tfinal=Tfinal)
        return sys.crystallize(method='frac', Tstep=Tstep)
    
    def should_progressively_cool(self, crystallizing_MORB):
        Temp_history = crystallizing_MORB.history.get_temps()
        assert Temp_history[-1] < Temp_history[0]
        
    def should_form_crystal_residue(self, crystallizing_MORB: magmaforge.System):
        # TODO: change this to asserting residue mass
        sys = crystallizing_MORB
        
        assert np.sum(sys.residue_elem_comp) > 0
        # assert sys.residue_mass > 0
        
    @mark.xfail
    def should_conserve_total_system_mass(self):
        assert False
        
    def should_conserve_elemental_abundances(self, crystallizing_MORB: magmaforge.System):
        sys = crystallizing_MORB
        
        final_elem_comp = sys.elem_comp + sys.residue_elem_comp
        assert allclose(final_elem_comp, sys.init_elem_comp)

    
class TestO2BufferedEvo:
    @fixture(scope='class')
    def Primitive_Basalt_comp(self):
        comp = GEOCOMP['Thingmuli'].copy()
        yield comp
        
    @fixture(scope='class')
    def crystallized_Basalt(self, Primitive_Basalt_comp, Tstep=10):
        sys = magmaforge.System(comp=Primitive_Basalt_comp, O2_buffer='NNO', del_fO2=-8)
        yield sys.crystallize(method='frac', fix_fO2=True, Tstep=Tstep)


    def should_conserve_elems_except_oxygen(self, crystallized_Basalt: magmaforge.System):
        sys = crystallized_Basalt
        final_elem_comp = sys.elem_comp + sys.residue_elem_comp
        
        assert allclose(final_elem_comp.drop('O'), sys.init_elem_comp.drop('O'))
        assert not allclose(final_elem_comp['O'], sys.init_elem_comp['O'])
        
    def should_go_to_completion(self, crystallized_Basalt: magmaforge.System):
        # TODO: Decrease this tolerance when fO2 buffer is working
        TOL = 0.15
        assert crystallized_Basalt.mass_fraction < TOL 
        
    def should_fully_evolve_liquid_comp(self, crystallized_Basalt:magmaforge.System):
        liq_comp = crystallized_Basalt.history.get_liquid_comps()
        
        Rhyolite_SiO2_min = 69
        SiO2_trend = [icomp['SiO2'] for icomp in liq_comp]
        
        assert np.max(SiO2_trend) >= Rhyolite_SiO2_min