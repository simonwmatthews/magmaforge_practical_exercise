from os import stat
# from symbol import parameters
from typing import Optional, Union

import numpy as np
import pandas as pd
import thermoengine as thermo
from numpy.typing import ArrayLike


import warnings

from magmaforge.state_info import StateData

from .system_calculator import (EquilCalculator, Phase, SolnPhase,
                                SystemCalculator)


class System:
    _melt_frac_cutoff:float
    _liquid_model: SolnPhase
    ELEMS_NO_CO2 = ['H','O','Na','Mg','Al','Si','P','K','Ca','Ti','Cr','Mn',
             'Fe','Co','Ni']
    OXIDES_NO_CO2 = [ 'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO',
              'MnO','MgO','NiO','CoO','CaO','Na2O','K2O','P2O5',
              'H2O']
    ELEMS_HAS_CO2 = ['H','C','O','Na','Mg','Al','Si','P','K','Ca','Ti','Cr','Mn',
             'Fe','Co','Ni']
    OXIDES_HAS_CO2 = [ 'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO',
              'MnO','MgO','NiO','CoO','CaO','Na2O','K2O','P2O5',
              'H2O', 'CO2']
    FE2O3_MIN = 1e-7
    
    @classmethod
    def adjust_init_redox(cls, T0:float, P0:float, comp:dict, del_fO2:float, 
                          O2_buffer:str, model_name:str) -> pd.Series:
        sys_O2 = cls(comp=comp, T0=T0, P=P0, del_fO2=del_fO2, O2_buffer=O2_buffer, 
                                model_name=model_name)

        comp_adj = sys_O2.state.liquid.comp
        assert comp_adj is not None
        return comp_adj
    
    @classmethod
    def get_liquidus_temp(cls, T0:float, P:float, comp:dict, model_name:str) -> float:
        def _step_down_to_liquidus(T0, Tstep, P=P, comp=comp, model_name=model_name):
            sys_T = cls(comp=comp, T0=T0, P=P,
                        melt_frac_cutoff=.99, model_name=model_name)
            sys_T.crystallize(method='equil', Tstep=Tstep)

            T_liquidus = sys_T.history.get_temps()[sys_T.history.get_melt_frac()==1][-1]
            return T_liquidus
        
        T_liquidus = _step_down_to_liquidus(T0, 10)
        T_liquidus = _step_down_to_liquidus(T_liquidus, 1)
        
        return T_liquidus

    
    def __init__(self, comp:Optional[dict]=None, P:float=1, 
                 T0:float=1800, Tfinal:float=900, melt_frac_cutoff:float=0.05,
                 O2_buffer:Optional[str]=None, del_fO2:float=0, frac_remains=1e-2,
                 S0:Optional[float]=None, min_potential:Optional[str]=None,
                 model_name:Optional[str]=None, has_CO2=None, Liq=None, solid_phases=None):
        
        T0 = float(T0)
        P = float(P)

        

        if comp is None:
            raise self.CompNotDefined
        
        if min_potential=='H':
            assert S0 is not None, 'Must define S0 if using enthalpy minimization'
        
        if (Liq is None) and (model_name is None):
            model_name = 'v1.0'
            
            
        if model_name == 'v1.2':
            has_CO2 = True
            # assert has_CO2, 'MELTS v1.2 includes CO2. The has_CO2 flag must be set to True.'
            
    
        if model_name in ['v1.0', 'pMELTS']:
            has_CO2 = False
            # assert not has_CO2, f'{model_name} does NOT include CO2. The has_CO2 flag must be set to False.'
            
            
        if has_CO2:
            self.ELEMS = self.ELEMS_HAS_CO2
            self.OXIDES = self.OXIDES_HAS_CO2
            
        else:
            self.ELEMS = self.ELEMS_NO_CO2
            self.OXIDES = self.OXIDES_NO_CO2
        
        self._frac_remains = frac_remains # settings
        self._Tfinal = Tfinal
        self._melt_frac_cutoff = melt_frac_cutoff
        
        calc = EquilCalculator(self.ELEMS, min_potential=min_potential, model_name=model_name, Liq=Liq)
        
        self._init_system_calculator(calc, min_potential)
        self._init_state_variables(T0, P, O2_buffer, del_fO2, S0)
        self._init_comp(comp)
        self._init_equil_state(P, S_target=S0)

    def _init_system_calculator(self, calculator: SystemCalculator,
                                min_potential:Optional[str]) -> None:
        # self._system_calc = Calculator(
        #     self.ELEMS, min_potential=min_potential)
        self._system_calc = calculator
        self._liquid_model = self._system_calc.get_liquid_model()
        
    
    def _init_state_variables(self, T0:float, P:float, O2_buffer:Optional[str], 
                              del_fO2:float, S0:Optional[float]):
        
        self._T0 = T0
        self._T = T0
        self._P = P
        self._del_NNO = self._remap_O2_buffer_to_NNO(O2_buffer, del_fO2)
        
    
    def _init_comp(self, input_comp:dict):
        comp = self._format_oxide_comp(input_comp, self.OXIDES)
        
        self._validate_comp(comp)        
        bulk_elem_comp = self._convert_oxide_to_elem_comp(comp)

        self._bulk_comp = comp
        self._elem_comp = bulk_elem_comp
        self._residue_elem_comp = pd.Series(0, index=self.ELEMS)

    @classmethod
    def _format_oxide_comp(cls, input_comp:dict[str,float], OXIDES:list[str]) -> dict[str,float]:
        if input_comp is None:
            assert False, 'must provide initial composition as a dict'
            
        comp = {ox:0.0 for ox in OXIDES}
        comp.update(input_comp)
        
        if comp['Fe2O3'] == 0:
            comp['Fe2O3'] = cls.FE2O3_MIN
        
        return comp

    def _validate_comp(self, comp:dict[str,float]):
        if 'Olivine' in self.phase_names:
            Mn_Ni_comp_invalid_for_olivine = (
                (comp['NiO'] > 0 and comp['MnO']==0) or 
                (comp['MnO'] > 0 and comp['NiO']==0))
            if Mn_Ni_comp_invalid_for_olivine:
                raise self.InvalidMnNiCompWithOlivine
    
    def _init_equil_state(self, P, S_target:Optional[float]=None):
        self._system_calc.equilibrate(self._T0, P, elem_comp=self._elem_comp, del_NNO=self._del_NNO, 
                                      S_target=S_target)
        self._history = StateHistory(self._liquid_model)
        self._store_system_state()
        
        self._init_mass = self.total_mass
        self._init_elem_comp = self.elem_comp

    def _remap_O2_buffer_to_NNO(self, O2_buffer, del_fO2):
        if O2_buffer is None:
            del_NNO = None
        elif O2_buffer=='NNO':
            del_NNO = del_fO2
        else:
            assert False, "O2 buffer not implimented, choose from ['none', 'NNO']"
            
        return del_NNO

    def _convert_oxide_to_elem_comp(self, comp):
        mol_elems = self._convert_wtoxides_to_molelems(comp)  
        bulk_elem_comp = self._filter_by_system_elements(mol_elems)
        return bulk_elem_comp

    def _filter_by_system_elements(self, mol_elems):
        bulk_elem_comp = []
        for elem in self.ELEMS:
            index = thermo.chem.PERIODIC_ORDER.tolist().index(elem)
            bulk_elem_comp.append(mol_elems[index])
        bulk_elem_comp = np.array(bulk_elem_comp)
        return bulk_elem_comp
    
    @classmethod
    def convert_wtoxides_to_liquid_endmems(cls, comp:dict[str,float], Liq:SolnPhase) \
        -> ArrayLike:
            
        comp = cls._format_oxide_comp(comp, cls.OXIDES_HAS_CO2)
        mol_oxides = thermo.chem.format_mol_oxide_comp(
            comp, convert_grams_to_moles=True)
        moles_end, oxide_res = Liq.calc_endmember_comp(
            mol_oxide_comp=mol_oxides, method='intrinsic', output_residual=True)
        
        if not Liq.test_endmember_comp(moles_end):
            print ("Calculated composition is infeasible!")
        return moles_end
    

    def _convert_wtoxides_to_molelems(self, comp):    
        Liq = self._liquid_model
        moles_end = self.convert_wtoxides_to_liquid_endmems(comp, Liq)        
        mol_elems = Liq.convert_endmember_comp(moles_end,output='moles_elements')
        return mol_elems
    
    # Calculator properties
    @property
    def nonvolatile_phase_names(self):
        return self._system_calc.nonvolatile_phase_names
    
    @property
    def phase_names(self):
        return self._system_calc.phase_names
    
    @property
    def solid_phase_names(self):
        return self._system_calc.solid_phase_names
    
    ###################
    # State properties
    ###################
    @property
    def state(self) -> StateData:
        return self._system_calc.state
    
    @property
    def melt_fraction(self):
        state = self.state
        return state.assemblage.melt_frac
    
    @property
    def total_entropy(self):
        state = self.state
        return state.properties.S
    
    @property
    def total_mass(self):
        state = self.state
        return state.properties.mass_tot
    
    @property
    def mass_of_nonvolatiles(self):
        state = self.state
        return state.mass_of_nonvolatiles
           
    @property
    def mass_fraction(self):
        return self.total_mass/self._init_mass

    @property
    def bulk_comp(self):
        return self._bulk_comp
    
    @property
    def elem_comp(self) -> pd.Series:
        state = self.state
        return state.properties.comp
    
    @property
    def init_elem_comp(self):
        return self._init_elem_comp.copy()
    
    @property
    def residue_elem_comp(self):
        return self._residue_elem_comp
    
    @property
    def T(self):
        return self._T
    
    @property
    def P(self):
        return self._P
    
    @property
    def history(self) -> 'StateHistory':
        return self._history
    
    @property
    def default_calc_args(self) -> dict:
        return self._system_calc.DEFAULT_ARGS
    
    def cool(self, P=None, dT=5, dS=None, del_NNO=None, calc_args:Optional[dict]=None) -> bool:
        """
        cool the system a single step and add to system history
        """
        
        if P is None:
            P = self._P
        
        if dS is not None:
            T = self.T
            S_target = self.total_entropy-dS
            
        else:
            T = self.T - dT
            S_target = None

        try:
            successful = self._equilibrate(
                T, P, self.elem_comp, del_NNO, dT, S_target=S_target, calc_args=calc_args)
        except:
            successful = False

        if successful:
            state = self._system_calc.state
            self._T = state.conditions.T
            self._P = state.conditions.P
            self._store_system_state()       
        
        return successful
    
    def crystallize(self, method='equil', fix_fO2=False, Tstep=15, 
                    calc_args:Optional[dict]=None) -> 'System':
        """
        Fully crystallize system
        """
        del_NNO = self._get_fO2_relative_to_NNO(fix_fO2)
        calc = self._system_calc
        
        while True:
            successful = self.cool(dT=Tstep, del_NNO=del_NNO, calc_args=calc_args)
            
            if self._crystallization_complete(method, self.T):
                break
            
            if not successful:
                # successful = self.cool(dT=Tstep, del_NNO=del_NNO, calc_args=calc_args)
                warnings.warn("Warning: Unsuccessful cooling step ended equilibrium calculation.")
                break
            
            if method=='frac':
                self._residue_elem_comp += calc.fractionate_phases(self._frac_remains)
        
        return self
    
    def fractionate_phases(self):
        calc = self._system_calc
        self._residue_elem_comp += calc.fractionate_phases(self._frac_remains)
        return self
    
    def _store_system_state(self):
        state = self._system_calc.state
        self._history._add_new_state(state)

        
    def _equilibrate(self, T:float, P:float, elem_comp:pd.Series,
                     del_NNO:Optional[float], Tstep:float,
                     S_target:Optional[float]=None,
                     calc_args:Optional[dict]=None):
        calc = self._system_calc
        
        try:
            assert self._cation_comp_approx_equal(self.elem_comp,elem_comp), 'comp cannot change'

            calc.equilibrate(T, P, del_NNO=del_NNO, S_target=S_target, calc_args=calc_args)
            
            assert self._cation_comp_approx_equal(self.elem_comp,elem_comp), 'comp cannot change'
        
        except ValueError:
            return self._restart_equilibrate(T, P, elem_comp, del_NNO, 
                                             Tstep, S_target=S_target, calc_args=calc_args)
        
        return True
        
    
    def _restart_equilibrate(self, T:float, P:float, elem_comp:pd.Series,
                             del_NNO:Optional[float], Tstep:float,
                             S_target:Optional[float]=None,
                             calc_args:Optional[dict]=None):
        

        assert self._cation_comp_approx_equal(self.elem_comp,elem_comp), 'comp cannot change'
        
        
        calc = self._system_calc
        Tbase = T + Tstep
        mini_step_num = 10
        step_count = 0
        
        if calc_args is None:
            calc_args = {}
            
        calc_args['debug'] = 1
        
        try:
            T = Tbase
            
            elem_comp_vals = elem_comp.values
            calc.equilibrate(T, P, elem_comp=elem_comp_vals, S_target=S_target,
                             del_NNO=del_NNO, calc_args=calc_args)
            # calc.equilibrate(T, P, 
            #                  del_NNO=del_NNO, restart=False,
            #                  stats=stats, debug=debug)
            assert self._cation_comp_approx_equal(self.elem_comp,elem_comp), 'comp cannot change'
            
            
        
            # while step_count < mini_step_num:
            #     step_count += 1
            #     T = Tbase - Tstep*step_count/mini_step_num
            #     calc.equilibrate(T, P, del_NNO=del_NNO, 
            #                      stats=stats, debug=debug)
            #     # calc.equilibrate(T, P, elem_comp=elem_comp.values,
            #     #              del_NNO=del_NNO, restart=True,
            #     #              stats=stats, debug=debug)
            
            T = Tbase-Tstep
            calc.equilibrate(T, P, del_NNO=del_NNO, S_target=S_target, calc_args=calc_args)
            
            assert self._cation_comp_approx_equal(self.elem_comp,elem_comp), 'comp cannot change'        
        
        except ValueError:
            return False
        
        return True
    
    
    def _cation_comp_approx_equal(self, elem_comp1:pd.Series, elem_comp2:pd.Series):
        cat_comp1 = elem_comp1.drop('O')
        cat_comp2 = elem_comp2.drop('O')
        return np.allclose(cat_comp1, cat_comp2)
    

    def _crystallization_complete(self, method:str, T:float) -> bool:
        
        if method=='equil':
            if self.melt_fraction < self._melt_frac_cutoff:
                return True
        
        if method=='frac':
            if self.mass_fraction < self._melt_frac_cutoff:
                return True
            
        if T < self._Tfinal:
            return True
        
        return False

    def _get_fO2_relative_to_NNO(self, fix_fO2):
        if fix_fO2:
            del_NNO = self._del_NNO
            if del_NNO is None:
                raise self.O2BufferNotDefined()
        else:
            del_NNO = None
            
        return del_NNO
    
    class O2BufferNotDefined(Exception):
        pass
    
    class InvalidMnNiCompWithOlivine(Exception):
        pass
    
    class CompNotDefined(Exception):
        pass


class StateHistory:
    _Liq: SolnPhase
    _states: list[StateData]
    
    def __init__(self, Liq):
        self._Liq = Liq
        self._states = []

    def _add_new_state(self, state_data: StateData):
        self._expand_state_info(state_data)
        self._states.append(state_data)
    
    def _expand_state_info(self, state_data:StateData):
        Tref = 1673.0
        Liq = self._Liq
        
        T, P, liq_comp = state_data.conditions.T, state_data.conditions.P, state_data.liquid.comp
        assert liq_comp is not None
        mol_endmem = System.convert_wtoxides_to_liquid_endmems(liq_comp, Liq)
        liq_rho = self._calc_phase_density(Liq, T, P, mol_endmem)
        liq_rho_ref = self._calc_phase_density(Liq, Tref, P, mol_endmem)
        
        state_data.liquid.rho = liq_rho
        state_data.liquid.rho_ref = liq_rho_ref
    
    def _calc_phase_density(self, phs:SolnPhase, T:float, P:float, mol_endmem:ArrayLike):
        J_BAR_in_cm3 = 10
        
        vol_J_BAR = phs.volume(T, P, mol=mol_endmem)
        assert vol_J_BAR is not None
        vol = J_BAR_in_cm3*vol_J_BAR
        mol_elems = phs.convert_endmember_comp(mol_endmem,output='moles_elements')
        mass = phs.convert_elements(mol_elems, output='total_grams')
        return mass/vol
    
    @property
    def liquid_comp_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_liquid_comps(), index=self.get_temps())
    
    @property
    def phase_frac_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_phase_fractions(), index=self.get_temps())
    
    def get_liquid_comps(self) -> list[dict]:
        liq_comps = [state.liquid.comp for state in self._states]
        return liq_comps  

    def get_temps(self) -> np.ndarray:
        return np.array([state.conditions.T for state in self._states])
    
    def get_press(self) -> np.ndarray:
        return np.array([state.conditions.P for state in self._states])
    
    def get_total_entropy(self) -> np.ndarray:
        return np.array([state.properties.S for state in self._states])
    
    def get_total_mass(self) -> np.ndarray:
        return np.array([state.properties.mass_tot for state in self._states])
    
    def get_melt_frac(self) -> np.ndarray:
        return np.array([state.assemblage.melt_frac for state in self._states])
    
    def get_phase_fractions(self) -> list[dict]:
        return [state.assemblage.phase_frac for state in self._states]
    
    def get_liquid_densities(self) -> np.ndarray:
        return np.array([state.liquid.rho for state in self._states])

    def get_liquid_ref_densities(self) -> np.ndarray:
        return np.array([state.liquid.rho_ref for state in self._states])

    def calc_chemical_potentials(self) -> list[pd.Series]:
        Liq = self._Liq
        liq_mol_comps = [state.liquid.mol_comp for state in self._states]
        T = self.get_temps()
        P = self.get_press()
        
        chempot = []
        for iT, iP, mol_comp in zip(T, P, liq_mol_comps):
            imu = Liq.chem_potential(iT, iP, mol=mol_comp).flatten()
            chempot.append(pd.Series(imu, index=Liq.endmember_names))
            
        return chempot
