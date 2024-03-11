from dataclasses import dataclass, replace
from email.policy import default
from os import stat
from statistics import mode
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
import thermoengine as thermo
from numpy.typing import ArrayLike

from . import state_info
from .state_info import StateData

EQstate = thermo.equilibrate.EquilState
Phase = thermo.phases.Phase
SolnPhase = thermo.phases.SolutionPhase

entropy_target: Optional[float] = None

class SystemCalculator(Protocol):
    @property
    def state(self) -> StateData:
        ...
    def get_liquid_model(self) -> SolnPhase:
        ...
    @property
    def phase_names(self) -> list[str]:
        ...
    @property
    def solid_phase_names(self) -> list[str]:
        ...
    @property
    def nonvolatile_phase_names(self) -> list[str]:
        ...
    @property
    def nonmelt_phase_names(self) -> list[str]:
        ...
    def fractionate_phases(self, frac_remains:float) -> None:
        ...
    def equilibrate(self, T:float, P:float,
                    elem_comp:Optional[ArrayLike]=None, S_target:Optional[float]=None,
                    del_NNO:Optional[float]=None, calc_args:Optional[dict]=None) -> None:
        ...
        
    
class EquilCalculator:
    _calc_state: EQstate
    _state: Optional[StateData] = None
    MODELS_IMPLEMENTED = ['v1.0','v1.2','pMELTS']
    
    OXIDES = [ 'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Cr2O3', 'FeO',
            'MnO','MgO','NiO','CoO','CaO','Na2O','K2O','P2O5',
            'H2O']
    
    DEFAULT_ARGS = {'stats':False, 'debug':0}
    
    def __init__(self, elems:list[str], model_name:str='v1.0',
                 min_potential:Optional[str]='G', Liq=None) -> None:
        
        if Liq is not None:
            model_name = 'v1.0'
            
        self._elems = elems
        self._init_thermo_database(model_name, Liq=Liq)
        self._init_equilibrium_calculator(min_potential)
    
    def _get_model_class_name(self, model_name:str):
        model_class = {
            'v1.0':'EquilibrateUsingMELTSv102',
            'v1.1':'EquilibrateUsingMELTSv110',
            'v1.2':'EquilibrateUsingMELTSv120',
            'pMELTS':'EquilibrateUsingpMELTSv561',
            'MELTSDEW':'EquilibrateUsingMELTSwithDEW',
            'Stixrude':'EquilibrateUsingStixrude'}
        
        
        if model_name not in self.MODELS_IMPLEMENTED:
            raise self.ModelNotImplemented()
        
        return model_class[model_name]
        
    class ModelNotImplemented(Exception):
        pass
    
    def _load_database_source_code(self, model_class_name:str) -> None:
        src_obj = thermo.core.get_src_object(model_class_name)
        
    def _init_thermo_database(self, model_name:str, Liq=None, solid_phases=None):
        model_class_name = self._get_model_class_name(model_name)
        self._load_database_source_code(model_class_name)
        self._modelDB = thermo.model.Database(liq_mod=model_name)
        self._init_phases(Liq=Liq, solid_phases=solid_phases)

    def _init_phases(self, Liq=None, solid_phases=None):
        default_phases = ['Liq','Fsp','Qz','SplS','Opx','Ol','Cpx']  
        modelDB = self._modelDB
        
        if solid_phases is None:
            solid_phases = {}
            
        phases = solid_phases
        
        if Liq is not None:
            phases['Liq'] = Liq
        
        for phs_abbrev in default_phases:
            if phs_abbrev not in phases:
                phases[phs_abbrev] = modelDB.get_phase(phs_abbrev)
        
        phases['H20'] = thermo.phases.PurePhase('WaterMelts', 'H2O', calib=False)

        
        # Include
        # phases['Fsp'] = modelDB.get_phase('Fsp')
        # phases['Qz'] = modelDB.get_phase('Qz')
        # phases['SplS'] = modelDB.get_phase('SplS')
        # phases['Opx'] = modelDB.get_phase('Opx')
        # # phases['Rhom'] = modelDB.get_phase('Rhom')
        # phases['H20'] = thermo.phases.PurePhase('WaterMelts', 'H2O', calib=False)
        # phases['Ol'] = modelDB.get_phase('Ol')
        # phases['Cpx'] = modelDB.get_phase('Cpx')
        
        
        
        # Ignore
        # phases['OrOx'] = modelDB.get_phase('OrOx')
        # phases['Grt'] = modelDB.get_phase('Grt')
        # phases['Mll'] = modelDB.get_phase('Mll')
        # phases['Cam'] = modelDB.get_phase('Cam')
        # phases['Oam'] = modelDB.get_phase('Oam')
        # phases['Hbl'] = modelDB.get_phase('Hbl')
        # phases['Bt'] = modelDB.get_phase('Bt')
        # phases['NphS'] = modelDB.get_phase('NphS')
        # phases['KlsS'] = modelDB.get_phase('KlsS')
        # phases['LctS'] = modelDB.get_phase('LctS')

        self._phases = phases
        
    ########
    #  API
    ########
    def get_liquid_model(self) -> SolnPhase:
        return self._phases['Liq']
        
    @property
    def state(self) -> StateData:
        state = self._state
        assert state is not None
        return state.copy()
    
    @property
    def phase_names(self) -> list[str]:
        return [phs.phase_name for phs in self._phases.values()] 
    
    def _filter_phase_names(self, exclude_names: list[str]):
        return [name for name in self.phase_names
                if not name.startswith(tuple(exclude_names))]
        
    @property
    def solid_phase_names(self) -> list[str]:
        return self._filter_phase_names(['Liquid','Water'])
    
    @property
    def nonvolatile_phase_names(self) -> list[str]:
        return self._filter_phase_names(['Water'])
        
    @property
    def nonmelt_phase_names(self) -> list[str]:
        return self._filter_phase_names(['Liquid'])
    
    def fractionate_phases(self, frac_remains:float) -> None:
        frac_phase_names = self.nonmelt_phase_names
        new_phs_abundances, residue_elem_comp = self._get_frac_phase_abundances(
            frac_remains, frac_phase_names)
        self._update_phase_abundances(-residue_elem_comp, new_phs_abundances)
        self._update_state()
        return residue_elem_comp
    
    def equilibrate(self, T:float, P:float,
                    elem_comp:Optional[ArrayLike]=None, S_target:Optional[float]=None,
                    del_NNO:Optional[float]=None, calc_args:Optional[dict]=None) -> None:
        equil=self._equil_calc
        
        if S_target is not None:
            self._set_entropy_target(S_target)
        
        calc_args = self._add_default_args(calc_args)
            
        debug = calc_args['debug']
        stats = calc_args['stats']
        
        if elem_comp is None:
            eqstate = self._calc_state
            eqstate = equil.execute(t=T, p=P, con_deltaNNO=del_NNO, state=eqstate, 
                                    debug=debug, stats=stats)
        else:
            eqstate = equil.execute(t=T, p=P, bulk_comp=elem_comp, con_deltaNNO=del_NNO,
                                    debug=debug, stats=stats)
            
        assert eqstate is not None, ('execute is returning None')

        self._calc_state = eqstate
        self._update_state()
    
    #######################
    #  Internal Properties
    #######################
    
    @property
    def _phase_props(self) -> dict:
        return self._calc_state.phase_d
    
    @property
    def _total_mass(self) -> float:
        calc_phase_names = self._phase_props.keys()
        return np.sum([self._calc_state.tot_grams_phase(name) 
                       for name in calc_phase_names])
    
    @property
    def _mass_of_nonvolatiles(self) -> float:
        return np.sum([self._calc_state.tot_grams_phase(name) 
                       for name in self.nonvolatile_phase_names])
    
    @property
    def _total_mass_of_every_phase(self) -> dict[str, float]:
        return {name:self._get_mass_of(name)
                for name in self.phase_names}
        
    @property
    def _total_entropy(self):
        return np.sum([self._calc_state.properties(
            phase_name=name, props='Entropy') for name in self.phase_names])
        
    

    def _get_frac_phase_abundances(self, frac_remains:float, frac_phase_names:list[str]):
        new_phase_abundances = {}
        residue_elem_comp = pd.Series(0, index=self._elems)
        
        for phase_name in frac_phase_names:
            moles_endmem = self._get_mol_endmem_of(phase_name)
            tot_moles = np.sum(moles_endmem)
            if tot_moles > 0:
                new_phase_abundances[phase_name] = frac_remains*moles_endmem
                moles_elem_removed = (1-frac_remains)*np.array(
                    self._get_mol_elems_of(phase_name))
                residue_elem_comp += pd.Series(moles_elem_removed, index=self._elems)
        return new_phase_abundances,residue_elem_comp

    def _update_phase_abundances(self, bulk_comp_change, new_phase_abundances):
        for phase_name, moles_remain in new_phase_abundances.items():
            self._set_mol_endmem_of(phase_name, moles_remain)
        
        self._modify_bulk_comp_by(bulk_comp_change.to_numpy())
        
        

    def _modify_bulk_comp_by(self, dmol_elems:np.ndarray) -> None:
        assert self._equil_calc._bulk_comp is not None
        self._equil_calc._bulk_comp += dmol_elems
            
    def _get_mol_endmem_of(self, phase_name:str) -> np.ndarray:
        return self._phase_props[phase_name]["moles"]
    
    def _set_mol_endmem_of(self, phase_name:str, mol_endmem:np.ndarray) -> None:
        self._phase_props[phase_name]["moles"] = mol_endmem        
    
    def _get_mass_of(self, phase_name:str) -> float:
        return self._calc_state.tot_grams_phase(phase_name)
        
    def _get_mol_elems_of(self, phase_name:str) -> np.ndarray:
        mol_elems = self._calc_state.moles_elements(phase_name)
        assert mol_elems is not None
        return mol_elems
    
    def _set_entropy_target(self, value:float):
        global entropy_target
        entropy_target = value
    
    def _init_equilibrium_calculator(self, min_potential:Optional[str]):
        phase_list = list(self._phases.values())
        if min_potential is None:
            min_potential = 'G'
        
        if min_potential=='G':
            equil_calc = thermo.equilibrate.Equilibrate(self._elems, phase_list)
            
        elif min_potential=='H':

            # self.set_entropy_target(250.0)
            
            def con(T:float, P:float, state:EQstate):
                # S = -state.dGdT(T, P)
                assert entropy_target is not None, 'entropy_target must be set to a valid value.'
                return -entropy_target
            
            equil_calc = thermo.equilibrate.Equilibrate(self._elems, phase_list, 
                                                        lagrange_l=[('T',con)])
            
        else:
            assert False, 'That thermodynamic potential not implemented. Select from ["G"]'
            
        self._equil_calc = equil_calc
        
    def _add_default_args(self, args:Optional[dict]) -> dict:
        if args is None:
            args = {}
        
        for argname, default_value in self.DEFAULT_ARGS.items():
            if argname not in args:
                args[argname] = default_value
                
        return args
        
    def _update_state(self) -> None:
        T = self._calc_state.temperature
        P = self._calc_state.pressure
        conditions = state_info.Conditions(T, P)
        
        mass_tot = self._total_mass
        S_tot = self._total_entropy
        mol_elems = self._calc_state.tot_moles_elements()
        comp = pd.Series(data=mol_elems, index=self._elems)
        properties = state_info.Properties(mass_tot, comp, S_tot)
        
        liq_mass = self._get_mass_of('Liquid')
        melt_frac = liq_mass/self._mass_of_nonvolatiles
        phase_frac = {phs:m/self._total_mass for phs,m in self._total_mass_of_every_phase.items()}
        assemblage = state_info.Assemblage(melt_frac, phase_frac)
        
        liq_mass = self._get_mass_of('Liquid')
        liq_comp = pd.Series(
            self._calc_state.compositions('Liquid', ctype='oxides', units='wt%'),
            index=self.OXIDES+['CO2'])
        liq_mol_comp = pd.Series(
            self._calc_state.compositions('Liquid', ctype='oxides', units='moles'),
            index=self.OXIDES+['CO2']
        )
        liquid = state_info.PhaseInfo(liq_mass, comp=liq_comp, mol_comp=liq_mol_comp)
        
        mass_of_H2O = self._get_mass_of('Water')
        water = state_info.PhaseInfo(mass_of_H2O)
        
        
        self._state = StateData(conditions, properties, assemblage, liquid, water)

        
    # def summarize_state_legacy(self) -> dict:
    #     state = {}

    #     state['T'] = self._get_state().temperature
    #     state['P'] = self._get_state().pressure
    #     state['melt_fraction'] = self.mass_of_melt/self.mass_of_nonvolatiles
    #     state['Stot'] = self.total_entropy
    #     state['liq_comp'] = pd.Series(
    #         self._get_state().compositions('Liquid', ctype='oxides', units='wt%'),
    #         index=self.OXIDES+['CO2'])
    #     state['tot_mass'] = self.total_mass
    #     state['phase_fractions'] =  {phs:m/self.total_mass for phs,m 
    #                                  in self.total_mass_of_every_phase.items()}
    #     return state
        
    
