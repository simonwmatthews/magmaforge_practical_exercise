import copy
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StateData:
    conditions: "Conditions"
    properties: "Properties"
    assemblage: "Assemblage"
    liquid: "PhaseInfo"
    water: "PhaseInfo"
    minerals: Optional[dict[str, "PhaseInfo"]] = None
    
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def mass_of_melt(self) -> float:
        return self.liquid.mass
    
    @property
    def mass_of_water(self):
        return self.water.mass
    
    @property
    def mass_of_nonvolatiles(self):
        return self.properties.mass_tot - self.mass_of_water
    

    



    
@dataclass
class PhaseInfo:
    mass: float
    rho: Optional[float] = None
    rho_ref: Optional[float] = None
    comp: Optional[pd.Series] = None
    mol_comp: Optional[pd.Series] = None
    
@dataclass
class Conditions:
    T: float
    P: float
    fO2: Optional[float] = None
    
@dataclass
class Properties:
    mass_tot: float
    comp: Optional[pd.Series]
    S: Optional[float] = None
    V: Optional[float] = None

@dataclass
class Assemblage:
    melt_frac: float 
    phase_frac: dict[str, float]

    
    