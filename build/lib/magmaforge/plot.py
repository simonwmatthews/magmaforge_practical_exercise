import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import mpltern
from pyrolite.mineral.normative import CIPW_norm

from .system import StateHistory

from typing import Optional

FloatList = list[float]


def _convert_temp_index(df, T_unit):
    if T_unit == 'C':
        df.index = df.index - 273.15
        T_label = 'Temp [C]'
    elif T_unit == 'K':
        T_label = 'Temp [K]'
    else:
        assert False, 'Not a valid T_unit choice. Choose from ["K", "C"].'
        
    return df, T_label
    

def phase_fractions(phase_frac_tbl, ax=None, T_unit='C'):
    mineral_frac_tbl = phase_frac_tbl.drop(columns=['Liquid','Water'])
    
    mineral_frac_tbl, T_label = _convert_temp_index(mineral_frac_tbl, T_unit)
        
    fTOL = 1e-4
    cols = mineral_frac_tbl.max(axis=0)>fTOL
    
    abundant_phs_tbl = mineral_frac_tbl.loc[::-1, cols].astype('float')

    abundant_phs_tbl.index = np.round(abundant_phs_tbl.index).astype('int')
    crystallization_tbl = pd.DataFrame(columns=abundant_phs_tbl.columns, 
                                       index=range(abundant_phs_tbl.index.min(),
                                                   abundant_phs_tbl.index.max()+1),
                                       dtype='float')
    
    crystallization_tbl.update(abundant_phs_tbl)
    crystallization_tbl = crystallization_tbl.interpolate(method='index')

    crystallization_tbl.plot.bar(stacked=True, ax=ax, width=1)

    if ax is None:
        ax = plt.gca()
        
    ax.set_xlabel(T_label)
    ax.set_ylabel('Mass Fraction')
    ax.set_xticks([])
    
def magma_evolution(history:StateHistory, T_unit:str='C', T_lims:Optional[FloatList]=None, title:Optional[str]=None):
    phase_frac_tbl = history.phase_frac_table
    
    liq_comp = history.liquid_comp_table
    liq_comp, T_label = _convert_temp_index(liq_comp, T_unit)

    fig, ax = plt.subplots(nrows=3, sharex=False, squeeze=True, figsize=(5,10) )

    if title is not None:
        ax[0].set_title(title)

    iax = ax[0]
    phase_fractions(phase_frac_tbl, ax=iax, T_unit=T_unit)
    iax.set_xticklabels([])
    iax.set_xlabel('')

    iax=ax[1]
    liq_comp.plot(y=['MgO','FeO','Fe2O3','Al2O3','K2O','Na2O','H2O'], ax=iax).legend(loc='upper left')
    iax.set_ylabel('Magma Comp [wt%]')
    iax.set_xticklabels([])


    iax=ax[2]
    liq_comp.plot(y='SiO2', ax=iax, legend=True)
    iax.set_xlabel(T_label)
    iax.set_ylabel('Magma Comp [wt%]')
    
    if T_lims is not None:
        ax[1].set_xlim(*T_lims)
        ax[2].set_xlim(*T_lims)
    
    data_lims = [liq_comp.index.min(), liq_comp.index.max()]
        
    axT_lims = np.round(ax[2].get_xlim())
    
    _adjust_bar_plot_limits(data_lims, axT_lims, ax[0])

def ternary_plots(history:StateHistory, projection:str='CIPW', **kwargs):
    if projection == 'CIPW':
        ems, un = ternaryEndmembersFromCIPW(history.liquid_comp_table)
    elif projection == 'Sack87':
        ems = TernaryEndmembersFromSack87(history.liquid_comp_table)
    
    fig = plt.figure()
    ax = TernaryPlotAxes(fig)

    ax.plot(ems)
    
def _adjust_bar_plot_limits(data_lims, ax_lims, ax_bar:plt.axis) -> None:
        
    bar_plot_lims = ax_bar.get_xlim()

    def xpos_map(Tvalue):
        slope = (bar_plot_lims[1]-bar_plot_lims[0])/(data_lims[1]-data_lims[0])
        return bar_plot_lims[0] + (Tvalue-data_lims[0])*slope
    
    expand_xlims = [0,0]
    expand_xlims[0] = xpos_map(ax_lims[0])
    expand_xlims[1] = xpos_map(ax_lims[1])
    ax_bar.set_xlim(expand_xlims)


class TernaryPlotAxes():
    def __init__(self, fig):
        self.ax=[]

        self.ax.append(fig.add_axes([0 + 0.1, 0, 0.5, 1], projection='ternary', rotation=-60.0))
        self.ax.append(fig.add_axes([0.25 + 0.1, 0, 0.5, 1], projection='ternary'))

        self.ax[1].set_tlabel('Di')
        self.ax[1].set_llabel('Ol')
        self.ax[1].set_rlabel('Qz')
        self.ax[0].set_llabel('Ne')

        for a in self.ax:
            a.raxis.set_major_formatter("")
            a.laxis.set_major_formatter("")
            a.taxis.set_major_formatter("")
            a.raxis.set_major_locator(MultipleLocator(0.10))
            a.laxis.set_major_locator(MultipleLocator(0.10))
            a.taxis.set_major_locator(MultipleLocator(0.10))
            a.taxis.set_label_rotation_mode('horizontal')
            a.raxis.set_label_rotation_mode('horizontal')
            a.laxis.set_label_rotation_mode('horizontal')
            a.tick_params(length=0)

            a.grid(c='0.5', lw=0.5)
            a.set_axisbelow(True)
    
    def __getitem__(self, key):
        return self.ax[key]

    def scatter(self, data, **kwargs):
        self.ax[1].scatter(data[:, 1][data[:, 3]>0], 
                           data[:, 0][data[:, 3]>0], 
                           data[:, 3][data[:, 3]>0],
                           **kwargs)
        self.ax[0].scatter(data[:, 1][data[:, 4]>0], 
                           data[:, 4][data[:, 4]>0], 
                           data[:, 0][data[:, 4]>0],
                           **kwargs)
    
    def plot(self, data, **kwargs):
        self.ax[1].plot(data[:, 1][data[:, 3]>0], 
                        data[:, 0][data[:, 3]>0], 
                        data[:, 3][data[:, 3]>0],
                        **kwargs)
        self.ax[0].plot(data[:, 1][data[:, 4]>0], 
                        data[:, 4][data[:, 4]>0], 
                        data[:, 0][data[:, 4]>0],
                        **kwargs)


def ternaryEndmembersFromCIPW(data, calcFeSpeciation=True, **kwargs):
    """
    Use the CIPW norm to calculate the endmembers for the basalt tetrahedron. The
    CIPW calculations are performed by pyrolite. If calculating the FeO and Fe2O3
    from a FeOT column, pyrolite may generate a warning which can be ignored.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, or dict
        The the compositions to project, in wt%.
    calcFeSpeciation : bool, default: True

    Returns
    -------
    numpy.Array [len(data), 5]
        The grouped endmember abundances, in the order olv, cpx, fsp, qtz, nph
    numpy.Array len(data)
        The unallocated mineral proportions.
    """
    grouping = {'quartz': 'qtz',
                'anorthite': 'fsp',
                'albite': 'fsp',
                'orthoclase': 'fsp',
                'nepheline': 'nph',
                'forsterite': None, # These are remapped olivine. Ignore from totals.
                'fayalite': None,
                'clinoferrosilite': None, # These are remapped diopside. Ignore from totals.
                'clinoenstatite': None,
                'ferrosilite': None, # These are remapped hypersthene. Ignore from totals.
                'enstatite': None,
                'wollastonite': 'cpx',
                'diopside': 'cpx',
                'hypersthene': {'olv': 0.5, 'qtz': 0.5},
                'olivine': 'olv'
                }
    
    majors = pd.DataFrame()
    for maj in ['SiO2', 'TiO2', 'Al2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']:
        if maj in data.columns:
            majors[maj] = data[maj]
        else:
            majors[maj] = 0.01
    
    if calcFeSpeciation and 'FeOT' in data.columns:
        majors['FeOT'] = data['FeOT']
    elif 'FeO' in data.columns and 'Fe2O3' in data.columns:
        majors['FeO'] = data['FeO']
        majors['Fe2O3'] = data['Fe2O3']
    else:
        raise ValueError("Need to provide either FeOT, or FeO and Fe2O3, or set calcFeSpeciation to True.")

    majors = majors.copy()
    cipw = CIPW_norm(majors)
    
    # Order: Olv, Cpx, Fsp, Qtz, Nph
    order = ['olv', 'cpx', 'fsp', 'qtz', 'nph']
    results = np.zeros([len(cipw), 5])

    not_allocated = np.zeros(len(cipw))
    
    for mineral in cipw.columns:
        if mineral in grouping:
            endmember = grouping[mineral]
            if isinstance(endmember, dict):
                for endm in endmember.keys():
                    results[:, order.index(endm)] += np.array(cipw[mineral]) * endmember[endm]
            elif endmember is not None:
                results[:, order.index(endmember)] += np.array(cipw[mineral])
        else:
            not_allocated += np.array(cipw[mineral])
    
    return results, not_allocated


def TernaryEndmembersFromSack87(data):
    """
    """

    oxidenames = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'Fe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    
    results = np.zeros([len(data), 5])

    for i, row in data.iterrows():
        # Extract the sample composition in the relevant chemical system
        wtpt = []
        
        for ox in oxidenames:
            if ox in row.keys():
                wtpt.append(row[ox])
            else:
                wtpt.append(0.0)

        # Convert to mole fractions
        molfracs = chem.wt_to_mol_oxide(wtpt, oxidenames)

        # Create a dictionary of mole fractions for easier access.
        mf = {}
        for j in range(len(oxidenames)):
            mf[oxidenames[j]] = molfracs[j]

        proj = np.zeros(5)

        # Order: Olv, Cpx, Fsp, Qtz, Nph

        proj[2] = ((mf['Al2O3'] + mf['Fe2O3'] - mf['TiO2'] - mf['Na2O'] - mf['K2O'])
                + (0.5*(mf['SiO2'] + 2*mf['TiO2']) 
                    - 0.25*(mf['Al2O3'] + mf['Fe2O3'] + mf['FeO'] + mf['MnO'] 
                            + mf['MgO'] + 3*mf['CaO'] + 3*mf['Na2O'] + 3*mf['K2O'])) )
        proj[1] = ((mf['CaO'] + mf['TiO2'] + mf['Na2O'] + mf['K2O']) 
                    - (mf['Al2O3'] + mf['Fe2O3']))
        proj[0] = ( 0.5 * (mf['FeO'] + mf['MgO'] + mf['MnO'] 
                            +  mf['Fe2O3'] + mf['Al2O3'] 
                            - mf['CaO'] - mf['Na2O'] - mf['K2O']) )
        
        proj[4] = (0.25 * (11 * mf['Na2O'] + 11 * mf['K2O'] + 3 * mf['CaO'] 
                            + mf['Al2O3'] + mf['Fe2O3'] + mf['FeO'] + mf['MnO'] 
                            + mf['MgO']) - 0.5 * (mf['SiO2'] + 2 * mf['TiO2']))
        if proj[4] < 0:
            proj[3] = -proj[4]
            proj[4] = 0
        else:
            proj[3] = 0
        
        results[i, :] = proj

    return results