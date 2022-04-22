"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Derivatech.                                                                                -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: diegolazareno                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/diegolazareno/ProyectoDerivatech                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# Librerías requeridas
import functions
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, interactive

def sectorButton():
    infoNasdaq = pd.read_csv("files/nasdaq_screener_1650511730482.csv")
    sectors = list(set(infoNasdaq["Sector"]))
    SectorButton = widgets.Dropdown(options = sectors[1:])
    
    return SectorButton, infoNasdaq

def companyButton(SectorButton, infoNasdaq):
    from functions import userFunction
    
    Stock = interactive(userFunction, 
                     Company = widgets.Combobox(placeholder = 'Name', options = list(infoNasdaq[infoNasdaq["Sector"] == SectorButton.value]["Name"]),
                                               description = 'Company:', ensure_option = True, disabled = False))
    return Stock