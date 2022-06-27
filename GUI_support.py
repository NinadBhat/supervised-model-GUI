import ipywidgets as widgets

from IPython.display import display
from ipywidgets import VBox
from joblib import load

import pandas as pd


PROCESSES = ['Artificial aged', 'Naturally aged',
       'No Processing', 'Processing_Solutionised',
       'Solutionised  + Artificially peak aged',
       'Solutionised + Artificially over aged',
       'Solutionised + Cold Worked + Naturally aged',
       'Solutionised + Naturally aged',
       'Strain Harderned (Hard)', 
       'Strain hardened']


PROCESSING = ["Processing"]

CONCENTRATIONS = [
    "Ag",
    "Al",
    "B",
    "Be",
    "Cd",
    "Co",
    "Cr",
    "Cu",
    "Er",
    "Eu",
    "Fe",
    "Ga",
    "Li",
    "Mg",
    "Mn",
    "Ni",
    "Pb",
    "Sc",
    "Si",
    "Sn",
    "Ti",
    "V",
    "Zn",
    "Zr",
]


FEATURE_COLUMNS = PROCESSING + CONCENTRATIONS

def calculate_properties(value_dict):
    preprocess_ys = load('../models/preprocessor_ys.joblib') 
    preprocess_ts = load('../models/preprocessor_ts.joblib') 
    preprocess_elong = load('../models/preprocessor_elong.joblib') 

    rf_ys = load('../models/rf_ys.joblib')
    rf_ts = load('../models/rf_ts.joblib')
    rf_elong = load('../models/rf_elong.joblib')
    preprocessed_input_ys = preprocess_ys.transform(pd.DataFrame(value_dict, index=[0]))
    preprocessed_input_ts = preprocess_ts.transform(pd.DataFrame(value_dict, index=[0]))
    preprocessed_input_elong = preprocess_elong.transform(pd.DataFrame(value_dict, index=[0]))
   
    ys = rf_ys.predict(preprocessed_input_ys)
    ts = rf_ts.predict(preprocessed_input_ts)
    elong = rf_elong.predict(preprocessed_input_elong)

    print(F"Yield Strength: {ys[0]:.2f} MPa")
    print(F"Tensile Strength: {ts[0]:.2f} MPa")
    print(F"Elongation: {elong[0]:.2f}")

def build_gui():
    
    def print_properties(b):    
        with output:
            values = [item.value for item in widget_list[:-2]]
            values[1:] = [concentrations/100 for concentrations in values[1:]]
            input_dict = dict(zip(FEATURE_COLUMNS, values))
            calculate_properties(input_dict)

    widget_list = []
    process_type = widgets.Dropdown(
        options=PROCESSES,
        value="Artificial aged",
        description="Process:",
        disabled=False,
    )
    widget_list.append(process_type)
    for element in CONCENTRATIONS:
        widget_list.append(
            widgets.BoundedFloatText(
                value=0,
                min=0,
                max=100.0,
                step=0.1,
                description=f"{element}:",
                disabled=False,
            )
        )

    button = widgets.Button(description="Calculate Properties")
    output = widgets.Output()

    button.on_click(print_properties)

    widget_list.append(button)

    widget_list.append(output)
    
    display(VBox(widget_list))
