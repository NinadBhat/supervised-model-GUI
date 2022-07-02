from turtle import right
import ipywidgets as widgets
from termcolor import colored

from IPython.display import display
from ipywidgets import VBox, HBox
from joblib import load
from tabulate import tabulate

import pandas as pd


PROCESSES = [
    "Artificial aged",
    "Naturally aged",
    "No Processing",
    "Solutionised",
    "Solutionised  + Artificially peak aged",
    "Solutionised + Artificially over aged",
    "Solutionised + Cold Worked + Naturally aged",
    "Solutionised + Naturally aged",
    "Strain Harderned (Hard)",
    "Strain hardened",
]

PROCESSES_ENCODING = {
    "as-cast or as-fabricated": "No Processing",
    "Annealed, Solutionised": "Solutionised",
    "H (soft)": "Strain hardened",
    "H (hard)": "Strain Harderned (Hard)",
    "T1": "Naturally aged",
    "T3 (incl. T3xx)": "Solutionised + Cold Worked + Naturally aged",
    "T4": "Solutionised + Naturally aged",
    "T5": "Artificial aged",
    "T6 (incl. T6xx)": "Solutionised  + Artificially peak aged",
    "T7 (incl. T7xx)": "Solutionised + Artificially over aged",
    "T8 (incl. T8xx)": "Solutionised + Artificially over aged",
}

PROCESSES_LIST = list(PROCESSES_ENCODING.keys())
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
    preprocess_ys = load("./models/preprocessor_ys.joblib")
    preprocess_ts = load("./models/preprocessor_ts.joblib")
    preprocess_elong = load("./models/preprocessor_elong.joblib")

    rf_ys = load("./models/rf_ys.joblib")
    rf_ts = load("./models/rf_ts.joblib")
    rf_elong = load("./models/rf_elong.joblib")
    preprocessed_input_ys = preprocess_ys.transform(pd.DataFrame(value_dict, index=[0]))
    preprocessed_input_ts = preprocess_ts.transform(pd.DataFrame(value_dict, index=[0]))
    preprocessed_input_elong = preprocess_elong.transform(
        pd.DataFrame(value_dict, index=[0])
    )

    ys = rf_ys.predict(preprocessed_input_ys)
    ts = rf_ts.predict(preprocessed_input_ts)
    elong = rf_elong.predict(preprocessed_input_elong)

    print()
    print()
    print("Yield Strength:", colored(f"{ys[0]:.2f} MPa", "green"))
    print("Tensile Strength:", colored(f"{ts[0]:.2f} MPa", "green"))
    print("Elongation:", colored(f"{elong[0]:.2f} %", "green"))


def build_gui():
    def print_properties(b):
        with output:
            left_conc = [item.value / 100 for item in concentration_widget_left]
            right_conc = [item.value / 100 for item in concentration_widget_right]
            process = PROCESSES_ENCODING[start_widget_list[0].value]
            values = [process] + left_conc + right_conc
            values.insert(2, 1 - sum(values[1:]))
            print("Calculating Aluminium as balance of other alloying element")
            input_dict = dict(zip(FEATURE_COLUMNS, values))
            print()
            print("Concentration of Elements:")
            print_conc_list = []
            for element, concentration in input_dict.items():
                if element != "Processing" and concentration > 1e-06:
                    print_conc_list.append([element, concentration * 100])
            print(
                tabulate(print_conc_list, headers=["Element", "Concentration (wt %)"])
            )

            calculate_properties(input_dict)

    start_widget_list = []
    concentration_widget_left = []
    concentration_widget_right = []
    end_widget_list = []

    process_type = widgets.Dropdown(
        options=PROCESSES_LIST,
        value="as-cast or as-fabricated",
        description="Process:",
        disabled=False,
    )
    start_widget_list.append(process_type)

    concentration_copy = CONCENTRATIONS[:]
    concentration_copy.remove("Al")
    for index, element in enumerate(concentration_copy):
        widget_conc = widgets.BoundedFloatText(
            value=0,
            min=0,
            max=100.0,
            step=0.1,
            description=f"{element}:",
            disabled=False,
        )
        if index <= len(concentration_copy) // 2:
            concentration_widget_left.append(widget_conc)
        else:
            concentration_widget_right.append(widget_conc)

    button = widgets.Button(description="Calculate Properties")
    output = widgets.Output()

    button.on_click(print_properties)

    end_widget_list.append(button)

    end_widget_list.append(output)
    left_widgets = VBox(concentration_widget_left)
    right_widgets = VBox(concentration_widget_right)
    display(VBox(start_widget_list))
    display(HBox([left_widgets, right_widgets]))
    display(VBox(end_widget_list))
