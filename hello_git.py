import requests  # Imports library to make requests to a repository

url = ('https://raw.githubusercontent.com/JulianDPastrana/signal_analysis/main/seniales_sep.py')  # Repository path
r = requests.get(url)  # The application is made

import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

with open('seniales_sep.py', 'w') as f:  # The path file is read
   f.write(r.text)  # A new document is created in which the code contained in the file is copied.

from seniales_sep import signal_generation  # The function that generates the signals is imported

data = signal_generation()  # The signs are contained in a dictionary
voltage_1, current_1 = data["Node 1"]
voltage_2, current_2 = data["Node 2"]
voltage_3, current_3 = data["Node 3"]

print(voltage_1)