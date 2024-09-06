from gui import run_gui
from helper_functions import unique_last_axis
from qarray import ChargeSensedDotArray, GateVoltageComposer, WhiteNoise, TelegraphNoise
import numpy as np

# defining the capacitance matrices
Cdd = np.array([0., 0.1], [0.1, 0.])  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = np.array([1., 0.2, 0.05], [0.2, 1., 0.05], )  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = np.array([0.02, 0.00])  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = np.array([0.06, 0.02, 1])  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating white noise model
white_noise = WhiteNoise(
    amplitude=1e-3
)

# creating telegraph noise model
telegraph_noise = TelegraphNoise(
    amplitude=1e-2,
    p01=1e-3,
    p10=1e-2,
)

# combining the noise models via addition
noise = white_noise + telegraph_noise

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=50,
    algorithm='default',
    implementation='python',
    noise_model=noise,
)
run_gui(model)
