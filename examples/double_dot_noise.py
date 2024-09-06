import numpy as np
from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer, WhiteNoise, TelegraphNoise

# defining the capacitance matrices
Cdd = [[0, 0.6], [0.6, 0]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[0.7, 0.2, 0.001], [0.1, 0.7, 0.001], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.1, 0.1]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.05, 0.02, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating white noise model
white_noise = WhiteNoise(amplitude=1e-2)

# creating telegraph noise model
telegraph_noise = TelegraphNoise(p01=5e-4, p10=5e-3, amplitude=1e-2)

# combining the noise models via addition
noise = white_noise + telegraph_noise

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.1, T=100,

    noise_model=noise,
)

# creating the voltage composer
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = model.gate_voltage_composer.do2d('P1', vy_min, vx_max, 100, 'P2', vy_min, vy_max, 100)
vg += model.optimal_Vg([0.5, 0.5, 0.6])

# creating the figure and axes
np.random.seed(0)
z, n = model.charge_sensor_open(vg)
# n_latched = add_latching_open(n, 0.05, 0.01)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(15, 5)

z_grad = np.abs(np.gradient(z, axis=1))

ax[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')

ax[1].imshow(z_grad, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
             interpolation='none')

z_latched = (n * np.array([0.9, 1.1])[np.newaxis, np.newaxis, :]).sum(axis=-1)

ax[2].imshow(z_latched, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
             cmap='Greys',
             interpolation='none')

if __name__ == '__main__':
    plt.show()