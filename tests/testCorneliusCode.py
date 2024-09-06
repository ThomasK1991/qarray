import numpy as np
from matplotlib import pyplot as plt
import random
from qarray import ChargeSensedDotArray, GateVoltageComposer, dot_occupation_changes, WhiteNoise, TelegraphNoise, LatchingModel, PSBLatchingModel
from tqdm import tqdm
import time

'''
This script generates simulated measurement samples, used for the training of the line extraction CNN
'''
def reformat_data(data):
    image = np.squeeze(data)  # reshaping dimensions to be (y_res, x_res)
    image = np.flipud(image)  # flipping vertically to map increasing row index with smaller y
    return image


sample_num = 100
qarray_data = []
qarray_lines = []
for s in tqdm(range(sample_num)):
    # define random parameters
    Cgd11 = random.uniform(0.6, 0.9)
    Cgd22 = random.uniform(0.6, 0.9)
    Cdd12 = random.uniform(0.02, 0.3)  # interdot length
    Cgd12 = random.uniform(0., 0.35)  # slantiness of transitions
    Cgd21 = random.uniform(0., 0.35)  # slantiness of transitions
    Cgd13 = random.uniform(0.001, 0.5)  # contrast
    Cgd23 = random.uniform(0.001, 0.5)  # contrast
    Cds1 = random.uniform(0.01, 0.25)  # visibility of transition lines and interdot (if different)
    Cds2 = random.uniform(0.01, 0.25)  # visibility of transition lines and interdot (if different)
    Cgs1 = random.uniform(0.03, 0.07)  # sensor peak periodicity
    Cgs2 = random.uniform(0.03, 0.07)  # sensor peak periodicity

    # defining the capacitance matrices
    Cdd = [[1, Cdd12],
           [Cdd12, 1]]  # an (n_dot, n_dot) dot to dot
    Cgd = [[Cgd11, Cgd12, Cgd13],  # charging energy (on-diagonal)
           [Cgd21, Cgd22, Cgd23]]  # an (n_dot, n_gate) # dots to gates
    Cds = [[Cds1, Cds2]]  # an (n_sensor, n_dot)  # dots to sensor dot
    # this can be used as a hack to get on a different side of a coulomb peak
    Cgs = [[Cgs1, Cgs2, 1]]  # an (n_sensor, n_gate) # gates to sensor gate

    coulomb_peak_width = random.uniform(0.1, 0.5)  # coulomb peak width

    coulomb_peak_width = random.uniform(0.1, 0.5)  # coulomb peak width
    vx_occ = np.random.normal(0.5, 0.06, 1)[0]  # dot occupation
    vy_occ = np.random.normal(0.5, 0.06, 1)[0]  # dot occupation
    s_occ = random.uniform(0.2, 1.0)  # sensor position

    white_noise_amp = random.uniform(0.005, 0.02)
    telegrapher_amp = random.uniform(0.002, 0.05)
    p01 = random.uniform(0.00001, 0.005)
    p10 = random.uniform(1, 100) * p01

    white_noise = WhiteNoise(
        amplitude=white_noise_amp
    )

    telegraph_noise = TelegraphNoise(
        amplitude=telegrapher_amp,
        p01=p01,
        p10=p10
    )

    noise = white_noise + telegraph_noise

    broaden_or_latch = random.randint(0, 10)  # include more latching examples
    if broaden_or_latch > 0:
        T = random.uniform(0.0, 250)  # thermal broadening
        lead_latching_model = None
    else:
        T = 0.0
        p_lead1 = random.uniform(0.08, 0.6)
        p_lead2 = random.uniform(0.08, 0.6)
        p_inter12 = random.uniform(0.08, 0.6)
        p_inter21 = random.uniform(0.08, 0.6)
        lead_latching_model = LatchingModel(
            n_dots=2,
            p_leads=[p_lead1, p_lead2],
            p_inter=[
                [0., p_inter12],
                [p_inter21, 0.],
            ]
        )

    # creating the model
    model_s = ChargeSensedDotArray(
        Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
        coulomb_peak_width=coulomb_peak_width, T=T,
        noise_model=noise, latching_model=lead_latching_model,
        algorithm='default', implementation='jax'
    )

    model_t = ChargeSensedDotArray(
        Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
        coulomb_peak_width=0., T=0., noise_model=None,
        algorithm='default', implementation='jax'
    )

    # model = ChargeSensedDotArray(
    #     Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    #     coulomb_peak_width=coulomb_peak_width, noise=noise,
    #     algorithm='thresholded', threshold=0.15, implementation='rust', T=T
    # )

    # model = ChargeSensedDotArray(
    #     Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    #     coulomb_peak_width=coulomb_peak_width, noise=noise,
    #     algorithm='brute_force', implementation='jax', max_charge_carriers=1,
    #     batch_size=400, T=T
    # )

    #doesn't accept j or J, r


    # creating the voltage composer of correct shape
    voltage_composer = GateVoltageComposer(n_gate=model_s.n_gate, n_dot=3)  # n_dot necessary if doing virtual gates


    # # using the (virtual) dot voltage composer to create the dot voltage array for the 2d sweep
    # voltage_composer.virtual_gate_matrix = np.array([
    #     [1, -0.2, -0.1],
    #     [-0.2, 1, -0.2],
    #     [0, 0, 1]
    # ])
    # voltage_composer.virtual_gate_origin = model.optimal_Vg(np.array([vx_occ, vy_occ]))
    # vg = voltage_composer.do2d_virtual(0, vy_min, vx_max, 200, 1, vy_min, vy_max, 200)


    # defining 2D voltage sweep ranges
    vx_min, vx_max = -.49, .49
    vy_min, vy_max = -.49, .49
    x_res, y_res = 100, 100

    vg = voltage_composer.do2d(0, vy_min, vx_max, x_res, 1, vy_min, vy_max, y_res)
    vg += model_s.optimal_Vg(np.array([vx_occ, vy_occ, s_occ]))

    # looping over the functions, computing the ground state
    z, _ = model_s.charge_sensor_open(vg=vg)  # charge sensing data
    z = (z - np.min(z)) / (np.max(z) - np.min(z))  # normalisation
    z = reformat_data(z)

    n = model_t.ground_state_open(vg=vg)  # noiseless ground state at T=0
    pad_width = ((1, 0), (0, 1), (0, 0))
    n_padded = np.pad(n, pad_width, mode='edge')  # pad top and right to ensure (y_res, x_res) is kept
    z_change = dot_occupation_changes(n_padded)  # transition line extraction
    z_change = reformat_data(z_change)
    z_change = z_change.astype(float)  # convert from boolean to float

    qarray_data.append(z)
    qarray_lines.append(z_change)


# creating the figure and axes
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 3)
axes[0].imshow(z)
axes[1].imshow(z_change, cmap='binary')
plt.show()


# np.save('training_data/qarray_data.npy', qarray_data)
# np.save('training_data/qarray_lines.npy', qarray_lines)