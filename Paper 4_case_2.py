#case study of transient and creep strain

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import math
from numba import jit

# Steel model
@jit
def steel_fire(fsy, Es, strain, T_n):
    # all the units are in Pa

    fsp = 1 * fsy

    # reduction of fsy, fsp, Es
    if 0 <= T_n <= 100:
        T_fsy = 1
        T_fsp = 1
        T_Es = 1
    elif 100 < T_n <= 200:
        T_fsy = 1
        T_fsp = 1 - 0.19 / 100 * (T_n - 100)
        T_Es = 1 - 0.1 / 100 * (T_n - 100)
    elif 200 < T_n <= 300:
        T_fsy = 1
        T_fsp = 0.81 - 0.2 / 100 * (T_n - 200)
        T_Es = 0.9 - 0.1 / 100 * (T_n - 200)
    elif 300 < T_n <= 400:
        T_fsy = 1
        T_fsp = 0.61 - 0.19 / 100 * (T_n - 300)
        T_Es = 0.8 - 0.1 / 100 * (T_n - 300)
    elif 400 < T_n <= 500:
        T_fsy = 1 - 0.22 / 100 * (T_n - 400)
        T_fsp = 0.42 - 0.06 / 100 * (T_n - 400)
        T_Es = 0.7 - 0.1 / 100 * (T_n - 400)
    elif 500 < T_n <= 600:
        T_fsy = 0.78 - 0.31 / 100 * (T_n - 500)
        T_fsp = 0.36 - 0.18 / 100 * (T_n - 500)
        T_Es = 0.6 - 0.29 / 100 * (T_n - 500)
    elif 600 < T_n <= 700:
        T_fsy = 0.47 - 0.24 / 100 * (T_n - 600)
        T_fsp = 0.18 - 0.11 / 100 * (T_n - 600)
        T_Es = 0.31 - 0.18 / 100 * (T_n - 600)
    elif 700 < T_n <= 800:
        T_fsy = 0.23 - 0.12 / 100 * (T_n - 700)
        T_fsp = 0.07 - 0.02 / 100 * (T_n - 700)
        T_Es = 0.13 - 0.04 / 100 * (T_n - 700)
    elif 800 < T_n <= 900:
        T_fsy = 0.11 - 0.05 / 100 * (T_n - 800)
        T_fsp = 0.05 - 0.01 / 100 * (T_n - 800)
        T_Es = 0.09 - 0.02 / 100 * (T_n - 800)
    elif 900 < T_n <= 1000:
        T_fsy = 0.06 - 0.02 / 100 * (T_n - 900)
        T_fsp = 0.04 - 0.02 / 100 * (T_n - 900)
        T_Es = 0.07 - 0.03 / 100 * (T_n - 900)
    elif 1000 < T_n <= 1100:
        T_fsy = 0.04 - 0.02 / 100 * (T_n - 1000)
        T_fsp = 0.02 - 0.01 / 100 * (T_n - 1000)
        T_Es = 0.04 - 0.02 / 100 * (T_n - 1000)
    else:
        T_fsy = 0
        T_fsp = 0
        T_Es = 0

    fsy_T = fsy * T_fsy
    fsp_T = fsp * T_fsp
    Es_T = Es * T_Es
    e_sp = fsp_T / Es_T
    e_sy = 0.02
    e_st = 0.15
    e_su = 0.20

    c_p = (fsy_T - fsp_T) ** 2 / ((e_sy - e_sp) * Es_T - 2 * (fsy_T - fsp_T))
    a_p = ((e_sy - e_sp) * (e_sy - e_sp + c_p / Es_T)) ** 0.5
    b_p = (c_p * (e_sy - e_sp) * Es_T + c_p ** 2) ** 0.5

    if strain >= 0:
        e_steel = strain
    else:
        e_steel = -1 * strain

    if 0 <= e_steel <= e_sp:
        stress = e_steel * Es * T_Es
    elif e_sp <= e_steel <= e_sy:
        stress = fsp_T - c_p + (b_p / a_p) * (a_p ** 2 - (e_sy - e_steel) ** 2) ** 0.5
    elif e_sy <= e_steel <= e_st:
        stress = fsy_T
    elif e_st <= e_steel <= e_su:
        stress = fsy_T * (1 - (e_steel - e_st) / (e_su - e_st))
    else:
        stress = 0

    if strain >= 0:
        s_stress = stress
    else:
        s_stress = -1 * stress

    return s_stress


@jit
def steel_fire_coe(T_n):
    # reduction of fsy, fsp, Es
    if 0 <= T_n <= 100:
        T_fsy = 1

    elif 100 < T_n <= 200:
        T_fsy = 1

    elif 200 < T_n <= 300:
        T_fsy = 1

    elif 300 < T_n <= 400:
        T_fsy = 1

    elif 400 < T_n <= 500:
        T_fsy = 1 - 0.22 / 100 * (T_n - 400)

    elif 500 < T_n <= 600:
        T_fsy = 0.78 - 0.31 / 100 * (T_n - 500)

    elif 600 < T_n <= 700:
        T_fsy = 0.47 - 0.24 / 100 * (T_n - 600)

    elif 700 < T_n <= 800:
        T_fsy = 0.23 - 0.12 / 100 * (T_n - 700)

    elif 800 < T_n <= 900:
        T_fsy = 0.11 - 0.05 / 100 * (T_n - 800)

    elif 900 < T_n <= 1000:
        T_fsy = 0.06 - 0.02 / 100 * (T_n - 900)

    elif 1000 < T_n <= 1100:
        T_fsy = 0.04 - 0.02 / 100 * (T_n - 1000)

    else:
        T_fsy = 0

    return T_fsy


@jit
def steel_exp(T_n):
    if 20 <= T_n <= 750:
        therm_strain = -2.416 * 10 ** (-4) + 1.2 * 10 ** (-5) * T_n + 0.4 * 10 ** (-8) * T_n ** 2
    elif 750 <= T_n <= 860:
        therm_strain = 11 * 10 ** (-3)
    elif 860 <= T_n:
        therm_strain = -6.2 * 10 ** (-3) + 2 * 10 ** (-5) * T_n
    else:
        therm_strain = 0

    return therm_strain


# Concrete in tension under fire
@jit
def conc_fire_ten(e_ten, T_n):
    # 未完待续
    stress = e_ten * T_n

    return stress


@jit
def conc_ten(T_n):
    # 未完待续
    coeff = 0.5

    return coeff


# Concrete model
'''
# Kodur model

@jit
def conc_fire(e_comp, fc, T_n):
    # HSC Kodur 2004
    H = 2.28 - 0.012 * fc / 10 ** 6
    e_max = 0.0018 + (6.7 * fc / 10 ** 6 + 6.0 * T_n + 0.03 * T_n ** 2) * 10 ** (-6)

    if T_n < 100:
        fc_T = fc * (1 - 0.003125 * (T_n - 20)) / 10 ** 6
    elif 100 <= T_n <= 400:
        fc_T = 0.75 * fc / 10 ** 6
    else:
        fc_T = fc * (1.33 - 0.00145 * T_n) / 10 ** 6

    if e_comp <= e_max:
        stress = fc_T * (1 - (((e_max - e_comp) / e_max)) ** H) * 10 ** 6
        if stress < 0:
            stress = 0
    else:
        stress = fc_T * (1 - (30 * (e_comp - e_max) / (130 - fc / 10 ** 6) / e_max) ** 2) * 10 ** 6
        if stress < 0:
            stress = 0

    return stress

@jit
def conc_fy(T_n):
    if T_n < 100:
        fc_T = 1 - 0.003125 * (T_n - 20)
    elif 100 <= T_n <= 400:
        fc_T = 0.75
    else:
        fc_T = 1.33 - 0.00145 * T_n

    return fc_T

@jit
def con_exp(T_n):
    # Kodur 2004 HSC
    therm_strain = (0.004 * (T_n ** 2 - 400) + 6 * (T_n - 20)) / 1000000
    return therm_strain   


@jit
def tran_e(stress, T_n, fc):

    tran_strain = 0

    return tran_strain

'''


@jit
def conc_fire(e_comp, fc, T_n):
    # EC2 Part1-2; 3.2.2; Siliceous

    # strength reduction
    if 20 <= T_n <= 100:
        T_fc = 1
        e_c1 = 0.0025 + 0.0015 / 80 * (T_n - 20)
        e_cu1 = 0.02 + 0.0025 / 80 * (T_n - 20)
    elif 100 < T_n <= 200:
        T_fc = 1 - 0.05 / 100 * (T_n - 100)
        e_c1 = 0.0040 + 0.0015 / 100 * (T_n - 100)
        e_cu1 = 0.0225 + 0.0025 / 100 * (T_n - 100)
    elif 200 < T_n <= 300:
        T_fc = 0.95 - 0.1 / 100 * (T_n - 200)
        e_c1 = 0.0055 + 0.0015 / 100 * (T_n - 200)
        e_cu1 = 0.0250 + 0.0025 / 100 * (T_n - 200)
    elif 300 < T_n <= 400:
        T_fc = 0.85 - 0.1 / 100 * (T_n - 300)
        e_c1 = 0.0070 + 0.003 / 100 * (T_n - 300)
        e_cu1 = 0.0275 + 0.0025 / 100 * (T_n - 300)
    elif 400 < T_n <= 500:
        T_fc = 0.75 - 0.15 / 100 * (T_n - 400)
        e_c1 = 0.01 + 0.0050 / 100 * (T_n - 400)
        e_cu1 = 0.03 + 0.0025 / 100 * (T_n - 400)
    elif 500 < T_n <= 600:
        T_fc = 0.6 - 0.15 / 100 * (T_n - 500)
        e_c1 = 0.015 + 0.01 / 100 * (T_n - 500)
        e_cu1 = 0.0325 + 0.0025 / 100 * (T_n - 500)
    elif 600 < T_n <= 700:
        T_fc = 0.45 - 0.15 / 100 * (T_n - 600)
        e_c1 = 0.025
        e_cu1 = 0.0350 + 0.0025 / 100 * (T_n - 600)
    elif 700 < T_n <= 800:
        T_fc = 0.3 - 0.15 / 100 * (T_n - 700)
        e_c1 = 0.025
        e_cu1 = 0.0375 + 0.0025 / 100 * (T_n - 700)
    elif 800 < T_n <= 900:
        T_fc = 0.15 - 0.07 / 100 * (T_n - 800)
        e_c1 = 0.025
        e_cu1 = 0.04 + 0.0025 / 100 * (T_n - 800)
    elif 900 < T_n <= 1000:
        T_fc = 0.08 - 0.04 / 100 * (T_n - 900)
        e_c1 = 0.025
        e_cu1 = 0.0425 + 0.0025 / 100 * (T_n - 900)
    elif 1000 < T_n <= 1100:
        T_fc = 0.04 - 0.03 / 100 * (T_n - 1000)
        e_c1 = 0.025
        e_cu1 = 0.045 + 0.0025 / 100 * (T_n - 1000)
    else:
        T_fc = 0
        e_c1 = 0.025
        e_cu1 = 0.045 + 0.0025 / 100 * (T_n - 1000)

    if 0 <= e_comp <= e_c1:
        stress = 3 * e_comp * fc * T_fc / e_c1 / (2 + (e_comp / e_c1) ** 3)
    elif e_c1 < e_comp <= e_cu1:
        stress = fc * T_fc * (e_cu1 - e_comp) / (e_cu1 - e_c1)
    else:
        stress = 0

    return stress


@jit
def conc_fy(T_n):
    # EC2 Part1-2; 3.2.2; Siliceous

    # strength reduction
    if 20 <= T_n <= 100:
        T_fc = 1
    elif 100 < T_n <= 200:
        T_fc = 1 - 0.05 / 100 * (T_n - 100)
    elif 200 < T_n <= 300:
        T_fc = 0.95 - 0.1 / 100 * (T_n - 200)
    elif 300 < T_n <= 400:
        T_fc = 0.85 - 0.1 / 100 * (T_n - 300)
    elif 400 < T_n <= 500:
        T_fc = 0.75 - 0.15 / 100 * (T_n - 400)
    elif 500 < T_n <= 600:
        T_fc = 0.6 - 0.15 / 100 * (T_n - 500)
    elif 600 < T_n <= 700:
        T_fc = 0.45 - 0.15 / 100 * (T_n - 600)
    elif 700 < T_n <= 800:
        T_fc = 0.3 - 0.15 / 100 * (T_n - 700)
    elif 800 < T_n <= 900:
        T_fc = 0.15 - 0.07 / 100 * (T_n - 800)
    elif 900 < T_n <= 1000:
        T_fc = 0.08 - 0.04 / 100 * (T_n - 900)
    elif 1000 < T_n <= 1100:
        T_fc = 0.04 - 0.03 / 100 * (T_n - 1000)
    else:
        T_fc = 0

    return T_fc


@jit
def con_exp(T_n):
    if 20 <= T_n <= 700:
        therm_strain = -1.8 * 10 ** (-4) + 9 * 10 ** (-6) * T_n + 2.3 * 10 ** (-11) * T_n ** 3
    elif 700 <= T_n:
        therm_strain = 14 * 10 ** (-3)
    else:
        therm_strain = 0

    return therm_strain


@jit
def tran_e(stress, T_n, fc):

    tran_strain = 0

    return tran_strain


# EC2

@jit
def cre_e(stress, T_n, fc, time):
    # creep strain of concrete
    # time is in minutes

    creep_strain = 0

    return creep_strain

@jit
def conc_E(fc, T_n):
    Ec = 22 * (fc / 10 ** 6 / 10) ** 0.3 * 10 ** 9
    if 20 <= T_n <= 700:
        cof = 1.025641 - 0.001282 * T_n
    else:
        cof = 0.01
    Ec_T = Ec * cof

    return Ec_T


@jit
def sec_mod(strain, stress):
    if strain <= 0:
        modulus = 0
    else:
        modulus = stress / strain

    return modulus


@jit
def solver(time, dt, p_a, h_0, s, e_0, height, width, cover, dx, D_sr, N_com_bar, N_ten_bar, fys, Es, fc, fct, Pa, ecc,
           length, BC):
    Nx = int(round(width / float(dx)))
    Ny = int(round(height / float(dx)))
    x = np.linspace(0, width, Nx + 1)
    dx = x[1] - x[0]
    Nt = int(round(time / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dt = t[1] - t[0]

    loc_rebar = int(round(cover / float(dx)))

    # heat transfer arrays
    T = np.zeros(shape=(Nx + 1, Ny + 1))
    T_n = np.zeros(shape=(Nx + 1, Ny + 1))
    k_n = np.zeros(shape=(Nx + 1, Ny + 1))
    c_n = np.zeros(shape=(Nx + 1, Ny + 1))
    p_n = np.zeros(shape=(Nx + 1, Ny + 1))
    F_n = np.zeros(shape=(Nx + 1, Ny + 1))

    # array for external strain limit
    size_strain = 150
    strain_limit = np.zeros(size_strain + 1)
    strain_start = 0.00008  # at least start from 0.0001; cannot be larger than this

    strain_neg = 0

    app_loading = np.zeros(size_strain + 1)
    deflection = np.zeros(size_strain + 1)
    loading = np.zeros(size_strain + 1)

    con_accuracy = 0.8 * 10 ** 6
    con_incre = 0.4 * 10 ** 6

    # heat transfer analysis
    for i in range(0, Nx + 1):  # intial condition
        for j in range(0, Ny + 1):
            T_n[i][j] = 30

    for n in range(0, Nt + 1):
        for i in range(0, Nx + 1):
            for j in range(0, Ny + 1):

                if 20 <= T_n[i][j] <= 1200:
                    k_n[i][j] = (1.36 - 0.136 * (T_n[i][j] / 100) + 0.0057 * (T_n[i][j] / 100) ** 2)     #NSC
                    # k_n[i][j] = (2 - 0.2451 * (T_n[i][j] / 100) + 0.0107 * (T_n[i][j] / 100) ** 2)  # HSC
                else:
                    k_n[i][j] = 0

                if 20 <= T_n[i][j] <= 100:
                    c_n[i][j] = 900
                elif 100 < T_n[i][j] <= 115:
                    c_n[i][j] = 2770
                elif 115 < T_n[i][j] <= 200:
                    c_n[i][j] = 2770 - 1770 * (T_n[i][j] - 115) / 85
                elif 200 < T_n[i][j] <= 400:
                    c_n[i][j] = 1000 + (T_n[i][j] - 200) / 2
                elif 400 < T_n[i][j] <= 1200:
                    c_n[i][j] = 1100
                else:
                    c_n[i][j] = 1100

                if 20 <= T_n[i][j] <= 115:
                    p_n[i][j] = p_a
                elif 115 < T_n[i][j] <= 200:
                    p_n[i][j] = p_a * (1 - 0.02 * (T_n[i][j] - 115) / 85)
                elif 200 < T_n[i][j] <= 400:
                    p_n[i][j] = p_a * (0.98 - 0.03 * (T_n[i][j] - 200) / 200)
                elif 400 < T_n[i][j] <= 1200:
                    p_n[i][j] = p_a * (0.95 - 0.07 * (T_n[i][j] - 400) / 800)
                else:
                    p_n[i][j] = 2150

        Tf_n = 20 + 345 * np.log10(8 * n * dt / 60 + 1)

        # fire curve of my test
        # if n < 1200:
        #     Tf_n = 20 + np.log10(1 * n / 60 + 1) * 450
        # elif n < 3300:
        #     Tf_n = 625 + 10.5 * (n / 60 - 20)
        # else:
        #     Tf_n = 992.5 + 5 * (n / 60 - 55)

        Ta_n = 20

        T_n[0][0] = Tf_n
        T_n[0][Ny] = Tf_n
        T_n[Nx][0] = Tf_n
        T_n[Nx][Ny] = Tf_n

        for i in range(1, Nx):
            F_n[i][0] = dt / (2 * p_n[i][0] * c_n[i][0] * dx ** 2)
            F_n[i][Ny] = dt / (2 * p_n[i][Ny] * c_n[i][Ny] * dx ** 2)

            T[i][0] = T_n[i][0] + F_n[i][0] * ((3 * k_n[i][0] + k_n[i][1]) * (T_n[i][1] - T_n[i][0]) +
                                               4 * dx * h_0 * (Tf_n - T_n[i][0]) + 4 * dx * s * e_0 * (
                                                       (Tf_n + 273.15) ** 4 - (T_n[i][0] + 273.15) ** 4))
            T[i][Ny] = T_n[i][Ny] + F_n[i][Ny] * ((3 * k_n[i][Ny] + k_n[i][Ny - 1]) * (T_n[i][Ny - 1] - T_n[i][Ny]) +
                                                  4 * dx * h_0 * (Tf_n - T_n[i][Ny]) + 4 * dx * s * e_0 * (
                                                          (Tf_n + 273.15) ** 4 - (T_n[i][Ny] + 273.15) ** 4))

        for j in range(1, Ny):
            F_n[0][j] = dt / (2 * p_n[0][j] * c_n[0][j] * dx ** 2)
            F_n[Nx][j] = dt / (2 * p_n[Nx][j] * c_n[Nx][j] * dx ** 2)
            T[0][j] = T_n[0][j] + F_n[0][j] * ((3 * k_n[0][j] + k_n[1][j]) * (T_n[1][j] - T_n[0][j]) +
                                               4 * dx * h_0 * (Tf_n - T_n[0][j]) + 4 * dx * s * e_0 * (
                                                       (Tf_n + 273.15) ** 4 - (T_n[0][j] + 273.15) ** 4))
            T[Nx][j] = T_n[Nx][j] + F_n[Nx][j] * ((3 * k_n[Nx][j] + k_n[Nx - 1][j]) * (T_n[Nx - 1][j] - T_n[Nx][j]) +
                                                  4 * dx * h_0 * (Tf_n - T_n[Nx][j]) + 4 * dx * s * e_0 * (
                                                          (Tf_n + 273.15) ** 4 - (T_n[Nx][j] + 273.15) ** 4))

        for i in range(1, Nx):  # start from 0 and end Nx, as boundary condition is considered next. Not include Nx
            for j in range(1, Ny):
                F_n[i][j] = dt / (2 * p_n[i][j] * c_n[i][j] * dx ** 2)
                T[i][j] = T_n[i][j] + F_n[i][j] * (
                        (k_n[i + 1][j] + k_n[i][j]) * (T_n[i + 1][j] - T_n[i][j]) - (k_n[i][j] + k_n[i - 1][j]) * (
                        T_n[i][j] - T_n[i - 1][j]) + \
                        (k_n[i][j + 1] + k_n[i][j]) * (T_n[i][j + 1] - T_n[i][j]) - (k_n[i][j] + k_n[i][j - 1]) * (
                                T_n[i][j] - T_n[i][j - 1]))

        T_n, T = T, T_n
        # complete heat transfer analysis at this time

        interval = 300  # display interval; in seconds
        upp_lim = int(round(time / interval)) + 1

        for q in range(0, upp_lim):
            if n == interval * q:
                # print("time =", q * interval / 60)

                for k in range(0, size_strain + 1):
                    strain_incre = 0.00008
                    strain_limit[k] = strain_start + (k - strain_neg) * strain_incre

                    app_loading[k] = Pa
                    t_force_c_c = np.zeros(Ny + 2)
                    t_moment_c_c = np.zeros(Ny + 2)
                    t_buckling_c_c = np.zeros(Ny + 2)
                    t_force_c_t = np.zeros(Ny + 2)
                    t_moment_c_t = np.zeros(Ny + 2)
                    total_force = np.zeros(Ny + 2)
                    total_moment = np.zeros(Ny + 2)
                    total_buckling = np.zeros(Ny + 2)
                    int_force_1 = np.zeros(Ny + 2)

                    mid_deflec = np.zeros(Ny + 2)
                    ecc_cro = np.zeros(Ny + 2)
                    load_stability = np.zeros(Ny + 2)
                    load_failure = np.zeros(Ny + 2)

                    if strain_limit[k] > 0:

                        for i in range(1, Ny + 1):
                            l_NA = i * dx  # location of neutral axis
                            rem = height - l_NA  # height of tension zone
                            m = int(round(rem / dx))

                            # concrete in compression array
                            e_c_comp_tot = np.zeros(i + 1)  # layer
                            con_expan = np.zeros(shape=(Nx + 1, i + 1))
                            con_tran = np.zeros(shape=(Nx + 1, i + 1))
                            con_creep = np.zeros(shape=(Nx + 1, i + 1))
                            c_m_strain = np.zeros(shape=(Nx + 1, i + 1))
                            stress_c_c = np.zeros(shape=(Nx + 1, i + 1))
                            c_s_trial = np.zeros(shape=(Nx + 1, i + 1))  # a trail concrete stress
                            mod_c_c_i = np.zeros(shape=(Nx + 1, i + 1))  #
                            mom_ini_c_c = np.zeros(i + 1)

                            modulus_c_c = np.zeros(shape=(i + 1, size_strain + 1))
                            force_c_c = np.zeros(shape=(i + 1, size_strain + 1))  # layer; strain limit
                            moment_c_c = np.zeros(shape=(i + 1, size_strain + 1))
                            buckling_c_c = np.zeros(shape=(i + 1, size_strain + 1))

                            # concrete in tension array
                            e_c_ten_tot = np.zeros(m + 1)  # layer
                            con_expan_ten = np.zeros(shape=(Nx + 1, m + 1))
                            con_tran_ten = np.zeros(shape=(Nx + 1, m + 1))
                            con_creep_ten = np.zeros(shape=(Nx + 1, m + 1))
                            c_m_strain_ten = np.zeros(shape=(Nx + 1, m + 1))
                            stress_c_t = np.zeros(shape=(Nx + 1, m + 1))
                            c_s_trial_t = np.zeros(shape=(Nx + 1, m + 1))

                            force_c_t = np.zeros(shape=(m + 1, size_strain + 1))  # layer; strain limit
                            moment_c_t = np.zeros(shape=(m + 1, size_strain + 1))

                            mid_deflec[i] = l_NA / strain_limit[k] * (1 - math.cos(length * strain_limit[k] / 2 / l_NA))

                            # print(mid_deflec[i] * 1000)

                            # concrete in compression side
                            for j in range(0, i + 1):
                                e_c_comp_tot[j] = j * dx / l_NA * strain_limit[k]
                                ij = i - j  # real location in terms of Nx; for T_n
                                modulus_c_c[j][k] = 0

                                mom_ini_c_c[j] = width * dx ** 3 / 12 + (width * dx) * (
                                        height / 2 - (i - j) * dx) ** 2  # inividual second moment of each layer: I

                                for pp in range(0, Nx + 1):  # horizontal direction in a layer
                                    c_s_trial[pp][j] = 0
                                    con_expan[pp][j] = con_exp(T_n[pp][ij])
                                    c_m_strain[pp][j] = e_c_comp_tot[j] + con_expan[pp][j] - con_tran[pp][j] - \
                                                        con_creep[pp][j]
                                    stress_c_c[pp][j] = conc_fire(c_m_strain[pp][j], fc, T_n[pp][ij])

                                    while abs(c_s_trial[pp][j] - stress_c_c[pp][j]) > con_accuracy:
                                        c_s_trial[pp][j] = c_s_trial[pp][j] + con_incre

                                        if c_s_trial[pp][j] > conc_fy(T_n[pp][ij]) * fc:  # ensure it's within the limit
                                            c_s_trial[pp][j] = conc_fy(T_n[pp][ij]) * fc
                                            stress_c_c[pp][j] = conc_fy(T_n[pp][ij]) * fc
                                        else:
                                            con_tran[pp][j] = tran_e(c_s_trial[pp][j], T_n[pp][ij], fc)
                                            con_creep[pp][j] = cre_e(c_s_trial[pp][j], T_n[pp][ij], fc, n / 60)
                                            c_m_strain[pp][j] = e_c_comp_tot[j] + con_expan[pp][j] - con_tran[pp][j] - \
                                                                con_creep[pp][j]
                                            stress_c_c[pp][j] = conc_fire(c_m_strain[pp][j], fc, T_n[pp][ij])

                                    mod_c_c_i[pp][j] = sec_mod(c_m_strain[pp][j], stress_c_c[pp][j])
                                    modulus_c_c[j][k] += mod_c_c_i[pp][j] / Nx  # total E of concrete of layer j

                                    force_c_c[j][k] += stress_c_c[pp][j] * dx * dx  # total force of concrete of layer j

                                moment_c_c[j][k] = force_c_c[j][k] * (height / 2 - (i - j) * dx)
                                buckling_c_c[j][k] = (math.pi ** 2) * modulus_c_c[j][k] * mom_ini_c_c[j] / (
                                        length * BC) ** 2

                                t_force_c_c[i] += force_c_c[j][k]
                                t_moment_c_c[i] += moment_c_c[j][k]
                                t_buckling_c_c[i] += buckling_c_c[j][k]
                                # print(t_buckling_c_c[i]/1000)

                            # steel reinforcement

                            steel_expan = steel_exp(T_n[loc_rebar][loc_rebar])
                            # steel in compression side
                            e_s_comp_tot = strain_limit[k] * (l_NA - cover) / l_NA
                            e_s_comp = e_s_comp_tot + steel_expan
                            stress_s_c = steel_fire(fys, Es, e_s_comp, T_n[loc_rebar][loc_rebar])
                            mod_s_c = sec_mod(e_s_comp, stress_s_c)

                            t_force_s_c = stress_s_c * (math.pi * D_sr ** 2 / 4) * N_com_bar
                            t_moment_s_c = t_force_s_c * (0.5 * height - cover)

                            # steel in tension side
                            e_s_ten_tot = strain_limit[k] * (l_NA - height + cover) / l_NA
                            e_s_ten = e_s_ten_tot + steel_expan
                            stress_s_t = steel_fire(fys, Es, e_s_ten, T_n[loc_rebar][loc_rebar])
                            mod_s_t = sec_mod(e_s_ten, stress_s_t)

                            t_force_s_t = stress_s_t * (math.pi * D_sr ** 2 / 4) * N_ten_bar
                            t_moment_s_t = t_force_s_t * (0.5 * height - cover) * -1

                            mom_ini_s = math.pi * D_sr ** 4 / 64 + (math.pi * D_sr ** 2 / 4) * (height / 2 - cover) ** 2
                            buckling_s = math.pi ** 2 * (mod_s_c * N_com_bar + mod_s_t * N_ten_bar) * mom_ini_s / (
                                        length * BC) ** 2

                            total_force[i] = t_force_c_c[i] + t_force_c_t[i] + t_force_s_c + t_force_s_t
                            total_moment[i] = t_moment_c_c[i] + t_moment_c_t[i] + t_moment_s_c + t_moment_s_t
                            total_buckling[i] = t_buckling_c_c[i] + buckling_s

                            # stability
                            if total_force[i] <= 0:
                                total_force[i] = 1
                            ecc_cro[i] = total_moment[i] / total_force[i]
                            if ecc_cro[i] <= 0:
                                ecc_cro[i] = 1
                            load_stability[i] = total_buckling[i] * (
                                    math.acos(ecc_cro[i] / (mid_deflec[i] + ecc_cro[i])) * 2 / math.pi) ** 2
                            if load_stability[i] <= 0:
                                load_stability[i] = 1

                            load_failure[i] = 1 / (1 / total_force[i] + 1 / load_stability[i])

                        # # pure compression
                        # con_expan = np.zeros(shape=(Nx + 1, Ny + 1))
                        # c_m_strain = np.zeros(shape=(Nx + 1, Ny + 1))
                        # c_stress = np.zeros(shape=(Nx + 1, Ny + 1))
                        # c_mod = np.zeros(shape=(Nx + 1, Ny + 1))
                        # c_mom_ini = np.zeros(shape=(Nx + 1, Ny + 1))
                        # c_f_total = 0
                        # c_euler = 0
                        #
                        # for i in range(0, Nx + 1):
                        #     for j in range(0, Ny + 1):
                        #         con_expan[i][j] = con_exp(T_n[i][j])
                        #         c_m_strain[i][j] = strain_limit[k] + con_expan[i][j]
                        #         c_stress[i][j] = conc_fire(c_m_strain[i][j], fc, T_n[i][j])
                        #         c_mod[i][j] = sec_mod(c_m_strain[i][j], c_stress[i][j])
                        #         c_mom_ini[i][j] = dx * dx ** 3 / 12 + (dx * dx) * (height / 2 - j * dx) ** 2
                        #
                        #         c_f_total += c_stress[i][j] * dx * dx
                        #         c_euler += conc_E(fc, T_n[i][j]) * c_mom_ini[i][j]
                        #
                        # steel_expan = steel_exp(T_n[loc_rebar][loc_rebar])
                        # s_m_strain = strain_limit[k] + steel_expan
                        # s_stress = steel_fire(fys, Es, s_m_strain, T_n[loc_rebar][loc_rebar])
                        # s_f_total = (N_com_bar + N_ten_bar) * math.pi * D_sr ** 2 / 4 * s_stress
                        # I_steel = math.pi * D_sr ** 4 / 64 + (math.pi * D_sr ** 2 / 4) * (height / 2 - cover) ** 2 * (N_com_bar + N_ten_bar)
                        #
                        # total_force[Ny + 1] = c_f_total + s_f_total
                        # total_moment[Ny + 1] = 0
                        # load_euler = (math.pi ** 2) * (c_euler + s_stress / s_m_strain * I_steel) / (length * BC) ** 2
                        # load_failure[Ny + 1] = 1 / (1 / total_force[Nx + 1] + 1 / load_euler)

                    if strain_limit[k] < 0:
                        e_c_comp_tot = np.zeros(Ny + 1)
                        c_s_trial = np.zeros(shape=(Nx + 1, Ny + 1))
                        con_expan = np.zeros(shape=(Nx + 1, Ny + 1))
                        c_m_strain = np.zeros(shape=(Nx + 1, Ny + 1))
                        stress_c_c = np.zeros(shape=(Nx + 1, Ny + 1))
                        con_tran = np.zeros(shape=(Nx + 1, Ny + 1))
                        con_creep = np.zeros(shape=(Nx + 1, Ny + 1))
                        force_c_c = np.zeros(shape=(Ny + 1, size_strain + 1))
                        moment_c_c = np.zeros(shape=(Ny + 1, size_strain + 1))
                        mom_ini_c_c = np.zeros(Ny + 1)
                        mod_c_c_i = np.zeros(shape=(Nx + 1, Ny + 1))
                        modulus_c_c = np.zeros(shape=(Ny + 1, size_strain + 1))
                        buckling_c_c = np.zeros(shape=(Ny + 1, size_strain + 1))

                        for i in range(1, Ny + 1):
                            mid_deflec[i] = i * dx / strain_limit[k] * (
                                    1 - math.cos(length * strain_limit[k] / (2 * i * dx)))

                            for j in range(0, Ny + 1):
                                e_c_comp_tot[j] = (j + i) / i * strain_limit[k]
                                mom_ini_c_c[j] = width * dx ** 3 / 12 + (width * dx) * (height / 2 - j * dx) ** 2

                                # concrete in compression
                                for pp in range(0, Nx + 1):  # horizontal direction in a layer
                                    c_s_trial[pp][j] = 0
                                    con_expan[pp][j] = con_exp(T_n[pp][j])
                                    c_m_strain[pp][j] = e_c_comp_tot[j] + con_expan[pp][j] - con_tran[pp][j] - \
                                                        con_creep[pp][j]
                                    if c_m_strain[pp][j] <= 0:
                                        stress_c_c[pp][j] = 0
                                    else:
                                        stress_c_c[pp][j] = conc_fire(c_m_strain[pp][j], fc, T_n[pp][j])
                                        while abs(c_s_trial[pp][j] - stress_c_c[pp][j]) > con_accuracy:
                                            c_s_trial[pp][j] = c_s_trial[pp][j] + con_incre
                                            if c_s_trial[pp][j] > conc_fy(T_n[pp][j]) * fc:
                                                c_s_trial[pp][j] = conc_fy(T_n[pp][j]) * fc
                                                stress_c_c[pp][j] = conc_fy(T_n[pp][j]) * fc
                                            else:
                                                con_tran[pp][j] = tran_e(c_s_trial[pp][j], T_n[pp][j], fc)
                                                con_creep[pp][j] = cre_e(c_s_trial[pp][j], T_n[pp][j], fc, n / 60)
                                                c_m_strain[pp][j] = e_c_comp_tot[j] + con_expan[pp][j] - con_tran[pp][
                                                    j] - con_creep[pp][j]
                                                stress_c_c[pp][j] = conc_fire(c_m_strain[pp][j], fc, T_n[pp][j])

                                    mod_c_c_i[pp][j] = sec_mod(c_m_strain[pp][j], stress_c_c[pp][j])
                                    modulus_c_c[j][k] += mod_c_c_i[pp][j] / Nx
                                    force_c_c[j][k] += stress_c_c[pp][j] * dx * dx

                                moment_c_c[j][k] = force_c_c[j][k] * (height / 2 - j * dx)
                                buckling_c_c[j][k] = (math.pi ** 2) * modulus_c_c[j][k] * mom_ini_c_c[j] / (
                                        length * BC) ** 2

                                t_force_c_c[i] += force_c_c[j][k]
                                t_moment_c_c[i] += moment_c_c[j][k]
                                t_buckling_c_c[i] += buckling_c_c[j][k]

                                # steel in compression side
                                steel_expan = steel_exp(T_n[loc_rebar][loc_rebar])
                                e_s_comp_tot = strain_limit[k] * (cover + i * dx) / (i * dx)
                                e_s_comp = e_s_comp_tot + steel_expan
                                stress_s_c = steel_fire(fys, Es, e_s_comp, T_n[loc_rebar][loc_rebar])
                                mod_s_c = sec_mod(e_s_comp, stress_s_c)
                                t_force_s_c = stress_s_c * (math.pi * D_sr ** 2 / 4) * N_com_bar
                                t_moment_s_c = t_force_s_c * (0.5 * height - cover)

                                # steel in tension side
                                e_s_ten_tot = strain_limit[k] * (height - cover + i * dx) / (i * dx)
                                e_s_ten = e_s_ten_tot + steel_expan
                                stress_s_t = steel_fire(fys, Es, e_s_ten, T_n[loc_rebar][loc_rebar])
                                mod_s_t = sec_mod(e_s_ten, stress_s_t)
                                t_force_s_t = stress_s_t * (math.pi * D_sr ** 2 / 4) * N_ten_bar
                                t_moment_s_t = t_force_s_t * (0.5 * height - cover) * -1

                                mom_ini_s = math.pi * D_sr ** 4 / 64 + (math.pi * D_sr ** 2 / 4) * (
                                        height / 2 - cover) ** 2
                                buckling_s = math.pi ** 2 * (mod_s_c * 2 + mod_s_t * 2) * mom_ini_s / (length * BC) ** 2

                                total_force[i] = t_force_c_c[i] + t_force_c_t[i] + t_force_s_c + t_force_s_t
                                total_moment[i] = t_moment_c_c[i] + t_moment_c_t[i] + t_moment_s_c + t_moment_s_t
                                total_buckling[i] = t_buckling_c_c[i] + buckling_s

                                # stability
                                if total_force[i] <= 0:
                                    total_force[i] = 1
                                ecc_cro[i] = total_moment[i] / total_force[i]
                                if ecc_cro[i] <= 0:
                                    ecc_cro[i] = 1
                                load_stability[i] = total_buckling[i] * (
                                        math.acos(ecc_cro[i] / (mid_deflec[i] + ecc_cro[i])) * 2 / math.pi) ** 2

                                if load_stability[i] <= 0:
                                    load_stability[i] = 1

                                load_failure[i] = 1 / (1 / total_force[i] + 1 / load_stability[i])

                    # pure compression
                    con_expan = np.zeros(shape=(Nx + 1, Ny + 1))
                    c_m_strain = np.zeros(shape=(Nx + 1, Ny + 1))
                    c_stress = np.zeros(shape=(Nx + 1, Ny + 1))
                    c_mod = np.zeros(shape=(Nx + 1, Ny + 1))
                    c_mom_ini = np.zeros(shape=(Nx + 1, Ny + 1))
                    c_f_total = 0
                    c_euler = 0

                    for i in range(0, Nx + 1):
                        for j in range(0, Ny + 1):
                            con_expan[i][j] = con_exp(T_n[i][j])
                            c_m_strain[i][j] = strain_limit[k] + con_expan[i][j]
                            c_stress[i][j] = conc_fire(c_m_strain[i][j], fc, T_n[i][j])
                            c_mod[i][j] = sec_mod(c_m_strain[i][j], c_stress[i][j])
                            c_mom_ini[i][j] = dx * dx ** 3 / 12 + (dx * dx) * (height / 2 - j * dx) ** 2

                            c_f_total += 0.85 * c_stress[i][j] * dx * dx
                            c_euler += conc_E(fc, T_n[i][j]) * c_mom_ini[i][j]

                    steel_expan = steel_exp(T_n[loc_rebar][loc_rebar])
                    s_m_strain = strain_limit[k] + steel_expan
                    s_stress = steel_fire(fys, Es, s_m_strain, T_n[loc_rebar][loc_rebar])
                    s_f_total = (N_com_bar + N_ten_bar) * math.pi * D_sr ** 2 / 4 * s_stress
                    I_steel = math.pi * D_sr ** 4 / 64 + (math.pi * D_sr ** 2 / 4) * (
                            height / 2 - cover) ** 2 * (N_com_bar + N_ten_bar)

                    total_force[Ny + 1] = c_f_total + s_f_total
                    total_moment[Ny + 1] = 0
                    load_euler = (math.pi ** 2) * (c_euler + s_stress / s_m_strain * I_steel) / (
                            length * BC) ** 2
                    load_failure[Ny + 1] = 1 / (1 / total_force[Nx + 1] + 1 / load_euler)

                    # print("k = ", k)
                    # for i in range(0, Nx + 2):
                    #     print(total_moment[i] / 1000)
                    # for i in range(0, Nx + 2):
                    #     print(total_force[i] / 1000)
                    # print("struc force")
                    # for i in range(0, Nx + 2):
                    #     print(load_failure[i] / 1000)

                    for i in range(0, Nx + 2):
                        int_force_1[i] = total_moment[i] / ecc

                    idx_1 = np.argwhere(np.diff(np.sign(int_force_1 - load_failure))).flatten()
                    moment_1 = (load_failure[idx_1] - (load_failure[idx_1 + 1] - load_failure[idx_1]) /
                                (total_moment[idx_1 + 1] - total_moment[idx_1]) * total_moment[idx_1]) / \
                               (1 / ecc - (load_failure[idx_1 + 1] - load_failure[idx_1]) /
                                (total_moment[idx_1 + 1] - total_moment[idx_1]))

                    # idx_1 = np.argwhere(np.diff(np.sign(int_force_1 - total_force))).flatten()
                    # moment_1 = (total_force[idx_1] - (total_force[idx_1 + 1] - total_force[idx_1]) /
                    #             (total_moment[idx_1 + 1] - total_moment[idx_1]) * total_moment[idx_1]) / \
                    #            (1 / ecc - (total_force[idx_1 + 1] - total_force[idx_1]) /
                    #             (total_moment[idx_1 + 1] - total_moment[idx_1]))

                    force_1 = moment_1 / ecc
                    real_i_1 = (total_moment[idx_1] - moment_1) / (
                            total_moment[idx_1] - total_moment[idx_1 + 1]) + idx_1 - 1

                    lower_i = math.floor(real_i_1[-1])
                    upper_i = math.ceil(real_i_1[-1])

                    deflection[k] = mid_deflec[lower_i] + (real_i_1[-1] - lower_i) / (upper_i - lower_i) * (
                            mid_deflec[upper_i] - mid_deflec[lower_i])
                    loading[k] = force_1[-1]

                    # print(deflection[k] * 1000)
                    # print(loading[k] / 1000)

                    real_strain_1 = (real_i_1 * dx - height / 2) * strain_limit[k] / (real_i_1 * dx)
                    # print("strain")
                    # print(real_strain_1[-1])
                    # print(real_strain_1)

                idx = np.argwhere(np.diff(np.sign(loading - app_loading))).flatten()
                middle_deflec = (Pa - loading[idx]) / (loading[idx + 1] - loading[idx]) * (
                        deflection[idx + 1] - deflection[idx]) + deflection[idx]
                # print('deflection')
                print(middle_deflec[0] * 1000)
                print(np.amax(loading) / 1000)


time = 7200
dt = 1
p_a = 2480
h_0 = 25
s = 5.67 * 10 ** -8
e_0 = 0.7

dx = 0.005
width = 0.25
height = 0.25
length = 3
BC = 1

cover = 50 * 10 ** (-3)
D_sr = 25 * 10 ** (-3)
N_com_bar = 2
N_ten_bar = 2

fc = 40 * 10 ** 6
fct = 0 * 10 ** 6  # Pa
fR1 = 0 * 10 ** 6
fR3 = 0 * 10 ** 6  # Pa
fys = 500 * 10 ** 6
fts = 600 * 10 ** 6
Es = 200 * 10 ** 9

Pa = 500 * 10 ** 3
ecc = 30 * 10 ** (-3)

solver(time, dt, p_a, h_0, s, e_0, height, width, cover, dx, D_sr, N_com_bar, N_ten_bar, fys, Es, fc, fct, Pa, ecc,
       length, BC)