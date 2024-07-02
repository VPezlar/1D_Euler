import numpy as np


def VanLeer(U1_old, a, M, u, gamma):
    Nc = len(U1_old)
    # Logical Vectorized (Using Logical Operator, only values at TRUE are filled, rest is zero)
    # Right Going Supersonic
    # Positive
    F1_p_r_s = (U1_old * a * M) * (M >= 1)
    F2_p_r_s = (U1_old * a ** 2 * (M ** 2 + 1 / gamma)) * (M >= 1)
    F3_p_r_s = (U1_old * a ** 3 * M * (0.5 * M ** 2 + 1 / (gamma - 1))) * (M >= 1)
    # Negative
    F1_n_r_s = np.zeros(Nc)
    F2_n_r_s = np.zeros(Nc)
    F3_n_r_s = np.zeros(Nc)

    # Left Going Supersonic
    # Positive
    F1_p_l_s = np.zeros(Nc)
    F2_p_l_s = np.zeros(Nc)
    F3_p_l_s = np.zeros(Nc)
    # Negative
    F1_n_l_s = (U1_old * a * M) * (M <= -1)
    F2_n_l_s = (U1_old * a ** 2 * (M ** 2 + 1 / gamma)) * (M <= -1)
    F3_n_l_s = (U1_old * a ** 3 * M * (0.5 * M ** 2 + 1 / (gamma - 1))) * (M <= -1)

    # Subsonic
    # Positive
    F1_p_sub = (U1_old * a * (((M + 1) / 2) ** 2)) * ((M > -1) & (M < 1))
    F2_p_sub = (((U1_old * a * (((M + 1) / 2) ** 2)) / gamma) * (((gamma - 1) * u) + (2 * a))) * ((M > -1) & (M < 1))
    F3_p_sub = (((U1_old * a * (((M + 1) / 2) ** 2)) / (2 * (gamma ** 2 - 1))) * (((gamma - 1) * u) + (2 * a)) ** 2) * (
                (M > -1) & (M < 1))
    # Negative
    F1_n_sub = ((-1) * U1_old * a * (((M - 1) / 2) ** 2)) * ((M > -1) & (M < 1))
    F2_n_sub = ((((-1) * U1_old * a * (((M - 1) / 2) ** 2)) / gamma) * (((gamma - 1) * u) - (2 * a))) * (
                (M > -1) & (M < 1))
    F3_n_sub = ((((-1) * U1_old * a * (((M - 1) / 2) ** 2)) / (2 * (gamma ** 2 - 1))) * (
                ((gamma - 1) * u) - (2 * a)) ** 2) * ((M > -1) & (M < 1))

    # Inter-cell Fluxes (Adding the three cases, left/right supersonic and subsonic)
    # Positive
    F1_p = F1_p_l_s + F1_p_r_s + F1_p_sub
    F2_p = F2_p_l_s + F2_p_r_s + F2_p_sub
    F3_p = F3_p_l_s + F3_p_r_s + F3_p_sub
    # Negative
    F1_n = F1_n_l_s + F1_n_r_s + F1_n_sub
    F2_n = F2_n_l_s + F2_n_r_s + F2_n_sub
    F3_n = F3_n_l_s + F3_n_r_s + F3_n_sub

    return F1_p, F2_p, F3_p, F1_n, F2_n, F3_n
