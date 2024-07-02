import numpy as np


def AUSM(U1_old, a, p, M, u, H):
    Nc = len(U1_old)
    # Logical Vectorized (Using Logical Operator, only values at TRUE are filled, rest is zero)
    # Right Going Supersonic------------------
    # Positive Convective Term
    F1C_p_r_s = (U1_old * a * M) * (M > 1)
    F2C_p_r_s = (U1_old * a * M * u) * (M > 1)
    F3C_p_r_s = (U1_old * a * M * H) * (M > 1)
    # Positive Pressure Term
    F1P_p_r_s = np.zeros(Nc)
    F2P_p_r_s = p * (M > 1)
    F3P_p_r_s = np.zeros(Nc)

    # Negative Convective Term
    F1C_n_r_s = np.zeros(Nc)
    F2C_n_r_s = np.zeros(Nc)
    F3C_n_r_s = np.zeros(Nc)
    # Negative Pressure Term
    F1P_n_r_s = np.zeros(Nc)
    F2P_n_r_s = np.zeros(Nc)
    F3P_n_r_s = np.zeros(Nc)
    #----------------------------------------

    # Left Going Supersonic
    # Positive Convective Term
    F1C_p_l_s = np.zeros(Nc)
    F2C_p_l_s = np.zeros(Nc)
    F3C_p_l_s = np.zeros(Nc)
    # Positive Pressure Term
    F1P_p_l_s = np.zeros(Nc)
    F2P_p_l_s = np.zeros(Nc)
    F3P_p_l_s = np.zeros(Nc)

    # Negative Convective Term
    F1C_n_l_s = (U1_old * a * M) * (M < -1)
    F2C_n_l_s = (U1_old * a * M * u) * (M < -1)
    F3C_n_l_s = (U1_old * a * M * H) * (M < -1)
    # Negative Pressure Term
    F1P_n_l_s = np.zeros(Nc)
    F2P_n_l_s = p * (M < -1)
    F3P_n_l_s = np.zeros(Nc)
    #----------------------------------------

    # Subsonic ------------------------------
    # Positive Convective Term
    F1C_p_sub = (U1_old * a * (((M + 1) ** 2) / 4)) * ((M >= -1) & (M <= 1))
    F2C_p_sub = (U1_old * a * (((M + 1) ** 2) / 4) * u) * ((M >= -1) & (M <= 1))
    F3C_p_sub = (U1_old * a * (((M + 1) ** 2) / 4) * H) * ((M >= -1) & (M <= 1))
    # Positive Pressure Term
    F1P_p_sub = np.zeros(Nc)
    F2P_p_sub = (p * ((1 + M) / 2)) * ((M >= -1) & (M <= 1))
    F3P_p_sub = np.zeros(Nc)

    # Negative Convective Term
    F1C_n_sub = (U1_old * a * (((-1) * ((M - 1) ** 2)) / 4)) * ((M >= -1) & (M <= 1))
    F2C_n_sub = (U1_old * a * (((-1) * ((M - 1) ** 2)) / 4) * u) * ((M >= -1) & (M <= 1))
    F3C_n_sub = (U1_old * a * (((-1) * ((M - 1) ** 2)) / 4) * H) * ((M >= -1) & (M <= 1))
    # Negative Pressure Term
    F1P_n_sub = np.zeros(Nc)
    F2P_n_sub = p * ((1 - M) / 2) * ((M > -1) & (M <= 1))
    F3P_n_sub = np.zeros(Nc)
    #----------------------------------------

    # Inter-cell Fluxes (Adding the three cases, left/right supersonic and subsonic)
    # Positive
    F1_p = (F1C_p_r_s + F1P_p_r_s) + (F1C_p_l_s + F1P_p_l_s) + (F1C_p_sub + F1P_p_sub)
    F2_p = (F2C_p_r_s + F2P_p_r_s) + (F2C_p_l_s + F2P_p_l_s) + (F2C_p_sub + F2P_p_sub)
    F3_p = (F3C_p_r_s + F3P_p_r_s) + (F3C_p_l_s + F3P_p_l_s) + (F3C_p_sub + F3P_p_sub)
    # Negative
    F1_n = (F1C_n_r_s + F1P_n_r_s) + (F1C_n_l_s + F1P_n_l_s) + (F1C_n_sub + F1P_n_sub)
    F2_n = (F2C_n_r_s + F2P_n_r_s) + (F2C_n_l_s + F2P_n_l_s) + (F2C_n_sub + F2P_n_sub)
    F3_n = (F3C_n_r_s + F3P_n_r_s) + (F3C_n_l_s + F3P_n_l_s) + (F3C_n_sub + F3P_n_sub)

    return F1_p, F2_p, F3_p, F1_n, F2_n, F3_n
