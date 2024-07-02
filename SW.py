import numpy as np


def SW(U1_old, gamma, a, u, H):
    # Getting Eigenvalues and Splitting them
    # Eigenvalues
    l1 = u - a
    l2 = u
    l3 = u + a

    # Eigenvalue Splitting (Stagger-Warmig)
    l1_p = (l1 + abs(l1)) / 2
    l1_n = (l1 - abs(l1)) / 2

    l2_p = (l2 + abs(l2)) / 2
    l2_n = (l2 - abs(l2)) / 2

    l3_p = (l3 + abs(l3)) / 2
    l3_n = (l3 - abs(l3)) / 2

    # Inter-cell Fluxes
    Coeff = U1_old / (2 * gamma)
    # Positive
    F1_p = Coeff * (l1_p + 2 * (gamma - 1) * l2_p + l3_p)
    F2_p = Coeff * ((u - a) * l1_p + 2 * (gamma - 1) * u * l2_p + (u + a) * l3_p)
    F3_p = Coeff * ((H - u * a) * l1_p + (gamma - 1) * (u ** 2) * l2_p + (H + u * a) * l3_p)
    # Negative
    F1_n = Coeff * (l1_n + 2 * (gamma - 1) * l2_n + l3_n)
    F2_n = Coeff * ((u - a) * l1_n + 2 * (gamma - 1) * u * l2_n + (u + a) * l3_n)
    F3_n = Coeff * ((H - u * a) * l1_n + (gamma - 1) * (u ** 2) * l2_n + (H + u * a) * l3_n)

    return F1_p, F2_p, F3_p, F1_n, F2_n, F3_n
