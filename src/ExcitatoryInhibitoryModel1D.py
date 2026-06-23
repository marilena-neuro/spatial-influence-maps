import numpy as np
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import signal
from random import gauss
from pandas import Series

def gaussian(x, a, sigma):
    gauss = (a / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * (x/sigma) ** 2)
    return gauss

# for 2D input in the middle (?)
def recurrent_connections(N, a, sigma):
    x = np.arange(N)
    dist = np.minimum(x, N-x) # berechnet kürzesten periodischen Abstand auf einem Ring mit N Punkten
    kernel = gaussian(dist, a, sigma)
    m = circulant(kernel)

    return m

def convert_matrix(W_EE, W_EI, W_IE, W_II, alpha):
    # M = np.array(M, dtype=float)
    
    top = np.hstack((W_EE, -W_EI))
    bottom = np.hstack((W_IE, -W_II))
    C = np.vstack((top, bottom))
    
    eigvals = np.linalg.eigvals(C)
    max_ev = np.max(np.real(eigvals))
    print(np.max(np.real(eigvals)))
    C = (alpha / max_ev) * C

    return C

def not_rescaled(W_EE, W_EI, W_IE, W_II):
    # M = np.array(M, dtype=float)
    
    top = np.hstack((W_EE, -W_EI))
    bottom = np.hstack((W_IE, -W_II))
    C = np.vstack((top, bottom))
    
    eigvals = np.linalg.eigvals(C)
    max_ev = np.max(np.real(eigvals))
    print(np.max(np.real(eigvals)))
    #C = (alpha / max_ev) * C

    return C

def compute_firerate(I, C, h): 
    return np.linalg.solve(I - C, h)

### PERTURBATION FUNCTIONS

# function contributed by Lorenzo Butti
def F(N, seed=42, amplitude = 0.5):
    np.random.seed(seed)
    sigma = 2
    M = 1

    xx, yy = np.meshgrid(np.arange(0, N, 1), np.arange(0, M, 1))
    dx = cdist(xx.T, xx.T)
    dx = np.minimum(dx, N * np.ones_like(dx) - dx)

    series1 = [gauss(0.0, 1.0) for k in range(N)]
    grf = Series(series1)

    w = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-dx**2 / (2 * sigma**2))

    D = int(N/2)
    correzione = np.zeros((N, N))

    for i in range(N):
        correzione.T[i] = signal.convolve(w.T[D], grf, mode='same')

    correzione /= np.std(correzione)   
    correzione *= amplitude            

    return correzione

def perturb_C(C, corr):
    N = corr.shape[0]

    C_EE = C[:N,     :N]
    C_EI = C[:N,     N:2*N]
    C_IE = C[N:2*N,  :N]
    C_II = C[N:2*N,  N:2*N]

    C_EE_het = C_EE * (1 + corr)
    C_EI_het = C_EI * (1 + corr)
    C_IE_het = C_IE * (1 + corr)
    C_II_het = C_II * (1 + corr)

    C_het = np.block([
        [C_EE_het, C_EI_het],
        [C_IE_het, C_II_het]
    ])

    return C_het

def make_gain_vector(N, seed=42, amplitude=0.3):
    rng = np.random.default_rng(seed)
    g = 1.0 + amplitude * rng.standard_normal(N)
    return g  

def perturb_C_pointwise(C, N, seed=42, amplitude=0.3):
    
    g_E = make_gain_vector(N, seed=seed,   amplitude=amplitude)
    g_I = make_gain_vector(N, seed=seed+1, amplitude=amplitude)

    G_EE = np.outer(g_E, g_E)
    G_EI = np.outer(g_E, g_I)
    G_IE = np.outer(g_I, g_E)
    G_II = np.outer(g_I, g_I)

    C_EE = C[:N,    :N]
    C_EI = C[:N,    N:2*N]
    C_IE = C[N:2*N, :N]
    C_II = C[N:2*N, N:2*N]

    C_het = np.block([
        [C_EE * G_EE, C_EI * G_EI],
        [C_IE * G_IE, C_II * G_II]
    ])
    return C_het
