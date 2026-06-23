import numpy as np
from scipy.linalg import circulant
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import signal
from pandas import Series
from random import gauss
import ipywidgets as widgets
from ipywidgets import interact, fixed

def compute_firerate(I, M, h): 
    return np.linalg.solve(I - M, h)
    
def convert_matrix(M, alpha = 0.97):
    # M = np.array(M, dtype=float) 
    eigvals = np.linalg.eigvals(M)
    max_ev = np.max(np.real(eigvals))
    # print(np.max(np.real(eigvals)))
    # M = (alpha / max_ev) * M

    return M

def gaussian (x, a_e, sigma_e, a_i,r):
    sigma_i = r * sigma_e
    gauss_e = (a_e / (np.sqrt(2*np.pi) * sigma_e)) * np.exp(-0.5 * (x/sigma_e) ** 2)
    gauss_i = (a_i / (np.sqrt(2*np.pi) * sigma_i)) * np.exp(-0.5 * (x/sigma_i) ** 2)

    return gauss_e - gauss_i

def recurrent_connections(N, a_e, a_i, sigma_e, r):
    x = np.arange(N)
    dist = np.minimum(x, N-x) # berechnet kürzesten periodischen Abstand auf einem Ring mit N Punkten
    kernel = gaussian(dist, a_e, sigma_e, a_i, r)
    m = circulant(kernel)

    return m

def autoval_distr2(G):

    valsG, vecsG = np.linalg.eig(G)
    N = G.shape[0]

    eigenvectors = vecsG.T  

    frequencies = []
    eigenvalues = []
    eigenvectors_list = []

    for i in range(N):
        amplitude = eigenvectors[i]

        # FFT
        fourierTransform = np.fft.fft(amplitude) / len(amplitude)
        fourierTransform = fourierTransform[:len(amplitude)//2]

        tpCount = len(amplitude)
        values = np.arange(tpCount//2)
        freq = values / tpCount

        
        dominant_freq = freq[np.argmax(abs(fourierTransform))]

        frequencies.append(dominant_freq)
        eigenvalues.append(valsG[i])
        eigenvectors_list.append(eigenvectors[i])

    idx = np.argsort(frequencies)

    frequencies = np.array(frequencies)[idx]
    eigenvalues = np.array(eigenvalues)[idx]
    eigenvectors_list = np.array(eigenvectors_list)[idx]

    # return frequencies, eigenvalues, eigenvectors_list

    return np.array(frequencies), np.array(eigenvalues), np.array(eigenvectors_list)

# function contributed by Lorenzo Butti
def F(N):

    sigma = 0.1
    M = 1

    xx, yy = np.meshgrid(np.arange(0, N, 1), np.arange(0, M, 1))
    dx = cdist(xx.T, xx.T)
    dx = np.minimum(dx, N * np.ones_like(dx) - dx)

    # Gaussian Random Field (1D Serie)
    series1 = [gauss(0.0, 1.0) for k in range(N)]
    grf = Series(series1)

    w = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-dx**2 / (2 * sigma**2))

    D = int(N / 2)
    correzione = np.zeros((N, N))

    
    for i in range(N):
        correzione.T[i] = signal.convolve(w.T[D], grf, mode='same')

    return correzione

def perturb_M (M_hom, correzione): 
    M_het = (1 + (3)*correzione) * M_hom

    return M_het

def compute_influence_matrix (M_het, N): 
   
    G = np.zeros((N, N))
    I = np.eye(N)

    for j in range(N):
        h = np.zeros(N)
        h[j] = 1.0

        r = compute_firerate(I, M_het, h)

        G[:, j] = r - h

    return G

def compute_variance_across_rows(G):
    return np.var(np.mean(M, axis=1))

def analyze_all_symmetric(
    M,
    pairs,
    x_percent,
):

    N = M.shape[0]

    results = {
        "Ms": [],
        "freqs": [],
        "eigvals": [],
        "eigvecs": [],
        "G": [],
        "variance_influence": [],
        "variance_connectivity": [],
        "dist_eigval": [],
        "dist_eigvec": [],
        "pairs": pairs,
        "x_percent": x_percent,
    }

    f0, e0, v0 = autoval_distr2(M)

    G0 = compute_influence_matrix(M, N)
    var_i0 = compute_variance_over_stimulus(G0)
    var_c0 = compute_variance_over_stimulus(M)

    results["Ms"].append(M)
    results["freqs"].append(f0)
    results["eigvals"].append(e0)
    results["eigvecs"].append(v0)
    results["G"].append(G0)
    results["variance_influence"].append(var_i0)
    results["variance_connectivity"].append(var_c0)
    results["dist_eigval"].append(0.0)
    results["dist_eigvec"].append(0.0)

    mean_influence = np.mean(G0)
    delta = x_percent * mean_influence

    M_i = M.copy()

    for (i, j) in pairs:
        M_i[i, j] += delta
        M_i[j, i] += delta  

    f_i, e_i, v_i = autoval_distr2(M_i)

    G_i = compute_influence_matrix(M_i, N)
    var_influence = compute_variance_over_stimulus(G_i)
    var_connectivity = compute_variance_over_stimulus(M_i)

    d_val = e_i - e0
    d_vec = np.abs(v_i) - np.abs(v0)

    results["Ms"].append(M_i)
    results["freqs"].append(f_i)
    results["eigvals"].append(e_i)
    results["eigvecs"].append(v_i)
    results["G"].append(G_i)
    results["variance_influence"].append(var_influence)
    results["variance_connectivity"].append(var_connectivity)
    results["dist_eigval"].append(d_val)
    results["dist_eigvec"].append(d_vec)

    return results

def plot_mean_and_variance_MG(results, index):

    M = results["Ms"][index]
    G = results["G"][index]

    mean_M = np.mean(M, axis=1)
    mean_G = np.mean(G, axis=1)

    var_M = results["variance_connectivity"][index]
    var_G = results["variance_influence"][index]

    neurons = np.arange(len(mean_M))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    axes[0, 0].plot(neurons, mean_M, linewidth=2)
    axes[0, 0].set_title("Mean Connectivity")
    axes[0, 0].set_ylabel("Mean over j")
    axes[0, 0].grid(True)

    axes[0, 1].plot(neurons, mean_G, linewidth=2)
    axes[0, 1].set_title("Mean Influence")
    axes[0, 1].grid(True)

    axes[1, 0].plot(neurons, var_M, linewidth=2)
    axes[1, 0].set_title("Variance Connectivity")
    axes[1, 0].set_xlabel("Neuron i")
    axes[1, 0].set_ylabel("Variance over j")
    axes[1, 0].grid(True)

    axes[1, 1].plot(neurons, var_G, linewidth=2)
    axes[1, 1].set_title("Variance Influence")
    axes[1, 1].set_xlabel("Neuron i")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def influence_stats(M):
    N = M.shape[0]
    G = compute_influence_matrix(M, N)

    print(f"Average influence: {np.mean(G):.6f}")
    print(f"Min influence:     {np.min(G):.6f}")
    print(f"Max influence:     {np.max(G):.6f}")
    print(f"Std influence:     {np.std(G):.6f}")

    return G
