# Import pacakges
from popsborder.contamination import *


# Testing from Clarke et al. (2023) paper (conversion done in ChatGPT)
ty    = [0, 4, 4, 7, 21]
freq  = [68, 1, 1, 1, 1]
pars  = bb_group_model(ty=ty, b=94, B=1000, Nbar=100,
                       freq=freq, theta=np.inf, R=1000,
                       startval=(0.0, 0.0), se=True)

print(f"alpha={pars['alpha']:.4f}, beta={pars['beta']:.4f}, mu={pars['mu']:.4f}")
print(f"rho={pars['rho']:.6f},  D={pars['D']:.3f}")
print(f"prob(leak)={pars['prob_leak']:.6e}")
