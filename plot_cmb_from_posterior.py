import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import camb
from importlib import import_module

def camb_tt_Dell_from_sample(sample, lmax=3000):
    H0      = float(sample["H0"])
    ombh2   = float(sample["ombh2"])
    omch2   = float(sample["omch2"])
    tau     = float(sample["tau"])
    ns      = float(sample["n_s"])
    As_in   = float(sample["A_s"])
    log10_zc = float(sample["log10_zc"])
    fEDE_zc  = float(sample["fEDE_zc"])
    n_power  = float(sample["n_power"])
    theta_i  = float(sample["theta_i"])

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As_in, ns=ns, pivot_scalar=0.05)

    eq_mod = import_module("camb.dark_energy")
    EarlyQuintessence = getattr(eq_mod, "EarlyQuintessence")
    ede = EarlyQuintessence()
    ede.n = n_power
    ede.theta_i = theta_i
    ede.use_zc = True
    ede.zc = 10.0**log10_zc
    ede.fde_zc = fEDE_zc
    ede.frac_lambda0 = 1.0
    ede.npoints = 800
    ede.min_steps_per_osc = 20
    pars.DarkEnergy = ede

    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    dl_tot = powers['total']
    ells = np.arange(dl_tot.shape[0])
    mask = ells >= 2
    return ells[mask], dl_tot[mask, 0]

# Load posterior sample
with open("PATH TO THE PARAMS.json") as f:    # <-- Change the Path
    best_sample = json.load(f)

# Compute theory TT
l_theory, DTT_theory = camb_tt_Dell_from_sample(best_sample)

# Load Planck TT bandpowers
planck_file = "PATH TO FILE COM_PowerSpect_CMB_R2.02.fits"   # <- change this path
planck_data_hi = Table.read(planck_file, hdu=7)
planck_data_lo = Table.read(planck_file, hdu=1)

plt.figure(figsize=(7,5))
plt.errorbar(planck_data_lo['ELL'], planck_data_lo['D_ELL'],
             yerr=[planck_data_lo['ERRDOWN'], planck_data_lo['ERRUP']],
             fmt='o', markersize=2, capsize=3, color='k')

plt.errorbar(planck_data_hi['ELL'], planck_data_hi['D_ELL'],
             yerr=planck_data_hi['ERR'], fmt='o', markersize=2, capsize=3, color='k',
             label="Planck 2018 TT (data)")

plt.plot(l_theory, DTT_theory, lw=2, label="EarlyQuintessence (TT)")

plt.xscale("log")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{TT}$  [$\mu$K$^2$]")
plt.xlim(2, 3000)
plt.legend()
plt.tight_layout()
plt.show()
