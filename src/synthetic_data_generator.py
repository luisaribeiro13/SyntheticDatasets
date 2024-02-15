from matplotlib import colors
from scipy.interpolate import Rbf
import csv
import scienceplots
import seaborn as sns
import sys
import time
import lmfit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('science')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


class Model:
    def __init__(self, fg_exp, mu_app_exp, Swc, Sgr, mi_w, mi_g, nw_model, ng_model, krw0_model, krg0_model, fmmob_model, SF_model, sfbet_model):
        self.fg_exp = np.asarray(fg_exp)
        self.mu_app_exp = np.asarray(mu_app_exp)
        self.Swc = Swc
        self.Sgr = Sgr
        self.mi_w = mi_w
        self.mi_g = mi_g
        self.nw_model = nw_model
        self.ng_model = ng_model
        self.krw0_model = krw0_model
        self.krg0_model = krg0_model
        self.fmmob_model = fmmob_model
        self.SF_model = SF_model
        self.sfbet_model = sfbet_model

    def krw(self, params, Sw):
        krw0 = params['krw0']
        nw = params['nw']
        se = (Sw - self.Swc) / (1.0 - self.Swc - self.Sgr)
        k = krw0 * np.power(se, nw)
        return k

    def krw2(self, params, Sw, nw, krw0):
        se = (Sw - self.Swc) / (1.0 - self.Swc - self.Sgr)
        k = krw0 * np.power(se, nw)
        return k

    def krg(self, params, Sw):
        krg0 = params['krg0']
        ng = params['ng']
        se = 1.0 - ((Sw - self.Swc) / (1.0 - self.Swc - self.Sgr))
        k = krg0 * np.power(se, ng)
        return k

    def water_mobility(self, params, Sw):
        return (self.krw(params, Sw) / self.mi_w)

    def water_mobility2(self, params, Sw, nw, krw0):
        return (self.krw2(params, Sw, nw, krw0) / self.mi_w)

    def gas_mobility(self, params, Sw):
        return (self.krg(params, Sw) / self.mi_g)

    def MRF(self, params, Sw):
        fmmob = params['fmmob']
        SF = params['SF']
        sfbet = params['sfbet']
        F2 = 0.5 + np.arctan(sfbet * (Sw - SF)) / np.pi
        MRF = 1 + fmmob * F2
        return MRF

    def resid_MRF(self, params, nw, krw0):
        krg0 = params['krg0']
        ng = params['ng']

        Sw = self.Swc + (1.0 - self.Swc - self.Sgr) * \
            np.power((self.mi_w * (1.0 - self.fg_exp)) /
                     (krw0 * self.mu_app_exp), 1.0 / nw)

        MRF_exp = (self.mu_app_exp * self.krg(params, Sw)) / self.fg_exp

        MRF_estim = self.MRF(params, Sw)

        return (MRF_estim - MRF_exp) / (np.max(MRF_exp))

    def Sw(self, nw, krw0):
        Sw = Swc + (1.0 - self.Swc - self.Sgr) * \
            np.power((self.mi_w * (1.0 - self.fg_exp)) /
                     (krw0 * self.mu_app_exp), 1.0 / nw)
        return Sw

    def resid_Sw(self, params):
        krw0 = params['krw0']
        nw = params['nw']
        Sw = self.Sw(nw, krw0)
        Sw_exp = self.Sw(self.nw_model, self.krw0_model)
        return (Sw - Sw_exp) / (np.max(Sw_exp))

    def apparent_viscosity2(self, params, nw, krw0):
        Sw_exp = self.Sw(self.nw_model, self.krw0_model)
        mobw = self.water_mobility2(params, Sw_exp, nw, krw0)
        mobg = self.gas_mobility(params, Sw_exp)
        MRF = 1.0  # self.MRF(params, Sw_exp)
        return 1.0 / (mobw + (mobg / MRF))

    def apparent_viscosity(self, params):
        krw0 = params['krw0']
        nw = params['nw']
        Sw = Swc + (1.0 - self.Swc - self.Sgr) * \
            np.power((self.mi_w * (1.0 - self.fg_exp)) /
                     (krw0 * self.mu_app_exp), 1.0 / nw)
        mobw = self.water_mobility(params, Sw)
        mobg = self.gas_mobility(params, Sw)
        MRF = 1.0  # self.MRF(params, Sw)
        return 1.0 / (mobw + (mobg / MRF))

    def resid_muapp(self, params, nw, krw0):
        mua_estim = self.apparent_viscosity2(
            params, nw, krw0)

        return (mua_estim - self.mu_app_exp) / (np.max(mu_app_exp))

    def find_closest_indices(self, numbers, target_points):
        closest_indices = []
        for target in target_points:
            closest_index = min(range(len(numbers)),
                                key=lambda i: abs(numbers[i] - target))
            closest_indices.append(closest_index)
        return closest_indices


if __name__ == "__main__":
    Swc = 0.2
    Sgr = 0.1
    nw_model = 4.2
    ng_model = 1.4
    krw0_model = 0.5
    krg0_model = 0.6
    fmmob_model = 4.342E3 - 1.0
    SF_model = 0.3409
    sfbet_model = 424
    mi_w = 0.5
    mi_g = 0.02112
    permeability = 368
    phi = 12
    L = 8.8
    d = 2.4

    fg_exp = []
    mu_app_exp = []
    Sw = np.linspace(Swc, 1 - Sgr, 500)
    model0 = Model(fg_exp, mu_app_exp, Swc, Sgr, mi_w,

                   mi_g,
                   nw_model, ng_model, krw0_model, krg0_model, fmmob_model, SF_model, sfbet_model)

    params0 = dict(fmmob=fmmob_model, SF=SF_model, sfbet=sfbet_model,
                   nw=nw_model, ng=ng_model, krw0=krw0_model, krg0=krg0_model)
    mobw = model0.water_mobility(params0, Sw)
    mobg = model0.gas_mobility(params0, Sw)
    MRF = 1.0
    mu_app_exp_aux = 1.0 / (mobw + (mobg / MRF))
    fg_aux = (mobg / MRF) * mu_app_exp_aux

    mu_app_exp = []
    fg_exp = []
    target_points = [0.4, 0.6, 0.7, 0.8, 0.9]
    closest_indices = model0.find_closest_indices(fg_aux, target_points)
    for target, closest_index in zip(target_points, closest_indices):
        fg_exp.append(fg_aux[closest_index])
        mu_app_exp.append(mu_app_exp_aux[closest_index])

    model2 = Model(fg_exp, mu_app_exp, Swc, Sgr, mi_w, mi_g,
                   nw_model, ng_model, krw0_model, krg0_model, fmmob_model, SF_model, sfbet_model)
    model = Model(fg_aux, mu_app_exp_aux, Swc, Sgr, mi_w, mi_g, nw_model,
                  ng_model, krw0_model, krg0_model, fmmob_model, SF_model, sfbet_model)
    plt.plot(fg_exp, mu_app_exp, 'o', label='Experimental data')
    plt.legend(loc='best')
    plt.xlabel(r'$f_g$')
    plt.ylabel(r'$\mu_{app}$')
    plt.savefig('mu_app_cen1a.pdf', format='pdf', dpi=300)
    plt.show()

    name_arq = "scenario1_mu_app.csv"
    with open(name_arq, mode='w', newline='') as arq_csv:
        csv_writer = csv.writer(arq_csv, delimiter='\t')
        csv_writer.writerow(["f_g", "mu_{app}"])
        for fg, mu_app in zip(fg_exp, mu_app_exp):
            csv_writer.writerow([fg, mu_app])

    swx_exp = model2.Sw(nw_model, krw0_model)

    plt.plot(fg_exp, swx_exp, 'ko', label='Data')
    plt.xlabel(r'$f_g$', fontsize=15)
    plt.ylabel(r'$S_{w}$', fontsize=15)
    plt.legend(loc='best')
    plt.savefig('saturation_cen1a.pdf', format='pdf', dpi=300)
    plt.show()

    name_arq = "scenario1_sw.csv"
    with open(name_arq, mode='w', newline='') as arq_csv:
        csv_writer = csv.writer(arq_csv, delimiter='\t')
        csv_writer.writerow(["f_g", "S_w"])
        for fg, sw in zip(fg_exp, swx_exp):
            csv_writer.writerow([fg, sw])
