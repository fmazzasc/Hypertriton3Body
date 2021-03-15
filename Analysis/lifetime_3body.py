#!/usr/bin/env python3
import ROOT
ROOT.gROOT.LoadMacro("helpers/fit_macros/comb_fit_gaus.C")
from ROOT import comb_fit_gaus
ROOT.gROOT.LoadMacro("helpers/fit_macros/comb_fit_erf.C")
from ROOT import comb_fit_erf
ROOT.gROOT.SetBatch()

import sys
sys.path.append('helpers/')
import argparse
import math
import os
import numpy as np
import pandas as pd
import yaml
from scipy import stats
import uproot
import hyp_plot_utils as hpu


np.random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-s', '--significance', help='Use the BDTefficiency selection from the significance scan', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define some globals
FILE_PREFIX = params['FILE_PREFIX']

DATA_PATH = os.path.expandvars(params["APPLICATION_PATHS"]['DATA_PATH'])

SPLIT_LIST = ['_matter','_antimatter'] if args.split else ['']
BKG_MODELS = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['expo']

CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX, EFF_STEP), 2)

SPLIT_MODE = args.split
SIGNIFICANCE_SCAN = args.significance

FIX_EFF = 0.7 if not SIGNIFICANCE_SCAN else 0

SYSTEMATICS_COUNTS = 100000
###############################################################################

###############################################################################
# input/output files
results_dir = "../Results"
tables_dir = "../Tables"
efficiency_dir = "../Results/Efficiencies"
utils_dir = "../Utils"
mcsigma_dir = utils_dir + '/FixedSigma'

# input data file
file_name = tables_dir + f'/pass1/applied_df_ct_analysis_data.parquet.gzip'
data_df = pd.read_parquet(file_name)
file_name = tables_dir + f'/pass1/applied_df_ct_analysis_ls.parquet.gzip'
ls_df = pd.read_parquet(file_name)

# output file
file_name = results_dir + f'/{FILE_PREFIX}_dist.root'
output_file = ROOT.TFile(file_name, 'recreate')

# preselection eff
file_name = efficiency_dir + f'/preseleff_cent090.root'
efficiency_file = ROOT.TFile(file_name, 'read')
EFFICIENCY = efficiency_file.Get('PreselEff').ProjectionY()

# absorption correction file
# file_name = utils_dir + '/he3abs/recCtHe3.root'
# abs_file = ROOT.TFile(file_name, 'read')
# ABSORPTION = abs_file.Get('Reconstructed ct spectrum')

# significance scan output
file_name = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
sigscan_dict = np.load(file_name, allow_pickle=True).item()
###############################################################################

###############################################################################
# define support globals and methods for getting hypertriton counts
RAW_COUNTS_H2 = {}
RAW_COUNTS_BEST = {}

CORRECTED_COUNTS_H2 = {}
CORRECTED_COUNTS_BEST = {}

for model in BKG_MODELS:
    # initialize histos for the hypertriton counts vs ct vs BDT efficiency
    RAW_COUNTS_H2[model] = ROOT.TH2D(f'raw_counts_{model}', '', len(CT_BINS)-1, np.array(CT_BINS, 'double'), len(EFF_ARRAY)-1, np.array(EFF_ARRAY, 'double'))
    RAW_COUNTS_BEST[model] = RAW_COUNTS_H2[model].ProjectionX(f'raw_counts_best_{model}')

    CORRECTED_COUNTS_H2[model] = RAW_COUNTS_H2[model].Clone(f'corrected_counts_{model}')
    CORRECTED_COUNTS_BEST[model] = RAW_COUNTS_BEST[model].Clone(f'corrected_counts_best_{model}')


def get_presel_eff(ctbin):
    return EFFICIENCY.GetBinContent(EFFICIENCY.FindBin((ctbin[0] + ctbin[1]) / 2))


def get_absorption_correction(ctbin):
    bin_idx = ABSORPTION.FindBin((ctbin[0] + ctbin[1]) / 2)
    return 1 - ABSORPTION.GetBinContent(bin_idx)


def fill_raw(bkg, ctbin, counts, counts_err, eff):
    print(counts, ctbin, eff)
    bin_idx = RAW_COUNTS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, eff + 0.005)
    RAW_COUNTS_H2[bkg].SetBinContent(bin_idx, counts)
    RAW_COUNTS_H2[bkg].SetBinError(bin_idx, counts_err)


def fill_raw_best(bkg, ctbin, counts, counts_err, eff):
    bin_idx = RAW_COUNTS_BEST[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)
    RAW_COUNTS_BEST[bkg].SetBinContent(bin_idx, counts)
    RAW_COUNTS_BEST[bkg].SetBinError(bin_idx, counts_err)


def fill_corrected(bkg, ctbin, counts, counts_err, eff):
    bin_idx = CORRECTED_COUNTS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, eff + 0.005)
    bin_idx1d = CORRECTED_COUNTS_BEST[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)
    # abs_corr = get_absorption_correction(ctbin)
    abs_corr=1
    presel_eff = get_presel_eff(ctbin)
    bin_width = CORRECTED_COUNTS_BEST[bkg].GetBinWidth(bin_idx1d)
    CORRECTED_COUNTS_H2[bkg].SetBinContent(bin_idx, counts/eff/presel_eff/abs_corr/bin_width)
    CORRECTED_COUNTS_H2[bkg].SetBinError(bin_idx, counts_err/eff/presel_eff/abs_corr/bin_width)
    

def fill_corrected_best(bkg, ctbin, counts, counts_err, eff):
    bin_idx = CORRECTED_COUNTS_BEST[bkg].FindBin((ctbin[0] + ctbin[1]) / 2)
    # abs_corr = get_absorption_correction(ctbin)
    abs_corr=1
    presel_eff = get_presel_eff(ctbin)
    bin_width = CORRECTED_COUNTS_BEST[bkg].GetBinWidth(bin_idx)
    CORRECTED_COUNTS_BEST[bkg].SetBinContent(bin_idx, counts/eff/presel_eff/abs_corr/bin_width)
    CORRECTED_COUNTS_BEST[bkg].SetBinError(bin_idx, counts_err/eff/presel_eff/abs_corr/bin_width)


def get_signscan_eff(ctbin):
    key = f'ct{ctbin[0]}{ctbin[1]}pt{PT_BINS[0]}{PT_BINS[1]}'
    return sigscan_dict[key]
    

def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)
    return int(idx)


def get_corrected_counts(bkg, ctbin, eff):
    bin_idx = CORRECTED_COUNTS_H2[bkg].FindBin((ctbin[0] + ctbin[1]) / 2, eff + 0.005)
    counts = CORRECTED_COUNTS_H2[bkg].GetBinContent(bin_idx)
    error = CORRECTED_COUNTS_H2[bkg].GetBinError(bin_idx)
    
    return counts, error


def get_effscore_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'
    return {round(e[0], 2): e[1] for e in np.load(file_name).T}

def get_mcsigma_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = mcsigma_dir + f'/sigma_array_{info_string}.npy'

    tmp_dict = np.load(file_name, allow_pickle=True).item()
    
    return {round(float(s), 2): tmp_dict[s] for s in tmp_dict}


def h1_invmass(counts, mass_range=[2.96, 3.04] , bins=34, name=''):
    th1 = ROOT.TH1D(f'{name}', f'{name}_x', int(bins), mass_range[0], mass_range[1])
    for index in range(0, len(counts)):
        th1.SetBinContent(index+1, counts[index])
        # th1.SetBinError(index + 1, np.sqrt(counts[index]))
    th1.SetDirectory(0)
    return th1

###############################################################################

# significance-scan/fixed efficiencies switch
if not SIGNIFICANCE_SCAN:
    eff_best_array = np.full(len(CT_BINS) - 1, FIX_EFF)
    # eff_best_array = np.array([0.7, 0.7, 0.5, 0.5, 0.5, 0.5])
else:
    eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(CT_BINS[:-1], CT_BINS[1:])]
    # eff_best_array = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

syst_eff_ranges = [list(range(int(x * 100) - 10, int(x * 100) + 11)) for x in eff_best_array]
eff_best_it = iter(eff_best_array)

for split in SPLIT_LIST:
    for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
        score_dict = get_effscore_dict(ctbin)
        mcsigma_dict = get_mcsigma_dict(ctbin)

        # get data slice for this ct bin
        data_slice_ct = data_df.query('@ctbin[0]<ct<@ctbin[1]')
        ls_slice_ct = ls_df.query('@ctbin[0]<ct<@ctbin[1]')

        subdir_name = f'ct{ctbin[0]}{ctbin[1]}'
        ct_dir = output_file.mkdir(subdir_name)
        ct_dir.cd()

        eff_best = next(eff_best_it)
        for eff in EFF_ARRAY:
            # define global RooFit objects
            mcsigma = mcsigma_dict[eff]
            mcsigma=-1
               
            # get the data slice as a RooDataSet
            tsd = score_dict[eff]
            n_bins = 38
            mass_range = [2.97, 3.03]
    
            data_selected = data_slice_ct.query('score>@tsd')
            ls_selected = ls_slice_ct.query('score>@tsd')

            data_selected_dalitz = data_selected.query("2.991 - 0.002 < m < 2.991 + 0.002 ")
            hpu.dalitz_plot(data_selected_dalitz["mppi"], data_selected_dalitz["mdpi"], eff=eff, ct_bin=ctbin,
                            x_axis=[55,1.16,1.2599], y_axis=[55,4.07,4.2199], x_label='m($p \pi$) [GeV$^2$/c$^4$]',
                            y_label='m($d \pi$) [GeV$^2$/c$^4$]')

            selected_data_counts, bins = np.histogram(data_selected['m'], bins=n_bins, range=mass_range)
            selected_ls_counts,_ = np.histogram(ls_selected['m'], bins=n_bins, range=mass_range)

            selected_data_hist = h1_invmass(selected_data_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_data')
            selected_ls_hist = h1_invmass(selected_ls_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_ls')

            ##superimposed histos
            cv_sup = ROOT.TCanvas(f"sig_and_bkg_{eff}")
            selected_data_hist.Draw("PE SAME")
            selected_ls_hist.Draw("PE SAME")
            selected_ls_hist.SetMarkerColor(ROOT.kRed)
            selected_ls_hist.SetLineColor(ROOT.kRed)

            cv_sup.Write()



            # define signal parameters
            raw_counts = comb_fit_gaus(selected_ls_hist, selected_data_hist, f"comb_fit_{eff}_gaus", mass_range[0], mass_range[1], mcsigma)
            # raw_counts_2 = comb_fit_erf(selected_ls_hist, selected_data_hist, f"comb_fit_{eff}_erf", mass_range[0], mass_range[1], mcsigma)



            raw_counts_err = np.sqrt(raw_counts)
            # fill the measured hypertriton counts histograms
            fill_raw(model, ctbin, raw_counts, raw_counts_err, eff)
            fill_corrected(model, ctbin, raw_counts, raw_counts_err, eff)
            if eff == eff_best:
                fill_raw_best(model, ctbin, raw_counts, raw_counts_err, eff)
                fill_corrected_best(model, ctbin, raw_counts, raw_counts_err, eff)


expo = ROOT.TF1('myexpo', '[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)', 2, 14)
expo.SetParLimits(1, 100, 5000)

kBlueC = ROOT.TColor.GetColor('#1f78b4')
kBlueCT = ROOT.TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = ROOT.TColor.GetColor('#e31a1c')
kRedCT = ROOT.TColor.GetColorTransparent(kRedC, 0.5)

output_file.cd()
RAW_COUNTS_H2[model].Write()
RAW_COUNTS_BEST[model].Write()
CORRECTED_COUNTS_H2[model].Write()
CORRECTED_COUNTS_BEST[model].Write()
CORRECTED_COUNTS_BEST[model].UseCurrentStyle()
CORRECTED_COUNTS_BEST[model].Fit(expo, 'MEI0+', '', 2, 14)
fit_function = CORRECTED_COUNTS_BEST[model].GetFunction('myexpo')
fit_function.SetLineColor(kRedC)

canvas = ROOT.TCanvas(f'ct_spectra_{model}')
canvas.SetLogy()
frame = ROOT.gPad.DrawFrame(-0.5, 1, 35.5, 2000, ';#it{c}t (cm);d#it{N}/d(#it{c}t) [(cm)^{-1}]')
pinfo = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, 'NDC')
pinfo.SetBorderSize(0)
pinfo.SetFillStyle(0)
pinfo.SetTextAlign(22)
pinfo.SetTextFont(43)
pinfo.SetTextSize(22)
strings = []
strings.append('#bf{ALICE Internal}')
strings.append('Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%')
strings.append(f'#tau = {fit_function.GetParameter(1):.0f} #pm {fit_function.GetParError(1):.0f} ps')
strings.append(f'#chi^{{2}} / NDF = {(fit_function.GetChisquare() / fit_function.GetNDF()):.2f}')
for s in strings:
    pinfo.AddText(s)
fit_function.Draw('same')
CORRECTED_COUNTS_BEST[model].Draw('ex0same')
CORRECTED_COUNTS_BEST[model].SetMarkerStyle(20)
CORRECTED_COUNTS_BEST[model].SetMarkerColor(kBlueC)
CORRECTED_COUNTS_BEST[model].SetLineColor(kBlueC)
CORRECTED_COUNTS_BEST[model].SetMinimum(0.001)
CORRECTED_COUNTS_BEST[model].SetMaximum(1000)
CORRECTED_COUNTS_BEST[model].SetStats(0)
frame.GetYaxis().SetRangeUser(7, 5000)
frame.GetXaxis().SetRangeUser(0.5, 35.5)
pinfo.Draw('x0same')
canvas.Write()

output_file.Close()

print(eff_best_array)