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
import hyp_analysis_utils as hau

np.random.seed(42)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-syst', '--syst', help='Compute systematic uncertainties', action='store_true')
parser.add_argument('-fixed', '--eff_fixed', help='Compute systematic uncertainties', action='store_true')
parser.add_argument('-s', '--significance', help='Use the BDT efficiency selection from the significance scan', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

SYSTEMATICS = args.syst
FIXED_EFF_SYSTEMATICS = args.eff_fixed
SIGNIFICANCE_SCAN = args.significance

# TRAINING_DIR = params["TRAINING_DIR"]
TRAINING_DIR = params["TRAINING_DIR"]
CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX, EFF_STEP), 2)
FIX_EFF = 0.6 if not SIGNIFICANCE_SCAN else 0
SYSTEMATICS_COUNTS = 10000

###############################################################################

###############################################################################
# input/output files
results_dir = "../Results/" +  TRAINING_DIR
tables_dir = "../Tables"
efficiency_dir = "../Results/Efficiencies/" + TRAINING_DIR
utils_dir = "../Utils"
mcsigma_dir = utils_dir + '/FixedSigma'

# input data file
data_path = results_dir + "/applied_df_data.parquet.gzip"
ls_path = results_dir + "/applied_df_ls.parquet.gzip"
em_path = results_dir + "/applied_df_em.parquet.gzip"
ls_pion_path = results_dir + "/applied_df_ls_pion.parquet.gzip"

data_df = pd.read_parquet(data_path)
ls_df = pd.read_parquet(ls_path)

# output file
file_name = results_dir + '/ct_analysis_results_0.6.root'
output_file = ROOT.TFile(file_name, 'recreate')

# preselection eff
file_name = efficiency_dir + f'/preseleff_cent090.root'
efficiency_file = ROOT.TFile(file_name, 'read')
EFFICIENCY = efficiency_file.Get('PreselEff').ProjectionY()


# significance scan output
# file_name = efficiency_dir + '/sigscan.npy'
# sigscan_dict = np.load(file_name, allow_pickle=True).item()
###############################################################################

###############################################################################

RAW_COUNTS_H2= ROOT.TH2D('raw_counts', '', len(CT_BINS)-1, np.array(CT_BINS, 'double'), len(EFF_ARRAY)-1, np.array(EFF_ARRAY, 'double'))
RAW_COUNTS_BEST = RAW_COUNTS_H2.ProjectionX(f'raw_counts_best')
CORRECTED_COUNTS_H2 = RAW_COUNTS_H2.Clone(f'corrected_counts')
CORRECTED_COUNTS_BEST = RAW_COUNTS_BEST.Clone(f'corrected_counts_best')


def get_presel_eff(ctbin):
    return EFFICIENCY.GetBinContent(EFFICIENCY.FindBin((ctbin[0] + ctbin[1]) / 2))


def get_absorption_correction(ctbin):
    bin_idx = ABSORPTION.FindBin((ctbin[0] + ctbin[1]) / 2)
    return 1 - ABSORPTION.GetBinContent(bin_idx)


def fill_raw(ctbin, counts, counts_err, eff):
    print(counts, ctbin, eff)
    bin_idx = RAW_COUNTS_H2.FindBin((ctbin[0] + ctbin[1]) / 2, eff + 0.005)
    RAW_COUNTS_H2.SetBinContent(bin_idx, counts)
    RAW_COUNTS_H2.SetBinError(bin_idx, counts_err)


def fill_raw_best(ctbin, counts, counts_err, eff):
    bin_idx = RAW_COUNTS_BEST.FindBin((ctbin[0] + ctbin[1]) / 2)
    RAW_COUNTS_BEST.SetBinContent(bin_idx, counts)
    RAW_COUNTS_BEST.SetBinError(bin_idx, counts_err)


def fill_corrected(ctbin, counts, counts_err, eff):
    bin_idx = CORRECTED_COUNTS_H2.FindBin((ctbin[0] + ctbin[1]) / 2, eff + 0.005)
    bin_idx1d = CORRECTED_COUNTS_BEST.FindBin((ctbin[0] + ctbin[1]) / 2)
    # abs_corr = get_absorption_correction(ctbin)
    abs_corr=1
    presel_eff = get_presel_eff(ctbin)
    bin_width = CORRECTED_COUNTS_BEST.GetBinWidth(bin_idx1d)
    CORRECTED_COUNTS_H2.SetBinContent(bin_idx, counts/eff/presel_eff/abs_corr/bin_width)
    CORRECTED_COUNTS_H2.SetBinError(bin_idx, counts_err/eff/presel_eff/abs_corr/bin_width)
    

def fill_corrected_best(ctbin, counts, counts_err, eff):
    print(counts)
    bin_idx = CORRECTED_COUNTS_BEST.FindBin((ctbin[0] + ctbin[1]) / 2)
    # abs_corr = get_absorption_correction(ctbin)
    abs_corr=1
    presel_eff = get_presel_eff(ctbin)
    bin_width = CORRECTED_COUNTS_BEST.GetBinWidth(bin_idx)
    print(counts, eff, presel_eff, bin_width)
    CORRECTED_COUNTS_BEST.SetBinContent(bin_idx, counts/eff/presel_eff/abs_corr/bin_width)
    CORRECTED_COUNTS_BEST.SetBinError(bin_idx, counts_err/eff/presel_eff/abs_corr/bin_width)


def get_signscan_eff(ctbin):
    key = f'ct{ctbin[0]}{ctbin[1]}pt{PT_BINS[0]}{PT_BINS[1]}'
    return sigscan_dict[key]
    

def get_eff_index(eff):
    idx = (eff - EFF_MIN + EFF_STEP) * 100
    if isinstance(eff, np.ndarray):
        return idx.astype(int)
    return int(idx)


def get_corrected_counts(ctbin, eff):
    bin_idx = CORRECTED_COUNTS_H2.FindBin((ctbin[0] + ctbin[1]) / 2, eff + 0.005)
    counts = CORRECTED_COUNTS_H2.GetBinContent(bin_idx)
    error = CORRECTED_COUNTS_H2.GetBinError(bin_idx)
    
    return counts, error

def normalize_ls(data_counts, ls_counts, bins):
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    side_region = np.logical_or(bin_centers<2.992-2*0.0025, bin_centers>2.992+2*0.0025)
    
    side_data_counts = np.sum(data_counts[side_region])
    side_ls_counts = np.sum(ls_counts[side_region])
    scaling_factor = side_data_counts/side_ls_counts
    return scaling_factor

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
    print(eff_best_array)
else:
    eff_best_array = [round(sigscan_dict[f'ct{ctbin[0]}{ctbin[1]}pt210'][0], 2) for ctbin in zip(CT_BINS[:-1], CT_BINS[1:])]

syst_eff_ranges = np.asarray([list(range(int(x * 100) - 10, int(x * 100) + 11)) for x in eff_best_array]) / 100 
eff_best_it = iter(eff_best_array)


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
        n_bins = 50
        mass_range = [2.97, 3.02]

        data_selected = data_slice_ct.query('score>@tsd')
        ls_selected = ls_slice_ct.query('score>@tsd')

        data_selected_dalitz = data_selected.query("2.991 - 0.002 < m < 2.991 + 0.002 ")
        hpu.dalitz_plot(data_selected_dalitz["mppi"], data_selected_dalitz["mdpi"], eff=eff, ct_bin=ctbin,
                        x_axis=[45,1.16,1.2599], y_axis=[45,4.07,4.2199], x_label='m($p \pi$) [GeV$^2$/c$^4$]',
                        y_label='m($d \pi$) [GeV$^2$/c$^4$]', training_dir=TRAINING_DIR)

        selected_data_counts, bins = np.histogram(data_selected['m'], bins=n_bins, range=mass_range)
        selected_ls_counts,_ = np.histogram(ls_selected['m'], bins=n_bins, range=mass_range)
        selected_ls_counts = selected_ls_counts*normalize_ls(selected_data_counts, selected_ls_counts, bins)

        selected_data_hist = h1_invmass(selected_data_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_data')
        selected_ls_hist = h1_invmass(selected_ls_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_ls')
        subtr_hist = h1_invmass(selected_data_counts - selected_ls_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_ls')
        

        ##superimposed histos
        cv_sup = ROOT.TCanvas(f"sig_and_bkg_{eff}")
        selected_data_hist.Draw("PE SAME")
        selected_ls_hist.Draw("PE SAME")
        selected_ls_hist.SetMarkerColor(ROOT.kRed)
        selected_ls_hist.SetLineColor(ROOT.kRed)

        cv_sup.Write()

            # define signal parameters
        raw_counts = hau.fit_hist(subtr_hist, [0,90], [2,10], ctbin, nsigma=3, model="pol0", fixsigma=-1, sigma_limits=None, mode=3)
        raw_counts = raw_counts[0]
        print(raw_counts)
        # raw_counts = comb_fit_erf(selected_ls_hist, selected_data_hist, f"comb_fit_{eff}_erf", mass_range[0], mass_range[1], mcsigma)



        raw_counts_err = np.sqrt(raw_counts)
        # fill the measured hypertriton counts histograms
        fill_raw(ctbin, raw_counts, raw_counts_err, eff)
        fill_corrected(ctbin, raw_counts, raw_counts_err, eff)
        if eff == eff_best:
            fill_raw_best(ctbin, raw_counts, raw_counts_err, eff)
            fill_corrected_best(ctbin, raw_counts, raw_counts_err, eff)


expo = ROOT.TF1('myexpo', '[0]*exp(-x/([1]*0.029979245800))/([1]*0.029979245800)', 2, 35)
expo.SetParLimits(1, 100, 5000)

kBlueC = ROOT.TColor.GetColor('#1f78b4')
kBlueCT = ROOT.TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = ROOT.TColor.GetColor('#e31a1c')
kRedCT = ROOT.TColor.GetColorTransparent(kRedC, 0.5)

output_file.cd()
RAW_COUNTS_H2.Write()
RAW_COUNTS_BEST.Write()
CORRECTED_COUNTS_H2.Write()
CORRECTED_COUNTS_BEST.Write()
CORRECTED_COUNTS_BEST.UseCurrentStyle()
CORRECTED_COUNTS_BEST.Fit(expo, 'MEI0+', '', 2, 14)
fit_function = CORRECTED_COUNTS_BEST.GetFunction('myexpo')
fit_function.SetLineColor(kRedC)

canvas = ROOT.TCanvas(f'ct_spectrum')
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
CORRECTED_COUNTS_BEST.Draw('ex0same')
CORRECTED_COUNTS_BEST.SetMarkerStyle(20)
CORRECTED_COUNTS_BEST.SetMarkerColor(kBlueC)
CORRECTED_COUNTS_BEST.SetLineColor(kBlueC)
CORRECTED_COUNTS_BEST.SetMinimum(0.001)
CORRECTED_COUNTS_BEST.SetMaximum(1000)
CORRECTED_COUNTS_BEST.SetStats(0)
frame.GetYaxis().SetRangeUser(7, 5000)
frame.GetXaxis().SetRangeUser(0.5, 35.5)
pinfo.Draw('x0same')
canvas.Write()



if SYSTEMATICS:
    # systematics histos
    lifetime_dist = ROOT.TH1D('syst_lifetime', ';#tau ps ;counts', 100, 100, 400)
    lifetime_prob = ROOT.TH1D('prob_lifetime', ';prob. ;counts', 100, 0, 1)

    tmp_ctdist = CORRECTED_COUNTS_BEST.Clone('tmp_ctdist')

    combinations = [] if FIXED_EFF_SYSTEMATICS else set()
    sample_counts = 0   # good fits
    iterations = 0  # total fits

    count_low = 0
    count_up = 0
    eff_low = []
    eff_up = []

    # stop with SYSTEMATICS_COUNTS number of good B_{Lambda} fits
    while sample_counts < SYSTEMATICS_COUNTS:
        tmp_ctdist.Reset()

        iterations += 1

        bkg_list = []
        eff_list = []
        eff_idx_list = []

        # loop over ctbins
        if FIXED_EFF_SYSTEMATICS:
            eff_list = np.random.choice(syst_eff_ranges[1])*np.ones(len(CT_BINS)-1)
            combo = eff_list[0]
            print("COMBO: ",  combo)


        else:
            for ctbin_idx in range(len(CT_BINS)-1):
                # random bkg model
                # random BDT efficiency in the defined range
                eff = np.random.choice(syst_eff_ranges[ctbin_idx])
                eff_list.append(eff)
                eff_idx = get_eff_index(eff)
                eff_idx_list.append(eff_idx)
            # convert indexes into hash and if already sampled skip this combination
            combo = ''.join(map(str, eff_idx_list))
        
        if sample_counts==len(syst_eff_ranges[1]):
            if FIXED_EFF_SYSTEMATICS:
                break
        
        if combo in combinations:
            continue

        # if indexes are good measure lifetime
        ctbin_idx = 1
        ct_bins = list(zip(CT_BINS[:-1], CT_BINS[1:]))
        print("EFF LIST: ", eff_list)

        for eff in eff_list:
            ctbin = ct_bins[ctbin_idx-1]

            counts, error = get_corrected_counts(ctbin, eff)

            tmp_ctdist.SetBinContent(ctbin_idx, counts)
            tmp_ctdist.SetBinError(ctbin_idx, error)

            ctbin_idx += 1
    
        tmp_ctdist.Fit(expo, 'MEI0+', '', 2, 14)

        if expo.GetParameter(1)>350 and count_up<5:
            tmp_ctdist.SetName(f"lifetime_up_{count_up}")
            tmp_ctdist.Write()
            count_up += 1
            eff_up.append(eff_list)

        if expo.GetParameter(1)<270 and count_low<5:
            tmp_ctdist.SetName(f"lifetime_low_{count_low}")
            tmp_ctdist.Write()
            count_low +=1
            eff_low.append(eff_list)
            

        # if ct fit is good use it for systematics
        if expo.GetChisquare() > 2 * expo.GetNDF():
            continue

        lifetime_dist.Fill(expo.GetParameter(1))
        lifetime_prob.Fill(expo.GetProb())

        combinations.append(combo) if FIXED_EFF_SYSTEMATICS else combinations.add(combo)
        sample_counts += 1
        print("SAMPLE COUNTS:", sample_counts)

    output_file.cd()

    lifetime_dist.Write()
    lifetime_prob.Write()
    print(f"Eff_up: ", eff_up)
    print(f"Eff_low: ", eff_low)

print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
print(f'\nGood iterations / Total iterations -> {SYSTEMATICS_COUNTS/iterations:.4f}')
print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')

output_file.Close()