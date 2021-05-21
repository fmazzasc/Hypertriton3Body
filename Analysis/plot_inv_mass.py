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

args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# TRAINING_DIR = params["TRAINING_DIR"]
TRAINING_DIR = params["TRAINING_DIR"]
CT_BINS = params['CT_BINS']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX, EFF_STEP), 2)


###############################################################################

###############################################################################
# input/output files
results_dir = "../Results/" +  TRAINING_DIR
tables_dir = "../Tables"
efficiency_dir = "../Results/Efficiencies/" + TRAINING_DIR
utils_dir = "../Utils"


# input data file
data_path = results_dir + "/applied_df_data.parquet.gzip"
ls_rot_path = results_dir + "/applied_df_ls.parquet.gzip"
ls_old_path = results_dir + "/applied_df_ls_old.parquet.gzip"

data_df = pd.read_parquet(data_path)
ls_rot_df = pd.read_parquet(ls_rot_path)
ls_new_df = pd.read_parquet(ls_rot_path).query("rotation==0")
ls_old_df = pd.read_parquet(ls_old_path)



# output file
file_name = results_dir + '/inv_mass_comparison.root'
output_file = ROOT.TFile(file_name, 'recreate')




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



def h1_invmass(counts, mass_range=[2.96, 3.04] , bins=34, name=''):
    th1 = ROOT.TH1D(f'{name}', f'{name}_x', int(bins), mass_range[0], mass_range[1])
    for index in range(0, len(counts)):
        th1.SetBinContent(index+1, counts[index])
        # th1.SetBinError(index + 1, np.sqrt(counts[index]))
    th1.SetDirectory(0)
    return th1

###############################################################################



for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
    print("-------------------------")
    print(f"{ctbin[0]} < ct < {ctbin[1]} ....")

    score_dict = get_effscore_dict(ctbin)

    # get data slice for this ct bin
    data_slice_ct = data_df.query('@ctbin[0]<ct<@ctbin[1]')
    ls_old_slice_ct = ls_old_df.query('@ctbin[0]<ct<@ctbin[1]')
    ls_new_slice_ct = ls_new_df.query('@ctbin[0]<ct<@ctbin[1]')
    ls_rot_slice_ct = ls_rot_df.query('@ctbin[0]<ct<@ctbin[1]')


    subdir_name = f'ct{ctbin[0]}{ctbin[1]}'
    ct_dir = output_file.mkdir(subdir_name)
    ct_dir.cd()

    for eff in EFF_ARRAY:
        # define global RooFit objects
        # get the data slice as a RooDataSet
        tsd = score_dict[eff]
        n_bins = 50
        mass_range = [2.97, 3.02]

        data_selected = data_slice_ct.query('score>@tsd')
        ls_old_selected = ls_old_slice_ct.query('score>@tsd')
        ls_new_selected = ls_new_slice_ct.query('score>@tsd')
        ls_rot_selected = ls_rot_slice_ct.query('score>@tsd')


        selected_data_counts, bins = np.histogram(data_selected['m'], bins=n_bins, range=mass_range)

        selected_ls_rot_counts,_ = np.histogram(ls_rot_selected['m'], bins=n_bins, range=mass_range)
        selected_ls_rot_counts = selected_ls_rot_counts*normalize_ls(selected_data_counts, selected_ls_rot_counts, bins)

        selected_ls_old_counts,_ = np.histogram(ls_old_selected['m'], bins=n_bins, range=mass_range)
        selected_ls_old_counts = selected_ls_old_counts*normalize_ls(selected_data_counts, selected_ls_old_counts, bins)

        selected_ls_new_counts,_ = np.histogram(ls_new_selected['m'], bins=n_bins, range=mass_range)
        selected_ls_new_counts = selected_ls_new_counts*normalize_ls(selected_data_counts, selected_ls_new_counts, bins)

        selected_data_hist = h1_invmass(selected_data_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_data')
        selected_ls_rot_hist = h1_invmass(selected_ls_rot_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_ls_rot')
        selected_ls_old_hist = h1_invmass(selected_ls_old_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_ls_old')
        selected_ls_new_hist = h1_invmass(selected_ls_new_counts, mass_range=mass_range, bins=n_bins, name=f'eff_{eff}_ls_new')



        ##superimposed histos
        cv_sup = ROOT.TCanvas(f"sig_and_bkg_{eff}")
        selected_data_hist.SetStats(0)
        selected_data_hist.Draw("PE SAME")
        selected_ls_rot_hist.Draw("L SAME")
        selected_ls_rot_hist.SetMarkerColor(ROOT.kRed)
        selected_ls_rot_hist.SetLineColor(ROOT.kRed)
        selected_ls_old_hist.Draw("L SAME")
        selected_ls_old_hist.SetMarkerColor(ROOT.kGreen)
        selected_ls_old_hist.SetLineColor(ROOT.kGreen)

        selected_ls_new_hist.Draw("L SAME")
        selected_ls_new_hist.SetMarkerColor(ROOT.kOrange)
        selected_ls_new_hist.SetLineColor(ROOT.kOrange)

        leg = ROOT.TLegend(0.58,0.7,0.93,0.9)
        leg.AddEntry(selected_data_hist, "Data")
        leg.AddEntry(selected_ls_rot_hist, "LS + Rotation")
        leg.AddEntry(selected_ls_old_hist, "LS Old")
        leg.AddEntry(selected_ls_new_hist, "LS New")

        leg.Draw()

        cv_sup.Write()

    print("Done.")
    print("-------------------------")

output_file.Close()