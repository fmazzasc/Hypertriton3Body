import math
import os
from concurrent.futures import ThreadPoolExecutor
from math import floor, log10

import aghast
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from hipe4ml.model_handler import ModelHandler

import ROOT
from ROOT import TF1, TH1D, TH2D, TH3D, TCanvas, TPaveStats, TPaveText, gStyle


def apply_on_large_data(tree_path, cent_classes, pt_bins, ct_bins, training_columns, split=''):

    handlers_path = '../Models/handlers'
    efficiencies_path = '../Results/Efficiencies'


    executor = ThreadPoolExecutor(max_workers=64)
    iterator = uproot.iterate(tree_path, executor=executor, library="pd")

    df_list = []

    for data in iterator:
        rename_df_columns(data)
    
        # print('current file: {}'.format(current_file))
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        # print(data.head())
        
        for cclass in cent_classes:
            for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
                for ctbin in zip(ct_bins[:-1], ct_bins[1:]):
                    info_string = '_{}{}_{}{}_{}{}'.format(cclass[0], cclass[1], ptbin[0], ptbin[1], ctbin[0], ctbin[1])

                    filename_handler = handlers_path + '/model_handler' + info_string + split + '.pkl'
                    filename_efficiencies = efficiencies_path + '/Eff_Score' + info_string + split + '.npy'

                    model_handler = ModelHandler()
                    model_handler.load_model_handler(filename_handler)

                    eff_score_array = np.load(filename_efficiencies)
                    tsd = eff_score_array[1][-1]

                    data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {ptbin[0]}<pt<{ptbin[1]} and {cclass[0]}<=centrality<{cclass[1]}'

                    df_tmp = data.query(data_range)
                    df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))

                    df_tmp = df_tmp.query('score>@tsd')
                    df_list.append(df_tmp)

    df_applied = pd.concat(df_list)
    print("\n******** Dataframe head ********")
    print(df_applied.head(10))
    return df_applied
    

def expected_signal_counts(bw, cent_range, pt_range, eff, nevents, n_body=2):
    correction = 0.4  # Very optimistic, considering it constant with centrality

    if n_body == 2:
        correction *= 0.25
        
    if n_body == 3:
        correction *= 0.4

    cent_bins = [10, 40, 90]

    signal = 0
    for cent in range(cent_range[0]+1, cent_range[1]):
        for index in range(0, 3):
            if cent < cent_bins[index]:
                signal = signal + \
                    nevents[cent] * \
                    bw[index].Integral(pt_range[0], pt_range[1], 1e-8)
                break

    return int(round(2*signal * eff * correction))


def significance_error(signal, background):
    signal_error = np.sqrt(signal + 1e-10)
    background_error = np.sqrt(background + 1e-10)

    sb = signal + background + 1e-10
    sb_sqrt = np.sqrt(sb)

    s_propag = (sb_sqrt + signal / (2 * sb_sqrt))/sb * signal_error
    b_propag = signal / (2 * sb_sqrt)/sb * background_error

    if signal+background == 0:
        return 0

    return np.sqrt(s_propag * s_propag + b_propag * b_propag)


def expo(x, tau):
    return np.exp(-x / (tau * 0.029979245800))


def h2_preselection_efficiency(ptbins, ctbins, name='PreselEff'):
    th2 = TH2D(name, ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Preselection efficiency',
               len(ptbins) - 1, np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_generated(ptbins, ctbins, name='Generated'):
    th2 = TH2D(name, ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm); Generated', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_rawcounts(ptbins, ctbins, name='RawCounts', suffix=''):
    th2 = TH2D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_significance(ptbins, ctbins, name='Significance', suffix=''):
    th2 = TH2D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Significance', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h1_invmass(counts, cent_class, pt_range, ct_range, name=''):
    ghist = aghast.from_numpy(counts)
    th1 = aghast.to_root(ghist, f'ct{ct_range[0]}{ct_range[1]}_pT{pt_range[0]}{pt_range[1]}_cen{cent_class[0]}{cent_class[1]}_{name}')
    th1.SetDirectory(0)
    return th1


def round_to_error(x, error):
    return round(x, -int(floor(log10(abs(error)))))


def get_ptbin_index(th2, ptbin):
    return th2.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))


def get_ctbin_index(th2, ctbin):
    return th2.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))



def rename_df_columns(df):
    rename_dict = {}

    for col in df.columns:

        if col.endswith('_f'):
            rename_dict[col] = col[:-2]
        if col.startswith('tpc_clus'):
            rename_dict[col] = 'tpc_ncls' + col[-5:-2]
        if col== 'tpc_sig_pi_f':
            rename_dict[col] = 'tpc_nsig_pi'
    
    df.rename(columns = rename_dict, inplace=True)


def ndarray2roo(ndarray, var):
    if isinstance(ndarray, ROOT.RooDataSet):
        print('Already a RooDataSet')
        return ndarray

    assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
    assert len(ndarray.shape) == 1, 'Can only handle 1d array'

    name = var.GetName()
    x = np.zeros(1, dtype=np.float64)

    tree = ROOT.TTree('tree', 'tree')
    tree.Branch(f'{name}', x ,f'{name}/D')

    for i in ndarray:
        x[0] = i
        tree.Fill()

    array_roo = ROOT.RooDataSet('data', 'dataset from tree', tree, ROOT.RooArgSet(var))
    return array_roo
