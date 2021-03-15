import sys
sys.path.append('helpers/')
import ROOT
import argparse
import os
import time
import warnings
import hyp_analysis_utils as hau
import numpy as np
import pandas as pd

import xgboost as xgb
import yaml
from analysis_classes import ModelApplication, TrainingAnalysis
from hipe4ml import analysis_utils
from hipe4ml.model_handler import ModelHandler

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
# ROOT.gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('-a', '--application', help='Apply ML predictions on data', action='store_true')
parser.add_argument('-s', '--significance', help='Run the significance optimisation studies', action='store_true')
parser.add_argument('-side', '--side', help='Use the sideband as background', action='store_true')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define analysis global variables
TRAINING_DIR = params['TRAINING_DIR']

LOAD_APPLIED_DATA = params['LOAD_APPLIED_DATA']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']
COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']

BKG_MODELS = params['BKG_MODELS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

TRAIN = args.train
SPLIT_MODE = args.split
OPTIMIZE = args.optimize
APPLICATION = args.application
SIGNIFICANCE_SCAN = args.significance
SIDEBANDS = args.side
SIGMA_MC = params['SIGMA_MC']
SPLIT_LIST = ['_matter','_antimatter'] if SPLIT_MODE else ['']


###############################################################################
# define paths for loading training data
signal_path = os.path.expandvars(params["TRAINING_PATHS"]['MC_PATH'])
bkg_path = os.path.expandvars(params["TRAINING_PATHS"]['BKG_PATH'])
# define paths for loading application data
data_path = None if params["APPLICATION_PATHS"]['DATA_PATH'] is None else os.path.expandvars(params["APPLICATION_PATHS"]['DATA_PATH'])
ls_path = None if params["APPLICATION_PATHS"]['LS_PATH'] is None else os.path.expandvars(params["APPLICATION_PATHS"]['LS_PATH'])
ls_pion_path = None if params["APPLICATION_PATHS"]['LS_PION_PATH'] is None else  os.path.expandvars(params["APPLICATION_PATHS"]['LS_PION_PATH'])
em_path = None if params["APPLICATION_PATHS"]['EM_PATH'] is None else  os.path.expandvars(params["APPLICATION_PATHS"]['EM_PATH'])
# define paths for loading analysis result
analysis_res_path = os.path.expandvars(params['ANALYSIS_RESULTS_PATH'])
#results dir
results_dir = "../Results/" + TRAINING_DIR
###############################################################################
start_time = time.time()

if TRAIN:
    for split in SPLIT_LIST:
        ml_analysis = TrainingAnalysis(signal_path, bkg_path, split)
        print(f'--- analysis initialized in {((time.time() - start_time) / 60):.2f} minutes ---\n')

        for cclass in CENT_CLASSES:
            ml_analysis.preselection_efficiency(cclass, CT_BINS, PT_BINS, split, TRAINING_DIR)

            for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                    print('\n==================================================')
                    print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)

                    part_time = time.time()

                    # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                    data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)
                    input_model = xgb.XGBClassifier(use_label_encoder=False)
                    model_handler = ModelHandler(input_model)
                    
                    model_handler.set_model_params(MODEL_PARAMS)
                    model_handler.set_model_params(HYPERPARAMS)
                    model_handler.set_training_columns(COLUMNS)

                    if OPTIMIZE:
                        model_handler.optimize_params_bayes(data, HYPERPARAMS_RANGE, 'roc_auc', init_points=30, n_iter=30, njobs=None)

                    model_handler.train_test_model(data)

                    print("train test model")
                    print(f'--- model trained and tested in {((time.time() - part_time) / 60):.2f} minutes ---\n')

                    y_pred = model_handler.predict(data[2])
                    data[2].insert(0, 'score', y_pred)
                    eff, tsd = analysis_utils.bdt_efficiency_array(data[3], y_pred, n_points=1000)
                    score_from_eff_array = analysis_utils.score_from_efficiency_array(data[3], y_pred, FIX_EFF_ARRAY)
                    fixed_eff_array = np.vstack((FIX_EFF_ARRAY, score_from_eff_array))

                    if SIGMA_MC:
                        ml_analysis.MC_sigma_array(data, fixed_eff_array, cclass, ptbin, ctbin, split)

                    ml_analysis.save_ML_analysis(model_handler, fixed_eff_array, cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split=split, training_dir=TRAINING_DIR)
                    ml_analysis.save_ML_plots(model_handler, data, [eff, tsd], cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split=split, training_dir=TRAINING_DIR)


        del ml_analysis

    print('')
    print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')

if APPLICATION:
    app_time = time.time()

    if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    sigscan_results = {}    

    for split in SPLIT_LIST:
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('\nStarting BDT appplication on MC data\n')
        tree_name = signal_path + ":/SignalTable"
        df_applied_mc = hau.apply_on_large_data(tree_name, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, split)
        df_applied_mc.query("gReconstructed and bw_accept", inplace=True)
        df_applied_mc.to_parquet(results_dir + f'/applied_df_mc.parquet.gzip', compression='gzip')

        if LOAD_APPLIED_DATA:
            path = results_dir + f'/applied_df_data.parquet.gzip'
            df_applied = pd.read_parquet(path)

        else:

            if ls_path is not None:
                print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
                print ('\nStarting BDT appplication on LS\n')
                tree_name = ls_path + ":/DataTable"
                df_applied = hau.apply_on_large_data(tree_name, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, split)
                df_applied.to_parquet(results_dir + f'/applied_df_ls.parquet.gzip', compression='gzip')

            if ls_pion_path is not None:
                print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
                print ('\nStarting BDT appplication on LS with swapped Pion\n')
                tree_name = ls_pion_path + ":/DataTable"
                df_applied = hau.apply_on_large_data(tree_name, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, split)
                df_applied.to_parquet(results_dir + '/applied_df_ls_pion.parquet.gzip', compression='gzip')

            if em_path is not None:
                print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
                print ('\nStarting BDT appplication on Event Mixing\n')
                tree_name = em_path + ":/DataTable"
                df_applied = hau.apply_on_large_data(tree_name, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, split)
                df_applied.to_parquet(results_dir + '/applied_df_em.parquet.gzip', compression='gzip')

            if data_path is not None:
                print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
                print ('\nStarting BDT appplication on Data\n')
                tree_name = data_path + ":/DataTable"
                df_applied = hau.apply_on_large_data(tree_name, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, split)
                df_applied.to_parquet(results_dir + '/applied_df_data.parquet.gzip', compression='gzip')


        if data_path is not None:
            ml_application = ModelApplication(df_applied, analysis_res_path, CENT_CLASSES, split)
            for cclass in CENT_CLASSES:
                th2_efficiency = ml_application.load_preselection_efficiency(cclass, split, TRAINING_DIR)
                df_sign = pd.DataFrame()
                for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                    ptbin_index = ml_application.presel_histo.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))
                    for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                        ctbin_index = ml_application.presel_histo.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))
                        print('\n==================================================')
                        print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)
                        print('Application and signal extraction ...', end='\r')

                        mass_bins = 40 if ctbin[1] < 16 else 36
                        presel_eff = ml_application.get_preselection_efficiency(ptbin_index, ctbin_index)
                        eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split, training_dir=TRAINING_DIR)
                        data_slice = ml_application.get_data_slice(cclass, ptbin, ctbin)

                        if SIGNIFICANCE_SCAN:
                            sigscan_eff, sigscan_tsd = ml_application.significance_scan(data_slice, presel_eff, eff_score_array, cclass, ptbin, ctbin, split, mass_bins)
                            eff_score_array = np.append(eff_score_array, [[sigscan_eff], [sigscan_tsd]], axis=1)
                            sigscan_results[f'ct{ctbin[0]}{ctbin[1]}pt{ptbin[0]}{ptbin[1]}{split}'] = [sigscan_eff, sigscan_tsd]
                        print('Application and signal extraction: Done!\n')
    

    try:
        sigscan_results = np.asarray(sigscan_results)
        filename_sigscan = results_dir + '/Efficiencies/' + TRAINING_DIR + '/sigscan.npy'
        np.save(filename_sigscan, sigscan_results)
    except:
        print('No sigscan, no sigscan results!')

    print (f'--- ML application time: {((time.time() - app_time) / 60):.2f} minutes ---')    
    print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
