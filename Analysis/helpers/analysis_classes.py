
import os
import sys
import ROOT
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from sklearn.model_selection import train_test_split
import hyp_analysis_utils as hau
import hyp_plot_utils as hpu


class TrainingAnalysis:

    def __init__(self, mc_file_name, bkg_file_name, split, entrystop=10000000):

        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\nStarting BDT training and testing ')
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')


        self.df_signal = uproot.open(mc_file_name)['SignalTable'].arrays(library="pd").query('bw_accept and cos_pa > 0 and pt > 2')
        self.df_generated = uproot.open(mc_file_name)['SignalTable'].arrays(library="pd").query('bw_accept')
        self.df_bkg = uproot.open(bkg_file_name)['DataTable'].arrays(library="pd", entry_stop=entrystop)
        hau.rename_df_columns(self.df_bkg)

            
        if split == '_antimatter':
            self.df_bkg = self.df_bkg.query('ArmenterosAlpha < 0')
            self.df_signal = self.df_signal.query('ArmenterosAlpha < 0')
            self.df_generated = self.df_generated.query('matter < 0.5')

        if split == '_matter':
            self.df_bkg = self.df_bkg.query('ArmenterosAlpha > 0')
            self.df_signal = self.df_signal.query('ArmenterosAlpha > 0')
            self.df_generated = self.df_generated.query('matter > 0.5')

        self.df_signal['y'] = 1
        self.df_bkg['y'] = 0

    def preselection_efficiency(self, cent_class, ct_bins, pt_bins, split='', training_dir = ""):
        cent_cut = f'{cent_class[0]}<=centrality<={cent_class[1]}'

        if (len(ct_bins)>2):
            cut  =  f'{pt_bins[0]}<=pt<={pt_bins[1]}'
        else:
            cut  =  f'{ct_bins[0]}<=ct<={ct_bins[1]}'            
            
        # rap_cut = ' and abs(rapidity)<0.5'
        pres_histo = hau.h2_preselection_efficiency(pt_bins, ct_bins)
        gen_histo = hau.h2_generated(pt_bins, ct_bins)

        for pt, ct in self.df_signal.query(cent_cut + " and " + cut)[['pt', 'ct']].to_records(index=False):
            pres_histo.Fill(pt,ct)
        
        cols = ['gPt', 'gCt']
        for pt, ct in self.df_generated.query(cent_cut)[cols].to_records(index=False):
            gen_histo.Fill(pt,ct)

        pres_histo.Divide(gen_histo)

        path = "../Results/Efficiencies/" + training_dir
        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + f'/preseleff_cent{cent_class[0]}{cent_class[1]}{split}.root'
        t_file = ROOT.TFile(filename, 'recreate')      
        pres_histo.Write()
        t_file.Close()

        return pres_histo

    def prepare_dataframe(self, training_columns, cent_class, pt_range, ct_range, test_size=0.5):
        data_range = f'{ct_range[0]}<ct<{ct_range[1]} and {pt_range[0]}<pt<{pt_range[1]} and {cent_class[0]}<=centrality<{cent_class[1]}'

        sig = self.df_signal.query(data_range)
        bkg = self.df_bkg.query(data_range)

        if (len(bkg) >= 10*len(sig)):
            bkg = bkg.sample(n=10*len(sig))

        print('\nNumber of signal candidates: {}'.format(len(sig)))
        print('Number of background candidates: {}\n'.format(len(bkg)))

        df = pd.concat([self.df_signal.query(data_range), self.df_bkg.query(data_range)])

        train_set, test_set, y_train, y_test = train_test_split(df[training_columns + ['m']], df['y'], test_size=test_size, random_state=42)

        return [train_set, y_train, test_set, y_test]

    def MC_sigma_array(self, data, eff_score_array, cent_class, pt_range, ct_range, split=''):
        info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}{split}'

        sigma_path = "../Utils" + '/FixedSigma'

        if not os.path.exists(sigma_path):
            os.makedirs(sigma_path)

        filename_sigma = sigma_path + '/sigma_array' + info_string + '.npy'
        filename_sigma_error = sigma_path + '/sigma_array_error' + info_string + '.npy'
        filename_mean = sigma_path + '/mean_array' + info_string + '.npy'
        filename_mean_error = sigma_path + '/mean_array_error' + info_string + '.npy'

        mean_dict = {}
        mean_error_dict ={}
        sigma_dict = {}
        sigma_error_dict ={}

        data[2]['score'] = data[2]['score'].astype(float)

        mass_bins = 40 if ct_range[1] < 16 else 36

        for eff, cut in zip(eff_score_array[0], eff_score_array[1]):
            counts = np.histogram(data[2][data[3].astype(bool)].query(f'score>{cut}')['m'], bins=mass_bins)

            histo_minv = hau.h1_invmass(counts, cent_class, pt_range, ct_range)

            histo_minv.Fit('gaus', 'Q')
            print("ciao")
            mean = histo_minv.GetFunction('gaus').GetParameter(1)
            mean_error = histo_minv.GetFunction('gaus').GetParError(1)
            sigma = histo_minv.GetFunction('gaus').GetParameter(2)
            sigma_error = histo_minv.GetFunction('gaus').GetParError(2)
            sigma = hau.round_to_error(sigma, sigma_error)

            del histo_minv

            mean_dict[f'{eff:.2f}'] = mean
            mean_error_dict[f'{eff:.2f}'] = mean_error
            sigma_dict[f'{eff:.2f}'] = sigma
            sigma_error_dict[f'{eff:.2f}'] = sigma_error
        
        np.save(filename_mean, np.array(mean_dict))
        np.save(filename_mean_error, np.array(mean_error_dict))
        np.save(filename_sigma, np.array(sigma_dict))
        np.save(filename_sigma_error, np.array(sigma_error_dict))

    def save_ML_analysis(self, model_handler, eff_score_array, cent_class, pt_range, ct_range, split="", training_dir=""):
        info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}{split}'

        handlers_path = "../Models/handlers/" + training_dir
        efficiencies_path = "../Results/Efficiencies/" + training_dir


        if not os.path.exists(handlers_path):
            os.makedirs(handlers_path)
        if not os.path.exists(efficiencies_path):
            os.makedirs(efficiencies_path)

        filename_handler = handlers_path + '/model_handler' + info_string + '.pkl'
        filename_efficiencies = efficiencies_path + '/Eff_Score' + info_string + '.npy'

        model_handler.dump_model_handler(filename_handler)
        np.save(filename_efficiencies, eff_score_array)

        print('ML analysis results saved.\n')

    def save_ML_plots(self, model_handler, data, eff_score_array, cent_class, pt_range, ct_range, split='', training_dir=""):
        fig_path = "../Figures/"
        info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}{split}'

        bdt_score_dir = fig_path + training_dir + '/TrainTest'
        bdt_eff_dir = fig_path + training_dir +'/Efficiency'
        feat_imp_dir = fig_path + training_dir + '/FeatureImp'

        bdt_score_plot = plot_utils.plot_output_train_test(model_handler, data, bins=100, log=True)
        if not os.path.exists(bdt_score_dir):
            os.makedirs(bdt_score_dir)

        bdt_score_plot.savefig(bdt_score_dir + '/BDT_Score' + info_string + '.pdf')

        bdt_eff_plot = plot_utils.plot_bdt_eff(eff_score_array[1], eff_score_array[0])
        if not os.path.exists(bdt_eff_dir):
            os.makedirs(bdt_eff_dir)

        bdt_eff_plot.savefig(bdt_eff_dir + '/BDT_Eff' + info_string + '.pdf')

        feat_imp = plot_utils.plot_feature_imp(data[2][model_handler.get_original_model().get_booster().feature_names], data[3], model_handler)
        if not os.path.exists(feat_imp_dir):
            os.makedirs(feat_imp_dir)

        plt.savefig(feat_imp_dir + '/FeatImp' + info_string + '.pdf')
        plt.close()

        print('ML plots saved.\n')


class ModelApplication:

    def __init__(self, data_filename, analysis_res_filename, cent_classes, split):

        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\nStarting BDT appplication and signal extraction')

        self.n_events = []

        self.df_data = data_filename if isinstance(data_filename, pd.DataFrame) else uproot.open(data_filename)['DataTable'].arrays(library="pd")
        self.hist_centrality = uproot.open(analysis_res_filename)["AliAnalysisTaskHypertriton3_summary"][11].values()

        for cent in cent_classes:
            self.n_events.append(sum(self.hist_centrality[cent[0]:cent[1]-1]))

        print('\nNumber of events: ', int(sum(self.hist_centrality[:])))

        if split == '_antimatter':
            self.df_data = self.df_data.query('ArmenterosAlpha < 0')
            print(f'\nNumber of anti-hyper-candidates: {len(self.df_data)}')

        if split == '_matter':
            self.df_data = self.df_data.query('ArmenterosAlpha > 0')
            print(f'Number of hyper-candidates: {len(self.df_data)}')

        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')

    def load_preselection_efficiency(self, cent_class, split, training_dir=""):
        efficiencies_path = "../Results/Efficiencies/" + training_dir
        filename_efficiencies = efficiencies_path + f'/preseleff_cent{cent_class[0]}{cent_class[1]}{split}.root'

        tfile = ROOT.TFile(filename_efficiencies)

        self.presel_histo = tfile.Get("PreselEff")
        self.presel_histo.SetDirectory(0)

        return self.presel_histo

    def load_ML_analysis(self, cent_class, pt_range, ct_range, split='', training_dir=""):

        info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}{split}'

        handlers_path = "../Models/handlers/" + training_dir
        efficiencies_path = "../Results/Efficiencies/" + training_dir

        filename_handler = handlers_path + '/model_handler' + info_string + '.pkl'
        filename_efficiencies = efficiencies_path + '/Eff_Score' + info_string + '.npy'

        eff_score_array = np.load(filename_efficiencies)

        model_handler = ModelHandler()
        model_handler.load_model_handler(filename_handler)

        return eff_score_array, model_handler

    def get_preselection_efficiency(self, ptbin_index, ctbin_index):
        return self.presel_histo.GetBinContent(ptbin_index, ctbin_index)

    def load_sigma_array(self, cent_class, pt_range, ct_range, split=''):
        info_string = '_{}{}_{}{}_{}{}{}'.format(cent_class[0], cent_class[1], pt_range[0],
                                                 pt_range[1], ct_range[0], ct_range[1], split)
        sigma_path = '../Utils/FixedSigma'
        filename_sigma = sigma_path + "/sigma_array" + info_string + '.npy'
        return np.load(filename_sigma)


    def get_data_slice(self, cent_class, pt_range, ct_range):
        data_range = f'{ct_range[0]}<ct<{ct_range[1]} and {pt_range[0]}<pt<{pt_range[1]} and {cent_class[0]}<=centrality<{cent_class[1]}'

        return self.df_data.query(data_range)

    def significance_scan(self, df_bkg, pre_selection_efficiency, eff_score_array, cent_class, pt_range, ct_range, split='', mass_bins=40):
        print('\nSignificance scan: ...')

        hyp_lifetime = 253
        hist_range = [2.96, 3.04]

        bdt_efficiency = eff_score_array[0]
        threshold_space = eff_score_array[1]

        expected_signal = []
        significance = []
        significance_error = []
        significance_custom = []
        significance_custom_error = []

        bw_file = bw_file = ROOT.TFile('../Utils' + '/BlastWaveFits.root', 'read')
        bw = [bw_file.Get('BlastWave/BlastWave{}'.format(i)) for i in [0, 1, 2]]
        bw_file.Close()

        for index, tsd in enumerate(threshold_space):
            df_selected = df_bkg.query('score>@tsd')

            counts, bins = np.histogram(df_selected['m'], bins=mass_bins, range=hist_range)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            side_map = (bin_centers < 2.98) + (bin_centers > 3.005)
            mass_map = np.logical_not(side_map)
            bins_side = bin_centers[side_map]
            counts_side = counts[side_map]

            h, residuals, _, _, _ = np.polyfit(bins_side, counts_side, 2, full=True)
            y = np.polyval(h, bins_side)

            exp_signal_ctint = hau.expected_signal_counts(
                bw, cent_class, pt_range, pre_selection_efficiency * bdt_efficiency[index],
                self.hist_centrality, 3)

            if split is not '':
                exp_signal_ctint = 0.5 * exp_signal_ctint

            ctrange_correction = hau.expo(ct_range[0], hyp_lifetime)-hau.expo(ct_range[1], hyp_lifetime)

            exp_signal = exp_signal_ctint * ctrange_correction
            exp_background = sum(np.polyval(h, bin_centers[mass_map]))

            expected_signal.append(exp_signal)

            if (exp_background < 0):
                exp_background = 0

            sig = exp_signal / np.sqrt(exp_signal + exp_background + 1e-10)
            sig_error = hau.significance_error(exp_signal, exp_background)

            significance.append(sig)
            significance_error.append(sig_error)

            sig_custom = sig * bdt_efficiency[index]
            sig_custom_error = sig_error * bdt_efficiency[index]

            significance_custom.append(sig_custom)
            significance_custom_error.append(sig_custom_error)

        nevents = sum(self.hist_centrality[cent_class[0]+1:cent_class[1]])

        max_index = np.argmax(significance_custom)
        max_score = threshold_space[max_index]
        max_significance = significance_custom[max_index]
        data_range_array = [ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1]]
        hpu.plot_significance_scan(
            max_index, significance_custom, significance_custom_error, expected_signal, df_bkg, threshold_space,
            data_range_array, nevents, 3, split, mass_bins, hist_range)

        bdt_eff_max_score = bdt_efficiency[max_index]

        print('Significance scan: Done!')

        # return max_score, bdt_eff_max_score, max_significance
        return bdt_eff_max_score, max_score


def load_mcsigma(cent_class, pt_range, ct_range, mode, split=''):
    info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}{split}'
    sigma_path = '../Utils/FixedSigma'

    file_name = f'{sigma_path}/sigma_array{info_string}.npy'

    return np.load(file_name)
