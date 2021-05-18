import os
import ROOT
from concurrent.futures import ThreadPoolExecutor
from math import floor, log10
import aghast
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from hipe4ml.model_handler import ModelHandler


def apply_on_large_data(tree_path, cent_classes, pt_bins, ct_bins, training_columns, split='', training_dir=""):

    handlers_path = '../Models/handlers/' + training_dir
    efficiencies_path = '../Results/Efficiencies/' + training_dir

    print("Handlers at: ", handlers_path, "\n")

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
    print("\n******** Dataframe head ********\n")
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
    th2 = ROOT.TH2D(name, ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Preselection efficiency',
               len(ptbins) - 1, np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_generated(ptbins, ctbins, name='Generated'):
    th2 = ROOT.TH2D(name, ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm); Generated', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_rawcounts(ptbins, ctbins, name='RawCounts', suffix=''):
    th2 = ROOT.TH2D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Raw counts', len(ptbins)-1,
               np.array(ptbins, 'double'), len(ctbins) - 1, np.array(ctbins, 'double'))
    th2.SetDirectory(0)

    return th2


def h2_significance(ptbins, ctbins, name='Significance', suffix=''):
    th2 = ROOT.TH2D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Significance', len(ptbins)-1,
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



def fit_hist(
        histo, cent_class, pt_range, ct_range, nsigma=3, model="pol0", fixsigma=-1, sigma_limits=None, mode=2, split=''):
    # canvas for plotting the invariant mass distribution
    cv = ROOT.TCanvas(f'cv_{histo.GetName()}')

    # define the number of parameters depending on the bkg model
    if isinstance(model, list):
        n_bkgpars = len(model)
        fit_tpl = ROOT.TF1('fitTpl', f'pol2(0) + gausn(3) + gausn({n_bkgpars})', 0, 5)
        bkg_tpl = ROOT.TF1('bkgTpl', f'pol2(0) + gausn(3)', 0, 5)
        for i in range(n_bkgpars):
            fit_tpl.SetParName(i, f'B_{i}')
            fit_tpl.FixParameter(i, model[i])
        

    
    else:
        if 'pol' in str(model):
            n_bkgpars = int(model[3]) + 1
        elif 'expo' in str(model):
            n_bkgpars = 2
        else:
            print(f'Unsupported model {model}')

        fit_tpl = ROOT.TF1('fitTpl', f'{model}(0)+gausn({n_bkgpars})', 0, 5)
        bkg_tpl = ROOT.TF1('fitTpl', f'{model}(0)', 0, 5)

    # redefine parameter names for the bkg_model
        for i in range(n_bkgpars):
            fit_tpl.SetParName(i, f'B_{i}')

    # define parameter names for the signal fit
    fit_tpl.SetParName(n_bkgpars, 'N_{sig}')
    fit_tpl.SetParName(n_bkgpars + 1, '#mu')
    fit_tpl.SetParName(n_bkgpars + 2, '#sigma')
    # define parameter values and limits
    fit_tpl.SetParameter(n_bkgpars, 40)
    fit_tpl.SetParLimits(n_bkgpars, -10, 10000)  # modificato
    fit_tpl.SetParameter(n_bkgpars + 1, 2.991)
    fit_tpl.SetParLimits(n_bkgpars + 1, 2.986, 3)

    # define signal and bkg_model TF1 separately
    sigTpl = ROOT.TF1('fitTpl', 'gausn(0)', 0, 5)
    

    # plotting stuff for fit_tpl
    fit_tpl.SetNpx(300)
    fit_tpl.SetLineWidth(2)
    fit_tpl.SetLineColor(2)
    # plotting stuff for bkg model
    bkg_tpl.SetNpx(300)
    bkg_tpl.SetLineWidth(2)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.SetLineColor(2)

    # define limits for the sigma if provided
    if sigma_limits != None:
        fit_tpl.SetParameter(n_bkgpars + 2, 0.5 *
                             (sigma_limits[0] + sigma_limits[1]))
        fit_tpl.SetParLimits(n_bkgpars + 2, sigma_limits[0], sigma_limits[1])
    # if the mc sigma is provided set the sigma to that value
    elif fixsigma > 0:
        fit_tpl.FixParameter(n_bkgpars + 2, fixsigma)
    # otherwise set sigma limits reasonably
    else:
        fit_tpl.SetParameter(n_bkgpars + 2, 0.002)
        fit_tpl.SetParLimits(n_bkgpars + 2, 0.001, 0.002)

    ########################################
    # plotting the fits
    if mode == 2:
        ax_titles = ';m (^{3}He + #pi) (GeV/#it{c}^{2});Counts' + f' / {round(1000 * histo.GetBinWidth(1), 2)} MeV'
    if mode == 3:
        ax_titles = ';m (d + p + #pi) (GeV/#it{c}^{2});Counts' + f' / {round(1000 * histo.GetBinWidth(1), 2)} MeV'

    # invariant mass distribution histo and fit
    histo.UseCurrentStyle()
    histo.SetLineColor(1)
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(1)
    histo.SetTitle(ax_titles)
    histo.SetMaximum(1.5 * histo.GetMaximum())
    histo.Fit(fit_tpl, "QRL", "", 2.96, 3.04)
    histo.Fit(fit_tpl, "QRL", "", 2.96, 3.04)
    histo.SetDrawOption("e")
    histo.GetXaxis().SetRangeUser(2.96, 3.04)
    # represent the bkg_model separately
    bkg_tpl.SetParameters(fit_tpl.GetParameters())
    bkg_tpl.SetLineColor(600)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.Draw("same")
    # represent the signal model separately
    sigTpl.SetParameter(0, fit_tpl.GetParameter(n_bkgpars))
    sigTpl.SetParameter(1, fit_tpl.GetParameter(n_bkgpars+1))
    sigTpl.SetParameter(2, fit_tpl.GetParameter(n_bkgpars+2))
    sigTpl.SetLineColor(600)
    # sigTpl.Draw("same")

    # get the fit parameters
    mu = fit_tpl.GetParameter(n_bkgpars+1)
    muErr = fit_tpl.GetParError(n_bkgpars+1)
    sigma = fit_tpl.GetParameter(n_bkgpars+2)
    sigmaErr = fit_tpl.GetParError(n_bkgpars+2)
    signal = fit_tpl.GetParameter(n_bkgpars) / histo.GetBinWidth(1)
    errsignal = fit_tpl.GetParError(n_bkgpars) / histo.GetBinWidth(1)
    bkg = bkg_tpl.Integral(mu - nsigma * sigma, mu +
                           nsigma * sigma) / histo.GetBinWidth(1)

    if bkg > 0:
        errbkg = np.sqrt(bkg)
    else:
        errbkg = 0
    # compute the significance
    if signal+bkg > 0:
        signif = signal/np.sqrt(signal+bkg)
        deriv_sig = 1/np.sqrt(signal+bkg)-signif/(2*(signal+bkg))
        deriv_bkg = -signal/(2*(np.power(signal+bkg, 1.5)))
        errsignif = np.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
    else:
        signif = 0
        errsignif = 0

    # print fit info on the canvas
    pinfo2 = ROOT.TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(30+3)
    pinfo2.SetTextFont(42)

    string = f'ALICE Internal, p-Pb 2018 {cent_class[0]}-{cent_class[1]}%'
    pinfo2.AddText(string)

    decay_label = {
        "": ['{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-} + c.c.', '{}^{3}_{#Lambda}H#rightarrow dp#pi^{-} + c.c.'],
        "_matter": ['{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-}', '{}^{3}_{#Lambda}H#rightarrow dp#pi^{-}'],
        "_antimatter": ['{}^{3}_{#bar{#Lambda}}#bar{H}#rightarrow ^{3}#bar{He}#pi^{+}', '{}^{3}_{#Lambda}H#rightarrow #bar{d}#bar{p}#pi^{+}'],
    }

    string = decay_label[split][mode-2]+', %i #leq #it{ct} < %i cm %i #leq #it{p}_{T} < %i GeV/#it{c} ' % (
        ct_range[0], ct_range[1], pt_range[0], pt_range[1])
    pinfo2.AddText(string)

    string = f'Significance ({nsigma:.0f}#sigma) {signif:.1f} #pm {errsignif:.1f} '
    pinfo2.AddText(string)

    string = f'S ({nsigma:.0f}#sigma) {signal:.0f} #pm {errsignal:.0f}'
    pinfo2.AddText(string)

    string = f'B ({nsigma:.0f}#sigma) {bkg:.0f} #pm {errbkg:.0f}'
    pinfo2.AddText(string)

    if bkg > 0:
        ratio = signal/bkg
        string = f'S/B ({nsigma:.0f}#sigma) {ratio:.4f}'
        pinfo2.AddText(string)

    string = f'#mu {mu:.4f} #pm {muErr:.4f}'
    pinfo2.AddText(string)

    pinfo2.Draw()
    ROOT.gStyle.SetOptStat(0)

    st = histo.FindObject('stats')
    if isinstance(st, ROOT.TPaveStats):
        st.SetX1NDC(0.12)
        st.SetY1NDC(0.62)
        st.SetX2NDC(0.40)
        st.SetY2NDC(0.90)
        st.SetOptStat(0)

    cv.Write()
    
    return (signal, errsignal, signif, errsignif, mu, muErr, sigma, sigmaErr)