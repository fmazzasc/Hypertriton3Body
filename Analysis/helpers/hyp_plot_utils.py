import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import hist
import hyp_analysis_utils as hau

matplotlib.use('pdf')

###############################################################################
# define custom colors
kBlueC = ROOT.TColor.GetColor('#1f78b4')
kBlueCT = ROOT.TColor.GetColorTransparent(kBlueC, 0.25)
kRedC = ROOT.TColor.GetColor('#e31a1c')
kRedCT = ROOT.TColor.GetColorTransparent(kRedC, 0.25)
kPurpleC = ROOT.TColor.GetColor('#911eb4')
kPurpleCT = ROOT.TColor.GetColorTransparent(kPurpleC, 0.25)
kOrangeC = ROOT.TColor.GetColor('#ff7f00')
kOrangeCT = ROOT.TColor.GetColorTransparent(kOrangeC, 0.25)
kGreenC = ROOT.TColor.GetColor('#33a02c')
kGreenCT = ROOT.TColor.GetColorTransparent(kGreenC, 0.25)
kMagentaC = ROOT.TColor.GetColor('#f032e6')
kMagentaCT = ROOT.TColor.GetColorTransparent(kMagentaC, 0.25)
kYellowC = ROOT.TColor.GetColor('#ffe119')
kYellowCT = ROOT.TColor.GetColorTransparent(kYellowC, 0.25)


def plot_efficiency_significance(mode, tsd, significance, efficiency, data_range_array):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'

    ax1.set_xlabel('BDT Score')
    ax1.set_ylabel('Significance', color=color)
    ax1.plot(tsd, significance, color=color)
    ax1.tick_params(axis='y', labelcolor=color, direction='in')

    ax2 = ax1.twinx()

    color = 'tab:red'
    # we already handled the x-label with ax1
    ax2.set_ylabel('BDT efficiency', color=color)
    ax2.plot(tsd, efficiency, color=color)
    ax2.tick_params(axis='y', labelcolor=color, direction='in')

    fig.tight_layout()

    fig_eff_path = '../../Figures/Significance'
    if not os.path.exists(fig_eff_path):
        os.makedirs(fig_eff_path)

    fig_name = '/sign_eff_ct{}{}_pT{}{}_cen{}{}.pdf'.format(
        data_range_array[0],
        data_range_array[1],
        data_range_array[2],
        data_range_array[3],
        data_range_array[4],
        data_range_array[5])
    plt.savefig(fig_eff_path + fig_name)
    plt.close()


def plot_significance_scan(
        max_index, significance, significance_error, expected_signal, bkg_df, score_list, data_range_array,
        n_ev, mode, split='', mass_bins=40, hist_range = [2.96,3.04]):

    label = 'Significance x Efficiency'

    raw_yield = expected_signal[max_index]
    max_score = score_list[max_index]

    selected_bkg = bkg_df.query('score>@max_score')

    bkg_counts, bins = np.histogram(
        selected_bkg['m'], bins=mass_bins, range=hist_range)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    signal_counts_norm = norm.pdf(bin_centers, loc=2.992, scale=0.0025)
    signal_counts = raw_yield * signal_counts_norm / sum(signal_counts_norm)

    side_map = (bin_centers < 2.98) + (bin_centers > 3.005)
    bins_side = bin_centers[side_map]
    mass_map = np.logical_not(side_map)

    bkg_side_counts = bkg_counts[side_map]

    bkg_roi_shape = np.polyfit(bins_side, bkg_side_counts, 2)
    bkg_roi_counts = np.polyval(bkg_roi_shape, bin_centers)

    tot_counts = (bkg_roi_counts + signal_counts)[mass_map]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].set_xlabel('Score')
    axs[0].set_ylabel(label)
    axs[0].tick_params(axis='x', direction='in')
    axs[0].tick_params(axis='y', direction='in')
    axs[0].plot(score_list, significance, 'b',
                label='Expected {}'.format(label))

    significance = np.asarray(significance)
    significance_error = np.asarray(significance_error)

    low_limit = significance - significance_error
    up_limit = significance + significance_error

    axs[0].fill_between(score_list, low_limit, up_limit,
                        facecolor='deepskyblue', label=r'$ \pm 1\sigma$')
    axs[0].grid()
    axs[0].legend(loc='upper left')

    bkg_side_error = np.sqrt(bkg_side_counts)
    tot_counts_error = np.sqrt(np.absolute(tot_counts))

    bins_mass = bin_centers[mass_map]

    axs[1].errorbar(bins_side, bkg_side_counts, yerr=bkg_side_error,
                    fmt='.', ecolor='k', color='b', elinewidth=1., label='Data')
    axs[1].errorbar(bins_mass, tot_counts, yerr=tot_counts_error,
                    fmt='.', ecolor='k', color='r', elinewidth=1., label='Pseudodata')
    axs[1].plot(bin_centers, bkg_roi_counts, 'g-', label='Background fit')

    x = np.linspace(2.9923 - 3 * 0.0025, 2.9923 + 3 * 0.0025, 1000)
    gauss_signal_counts = norm.pdf(x, loc=2.992, scale=0.0025)
    gauss_signal_counts = (raw_yield / sum(signal_counts_norm)) * \
        gauss_signal_counts + np.polyval(bkg_roi_shape, x)

    axs[1].plot(x, gauss_signal_counts, 'y', color='orange',
                label='Signal model (Gauss)')
    axs[1].set_xlabel(r'$m_{\ ^{3}He+\pi^{-}}$')
    axs[1].set_ylabel(r'Events /  ${:.3}\ \rm{{MeV}}/c^{{2}}$'.format((hist_range[1] - hist_range[0])/mass_bins))
    axs[1].tick_params(axis='x', direction='in')
    axs[1].tick_params(axis='y', direction='in')
    axs[1].legend(loc='best', frameon=False)
    plt.ylim(bottom=0)

    s = sum(tot_counts) - sum(bkg_roi_counts[mass_map])
    b = sum(bkg_roi_counts[mass_map])

    sign_score = s / np.sqrt(s + b)

    plt.suptitle(r'%1.f$\leq ct \leq$%1.f %1.f$\leq \rm{p}_{T} \leq$%1.f  Cut Score=%0.2f  Significance=%0.2f  Raw yield=%0.2f' % (
        data_range_array[0], data_range_array[1], data_range_array[2], data_range_array[3], max_score,  sign_score, raw_yield))

    # text = '\n'.join(
    #     r'%1.f GeV/c $ \leq \rm{p}_{T} < $ %1.f GeV/c ' % (data_range_array[0], data_range_array[1]),
    #     r' Significance/Sqrt(Events) = %0.4f$x10^{-4}$' % (max_significance / np.sqrt(n_ev) * 1e4))

    # props = dict(boxstyle='round', facecolor='white', alpha=0)

    # axs[1].text(0.37, 0.95, text, transform=axs[1].transAxes, verticalalignment='top', bbox=props)

    fig_name = 'Significance_ct{}{}_pT{}{}_cen{}{}{}.pdf'.format(
        data_range_array[0],
        data_range_array[1],
        data_range_array[2],
        data_range_array[3],
        data_range_array[4],
        data_range_array[5],
        split)

    fig_sig_path = '../../Figures/Significance'
    if not os.path.exists(fig_sig_path):
        os.makedirs(fig_sig_path)

    plt.savefig(fig_sig_path + '/' + fig_name)
    plt.close()


def plot_confusion_matrix(y_true, df, mode, score,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, fig_name='confusion.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # if the score is closer to max then to min it's recognised as signal
    y_pred = [1 if i > score else 0 for i in df['score']]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['Background', 'Signal']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    fig_sig_path = '../../Figures/Confusion'
    if not os.path.exists(fig_sig_path):
        os.makedirs(fig_sig_path)

    plt.savefig(fig_sig_path + '/' + fig_name)
    plt.close()

    return ax


def mass_plot_makeup(histo, model, ptbin, split):
    mass, mass_error = hau.histo_weighted_mean(histo)

    mass_low = mass - mass_error
    mass_up = mass + mass_error

    blambda = 1115.683 + 1875.61294257 - mass

    print(f'B_lambda = {blambda:.3f} +- {mass_error:.3f}')

    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(kBlueC)
    histo.SetLineColor(kBlueC)

    canvas = ROOT.TCanvas(f'hyp_mass_{model}{split}')
            
    pad_range = [2990.85, 2992.15]
    label = 'm_{ {}^{3}_{#bar{#Lambda}} #bar{H}}' if split is '_antimatter' else 'm_{ {}^{3}_{#Lambda}H}'
    frame = ROOT.gPad.DrawFrame(ptbin[0], pad_range[0], ptbin[-1], pad_range[1], ';#it{p}_{T} (GeV/#it{c});' + label + ' [ MeV/#it{c}^{2} ]')
    frame.GetYaxis().SetTitleSize(22)
    frame.GetYaxis().SetTitleOffset(1.4)
     
    pinfo = ROOT.TPaveText(0.142, 0.652, 0.521, 0.849, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(11)
    pinfo.SetTextFont(43)
    pinfo.SetTextSize(24)

    string_list = []
    string_list.append('#bf{ALICE Internal}')
    string_list.append('Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%')
    string_list.append('B_{#Lambda}'+' = {:.3f} #pm {:.3f} '.format(round(blambda, 3), round(mass_error, 3)) + 'MeV')
        
    for s in string_list:
        pinfo.AddText(s)

    ROOT.gPad.Update()
    mass_line = ROOT.TLine(ROOT.gPad.GetUxmin(), mass, ROOT.gPad.GetUxmax(), mass)
    mass_line.SetLineColor(kOrangeC)
    mass_line.SetLineWidth(1)

    mass_box = ROOT.TBox(ROOT.gPad.GetUxmin(), mass_low, ROOT.gPad.GetUxmax(), mass_up)
    mass_box.SetFillColor(kOrangeCT)
    mass_box.SetFillStyle(1001)
    mass_box.SetLineWidth(1)
    mass_box.SetLineStyle(2)
    mass_box.Draw('same')

    mass_line.Draw('same')
    pinfo.Draw('x0same')
    histo.Draw('ex0same')
    canvas.Write()
    # canvas.SaveAs(f'mass_{model}.pdf')


def sigma_plot_makeup(histo, model, ptbin, split):
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(kBlueC)
    histo.SetLineColor(kBlueC)

    canvas = ROOT.TCanvas(f'hyp_width_{model}_{split}')
            
    pad_range = [1.15, 2.9]
    label = '#sigma_{ {}^{3}_{#bar{#Lambda}} #bar{H}}' if split is '_antimatter' else '#sigma_{ {}^{3}_{#Lambda}H}'
    frame = ROOT.gPad.DrawFrame(ptbin[0], pad_range[0], ptbin[-1], pad_range[1], ';#it{p}_{T} (GeV/#it{c});' + label + ' [ MeV/#it{c}^{2} ]')
    frame.GetYaxis().SetTitleSize(22)
    frame.GetYaxis().SetTitleOffset(1.1)
     
    pinfo = ROOT.TPaveText(0.141, 0.716, 0.521, 0.848, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(11)
    pinfo.SetTextFont(43)
    pinfo.SetTextSize(24)

    pinfo.AddText('#bf{ALICE Internal}')
    pinfo.AddText('Pb-Pb  #sqrt{#it{s}_{NN}} = 5.02 TeV,  0-90%')

    pinfo.Draw('x0same')
    histo.Draw('ex0same')
    canvas.Write()


def dalitz_plot(x_data, y_data, eff, ct_bin, x_axis, y_axis, x_label='', y_label='', training_dir=""):
    
    plot = (
        hist.Hist.new
        .Reg(x_axis[0],x_axis[1],x_axis[2], name='x', label=x_label)
        .Reg(y_axis[0],y_axis[1],y_axis[2], name='y', label=y_label)
        .Double()
        )

    plot.fill(x=x_data,y=y_data)

    ax = plot.plot2d_full(
        main_cmap="cividis",
        top_color="steelblue",
        top_lw=2,
        side_lw=2,
        side_color="steelblue"
        )
    path = f'../Figures/{training_dir}/Dalitz/ct_{ct_bin[0]}_{ct_bin[1]}'
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(path + f'/dal_plot_eff_{eff}.png', dpi=300, facecolor='white')
    plt.close()