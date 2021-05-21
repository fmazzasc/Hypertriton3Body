import hipe4ml.plot_utils as pu
import hipe4ml.analysis_utils as au
from hipe4ml.tree_handler import TreeHandler
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

results_dir = "../../Results/"
pdf = matplotlib.backends.backend_pdf.PdfPages(results_dir + "distributions.pdf")

def rename_df_columns(hndl):
    df = hndl.get_data_frame()
    rename_dict = {}

    for col in df.columns:

        if col.endswith('_f'):
            rename_dict[col] = col[:-2]
        if col.startswith('tpc_clus'):
            rename_dict[col] = 'tpc_ncls' + col[-5:-2]
        if col== 'tpc_sig_pi_f':
            rename_dict[col] = 'tpc_nsig_pi'
    
    df.rename(columns = rename_dict, inplace=True)

column = ["m", "pt","cos_pa", "tpc_ncls_de", "tpc_ncls_pr","tpc_ncls_pi","tpc_nsig_de","tpc_nsig_pr","tpc_nsig_pi", "dca_de", "dca_pr", "dca_pi","dca_de_pr","dca_de_pi","dca_pr_pi","dca_de_sv","dca_pr_sv","dca_pi_sv","chi2","cos_theta_ppi_H"]

path_to_data = "/data/fmazzasc/PbPb_3body/pass3/tables/"
dataH = TreeHandler(path_to_data + "HypDataTable_data.root", "DataTable", entry_stop=1e6)
lsH = TreeHandler(path_to_data + "HypDataTable_ls_rot.root", "DataTable", entry_stop=1e6)
lsH_non_rot = lsH.apply_preselections("rotation_f==0", inplace=False)
lsH_old = TreeHandler(path_to_data + "HypDataTable_ls.root", "DataTable", entry_stop=1e6)
emH = TreeHandler(path_to_data + "HypDataTable_em.root", "DataTable", entry_stop=1e6)
signalH = TreeHandler(path_to_data + "SignalTable_20g7.root", "SignalTable", entry_stop=1e6)
signalH.apply_preselections("pt>0 and bw_accept==True and cos_pa>0")


rename_df_columns(dataH)
rename_df_columns(lsH)
rename_df_columns(lsH_non_rot)
rename_df_columns(lsH_old)
rename_df_columns(emH)
rename_df_columns(signalH)


pu.plot_distr([signalH, dataH], column=column,labels=["Sig", "Data"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
pdf.savefig()

pu.plot_distr([dataH, lsH],column=column,labels=["Data", "LS + Rotation"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
pdf.savefig()

pu.plot_distr([dataH, lsH_non_rot],column=column,labels=["Data", "LS"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
pdf.savefig()

pu.plot_distr([dataH, lsH_old],column=column,labels=["Data", "LS OLD"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
pdf.savefig()

pu.plot_distr([lsH_old, lsH],column=column,labels=["LS OLD", "LS + Rotation"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
pdf.savefig()

pu.plot_distr([lsH_non_rot, lsH],column=column,labels=["LS", "LS + Rotation"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
pdf.savefig()


# pu.plot_distr([dataH, emH],column=column,labels=["Data", "EM"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
# pdf.savefig()

# pu.plot_distr([lsH, emH],column=column,labels=["LS + Rotation", "EM"], log=True, density=True, figsize=(20, 20), alpha=0.3, grid=False)
# pdf.savefig()

pdf.close()