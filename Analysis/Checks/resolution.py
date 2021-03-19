import numpy as np
import matplotlib.pyplot as plt
from hipe4ml.tree_handler import TreeHandler
import matplotlib.backends.backend_pdf

results_dir = "../../Results/"
pdf = matplotlib.backends.backend_pdf.PdfPages(results_dir + "resolutions.pdf")

hndl = TreeHandler("/data/fmazzasc/PbPb_3body/pass3/tables/SignalTable_20g7.root", "SignalTable")
hndl.apply_preselections("gReconstructed==1")
hndl.eval_data_frame("pt_res = gPt - pt", inplace=True)
hndl.eval_data_frame("ct_res = gCt - ct", inplace=True)

plt.hist(hndl["pt_res"], bins=1000, range=[-1,1])
plt.xlabel(r"p$_T$ resolution")
pdf.savefig()

plt.figure()
plt.hist(hndl["ct_res"], bins=1000, range=[-5,5])
plt.xlabel(r"$c$t resolution")
pdf.savefig()


pdf.close()

