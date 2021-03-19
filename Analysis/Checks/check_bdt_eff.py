import numpy as np
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler

efficiency_dir = "../../Results/Efficiencies/" + "ls_training_not_opt"
model_dir = "../../Models/handlers/" + "ls_training_not_opt"


def get_effscore_dict(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = efficiency_dir + f'/Eff_Score_{info_string}.npy'
    return {round(e[0], 2): e[1] for e in np.load(file_name).T}

def get_model_handler(ctbin):
    info_string = f'090_210_{ctbin[0]}{ctbin[1]}'
    file_name = model_dir + f'/model_handler_{info_string}.pkl'
    hndl = ModelHandler()
    hndl.load_model_handler(file_name)
    return hndl

eff = 0.7
score = get_effscore_dict([2,4])[eff]
print(score)

rec_hndl = TreeHandler("/data/fmazzasc/PbPb_3body/pass3/tables/SignalTable_20g7.root", "SignalTable")
rec_hndl.apply_preselections("bw_accept==True and 2<ct<4 and gReconstructed==True and pt>2")

hndl = get_model_handler([2,4])
rec_hndl.apply_model_handler(hndl)
print("BDT EFF: ", len(rec_hndl.get_subset("model_output>5.634624538361291"))/len(rec_hndl))


