import os
import yaml

tree_name = "HyperTritonTree3.root"
output_name = "/data/fmazzasc/PbPb_3body/pass3/trees/HyperTritonTree3_LS_rotation.root"

with open('data_path_list') as f:
    lines = f.read().splitlines()
print(lines)

for ind,line in enumerate(lines):
    print(ind)
    input_files = line + "/" + tree_name
    os.system(f' alien_cp -T 32 {input_files} /data/fmazzasc/PbPb_3body/pass3/trees/part_merging/{tree_name}{ind}')

# os.system(f"alihadd {output_name} part_merging/{tree_name}*")
# os.system(f"rm -rf part_merging/*")
