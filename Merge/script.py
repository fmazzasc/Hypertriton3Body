import os
import yaml

tree_name = "AnalysisResults.root"
output_name = "AnalysisResults_em.root"

with open('list_em') as f:
    lines = f.read().splitlines()
print(lines)

for ind,line in enumerate(lines):
    print(ind)
    input_files = line + "/" + tree_name
    os.system(f'alien_cp {input_files} part_merging/{tree_name}{ind}')

os.system(f"alihadd {output_name} part_merging/{tree_name}*")
os.system(f"rm -rf part_merging/*")
