#!/usr/bin/env python3

import os
from ROOT import gROOT

gROOT.SetBatch(True)
gROOT.LoadMacro("PrepareDataFrames.cc")
gROOT.LoadMacro("GenerateTableO2.cc")
from ROOT import PrepareDataFrames, GenerateTableO2

input_dir = "/data/fmazzasc/PbPb_3body/pass3/trees"
output_dir = "/data/fmazzasc/PbPb_3body/pass3/tables"

# print("++++++++++++++++++++++++++++++++++++++++++")
# print("Generate Signal Table")
# print("++++++++++++++++++++++++++++++++++++++++++")
# GenerateTableO2(input_dir + "/HyperTritonTree3_20g7.root", output_dir + "/SignalTable_20g7.root")
# print("++++++++++++++++++++++++++++++++++++++++++")
# print("Generate Data Table")
# print("++++++++++++++++++++++++++++++++++++++++++")
# PrepareDataFrames("data", input_dir, output_dir)
# print("++++++++++++++++++++++++++++++++++++++++++")
# print("Generate Like-Sign Background Table")
# print("++++++++++++++++++++++++++++++++++++++++++")
# PrepareDataFrames("ls", input_dir, output_dir)
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Event Mixing Background Table")
print("++++++++++++++++++++++++++++++++++++++++++")
PrepareDataFrames("em", input_dir, output_dir)
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate LS pion Background Table")
print("++++++++++++++++++++++++++++++++++++++++++")
PrepareDataFrames("ls_pion", input_dir, output_dir)
