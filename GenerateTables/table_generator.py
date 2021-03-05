#!/usr/bin/env python3

import os
from ROOT import gROOT

gROOT.SetBatch(True)


gROOT.LoadMacro("PrepareDataFrames.cc")
from ROOT import PrepareDataFrames

input_dir = "/data/fmazzasc/PbPb_3body/pass3"
output_dir = "/data/fmazzasc/PbPb_3body/pass3"

# print("++++++++++++++++++++++++++++++++++++++++++")
# print("Generate Signal Table")
# print("++++++++++++++++++++++++++++++++++++++++++")
# GenerateTableO2(True, input_dir + "/HyperTritonTree_19d2.root", + "/SignalTable_19d2.root")
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Data Table")
print("++++++++++++++++++++++++++++++++++++++++++")
PrepareDataFrames("data", input_dir, output_dir)
print("++++++++++++++++++++++++++++++++++++++++++")
print("Generate Like-Sign Backgound Table")
print("++++++++++++++++++++++++++++++++++++++++++")
PrepareDataFrames("ls", input_dir, output_dir)
# print("++++++++++++++++++++++++++++++++++++++++++")
# print("Generate Event Mixing Background Table")
# print("++++++++++++++++++++++++++++++++++++++++++")
# PrepareDataFrames("EM", "", output_dir)
