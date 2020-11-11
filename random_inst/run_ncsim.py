#============================================================
# FileName    : run_ncsim.py
# Usage       : $ python3 run_ncsim.py (default)
# Description : Run Ncverilog Simulation.
# Dependency  : python3.6 (Required packages add in env_setting.tcl) 
# Author      : Lynn <lynn840429@gmail.com>
# Version     : 2020/04/22
#============================================================
import os, sys
import pandas as pd


## PATH SETTING
CUR_PATH = os.getcwd() 						# pwd
os.chdir(CUR_PATH + "/../../")
TOP_PATH = os.getcwd()						# StimuliGen/
os.chdir(TOP_PATH + "/Simulation/")
GEN_PATH = os.getcwd()						# Simulation/
os.chdir(GEN_PATH + "/tests/test_program/")
MEM_PATH = os.getcwd()						# test_program/
os.chdir(GEN_PATH + "/run/")
SIM_PATH = os.getcwd()						# run/
os.chdir(TOP_PATH + "/PreLearn/")
PRE_PATH = os.getcwd()						# PreLearn/
os.chdir(TOP_PATH + "/Simulation_RL/")
SRL_PATH = os.getcwd()						# Simulation_RL/

DATA_PATH = SIM_PATH + "/out/log/"
RAND_INST_PATH = PRE_PATH + "/random_inst/"


def run_verilogsim():
	os.chdir(SIM_PATH)
	os.system("make run_fpt")


def check_data_shape():
	data = pd.read_csv(DATA_PATH+"test-pipereg.log", dtype=str)
	df_row, df_col = data.shape
	print("Data Shape: (%d, %d)" %(df_row, df_col))
	return df_row

def check_data():
	df_r = check_data_shape()

	while (df_r<1000):
		os.chdir(RAND_INST_PATH)
		os.system("python3 rand_inst_generator.py")
		os.system("mv ../test_programs/mem.hex ../../Simulation/tests/test_program/mem.hex")
		run_verilogsim()
		df_r = check_data_shape()


def main():
	run_verilogsim()
	check_data()


if __name__ == '__main__':
	main()
