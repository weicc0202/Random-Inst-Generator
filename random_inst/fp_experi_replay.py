#============================================================
# FileName    : fp_experi_replay.py
# Usage       : $ python3 fp_experi_replay.py (default)
# Description : Generate test program by selecting random instructions.
# Dependency  : python3.6 (Required packages add in env_setting.tcl) 
# Author      : Lynn <lynn840429@gmail.com>
# Version     : 2020/03/12
#============================================================
import os
import random

SEED_ITER_PATH = '../catpg/tmax_file/rpt/'


def rand_seed():
	path, dirs, files = next(os.walk(SEED_ITER_PATH))
	file_count = len(files)
	Seed_list = [42, 35, 81, 60, 61]

	return Seed_list[file_count]


def main():
	seed = rand_seed()


if __name__ == '__main__':
	main()
