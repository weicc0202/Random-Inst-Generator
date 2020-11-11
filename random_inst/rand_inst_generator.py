#============================================================
# FileName    : rand_inst_generator.py
# Usage       : $ python3 rand_inst_generator.py (default)
# Description : Generate test program by selecting random instructions.
# Dependency  : python3.6 (Required packages add in env_setting.tcl) 
# Author      : Lynn <lynn840429@gmail.com>
# Version     : 2020/03/12
#============================================================
import os, sys, argparse
import numpy as np
import pandas as pd
import random, math, pickle
import collections
import json
import fp_experi_replay as exp_seed

from bitarray import bitarray


### User Parameter & Processor Info
TP_LENGTH = 1000
seed_num = exp_seed.rand_seed()
print(seed_num)
random.seed ( seed_num )


### Path Setting & Parameters Setting
ISA_PATH = "./ISA_File/"
PROGRAM_PATH = "../test_programs/"
MEM_FILE = PROGRAM_PATH + "mem.hex"
OP_TYPE = ["R-Type", "R2-Type", "I-Type", "J-Type", "CleanUp"]
INST_FILEDS = [[6, 5, 5, 5, 5, 6], [6, 5, 5, 16], [6, 26]]
NOP_INST = "00000000000000000000000000000000"
LUI_INST = "001111"
ORI_INST = "001101"
REG_MAX_VALUE = 2**31 - 1

if not os.listdir("../test_programs/test_pipereg"):
    print("Directory is empty")
    Reg_classify = True
else:    
    print("Directory is not empty")
    Reg_classify = False


### Class - templete
class Templete():
	def __init__(self, op_code=[], \
					   rs_code=[], rt_code=[], rd_code=[], \
					   shamt_code=[], func_code=[], imm_code=[], addr_code=[], \
					   bit_len=[]):
		self.op_code 	= op_code
		self.rs_code 	= rs_code
		self.rt_code 	= rt_code
		self.rd_code 	= rd_code
		self.shamt_code = shamt_code
		self.func_code 	= func_code
		self.imm_code 	= imm_code
		self.addr_code 	= addr_code
		self.bit_len 	= bit_len
		self.inst_len 	= sum(bit_len[0]) 
		self.inst 		= ["0"*self.inst_len]

	def templateRType(self, pos=0):
		op  	= random.choices(self.op_code[pos])[0]
		rs 		= random.choices(self.rs_code[pos])[0]
		rt 		= random.choices(self.rt_code[pos])[0]
		rd 		= random.choices(self.rd_code[pos])[0]
		shumt 	= getRandomBitString(5)
		funct   = random.choices(self.func_code[pos])[0]
		self.inst.clear()
		
		if (funct=="100000" or funct=="100001"):	# ADD & ADDU
			self.inst.extend(self.tmeplateADD(op, rs, rt, rd, funct))
			return self.inst
		elif (funct=="011000" or funct=="011001"):	# MULT & MULTU
			self.inst.extend(self.templateMULT(op, rs, rt, rd, funct))
			return self.inst
		else:
			rtype_inst = op + rs + rt + rd + shumt + funct
			self.inst.append(rtype_inst)
			return self.inst

		return self.inst

	def templateR2Type(self, pos=1):
		op  	= random.choices(self.op_code[pos])[0]
		rs 		= random.choices(self.rs_code[pos])[0]
		rt 		= random.choices(self.rt_code[pos])[0]
		rd 		= random.choices(self.rd_code[pos])[0]
		shumt 	= getRandomBitString(5)
		funct   = random.choices(self.func_code[pos])[0]
		
		r2type_inst = op + rs + rt + rd + shumt + funct
		self.inst.clear()
		self.inst.append(r2type_inst)
		return self.inst
	
	def templateIType(self, pos=2): 	# rt <- rs op imm
		op  	= random.choices(self.op_code[pos])[0]
		rs 		= random.choices(self.rs_code[pos])[0]
		rt 		= random.choices(self.rt_code[pos])[0]
		while (rs==rt): rt  = random.choices(self.rt_code[pos])[0]

		if (Reg_classify):
			self.inst.clear()

			self.inst = ["00111100000010000000000000000000",
						 "00110101000010000000000000000000",
						 "00111100000010010000000000000000",
						 "00110101001010010000000000000000",
						 "00000000000000000000000000000000",
						 "00000000000000000000000000000000"]

			for i in range(0, 25*TP_LENGTH*2, 25):
				self.inst.append("0010000100101000" + str(decimalTobin(i, 16)))
				self.inst.append("0010000100001001" + str(decimalTobin(i, 16)))

			return self.inst

		else:
			#imm 	= getRandomBitString(16)
			imm 	= getRandomBitString(3).zfill(16)
			self.inst.clear()
		
			reg_val = random.randint(0, (REG_MAX_VALUE - binTodecimal(imm))//1000) 	# value for rs
			reg_val = decimalTobin(reg_val, 32)
		
			if (op=="000100" or op=="000101"): 	# OP_BEQ or OP_BNE
				self.inst.extend(self.templateBranch(op, rs, rt, imm, reg_val))
				return self.inst
			else:
				self.inst.append(LUI_INST + "00000" + rs + reg_val[:16])	# inst_lui_rs
				self.inst.append(ORI_INST + rs + rs + reg_val[16:])			# inst_ori_rs
				self.inst.append(NOP_INST)									# NOP
				self.inst.append(op + rs + rt + imm)						# I-Type inst
				return self.inst

		return self.inst
	
	def templateJType(self, pos=3, inst_mem=[]):
		op  	= random.choices(self.op_code[pos])[0]
		addr 	= getRandomBitString(26)
		addr_pos = len(inst_mem)+1
		self.inst.clear()

		if (op=="000010"): 	# J
			self.inst.extend(self.templateJump(op, addr, addr_pos))
			#self.inst = ["0"*self.inst_len]
			return self.inst

		self.inst = ["0"*self.inst_len]
		return self.inst

	def templateCleanUp(self):
		inst = ['3C080000',  # LUI $t8 0x0000
				'35080000',  # ORI $t8 $t8 0x0000
				'3C090000',  # $t9
				'35290000',
				'3C0A0000',  # $t10
				'354A0000',
				'3C0B0000',  # $t11
				'356B0000',
				'3C0C0000',  # $t12
				'358C0000',
				'3C0D0000',  # $t13
				'35AD0000',
				'3C0E0000',  # $t14
				'35CE0000',
				'3C0F0000',  # $t15
				'35EF0000',
				'00000000','00000000','00000000','00000000','00000000']

		return inst

	def tmeplateADD(self, op, rs, rt, rd, funct):	# rd <- rs + rt, REG_MAX_VALUE = 2147483647
		inst = []
		reg_seg = random.randint(0, REG_MAX_VALUE)
		reg_0 = str(decimalTobin(random.randint(0, reg_seg), 32))
		reg_1 = str(decimalTobin(random.randint(0, REG_MAX_VALUE-reg_seg), 32))

		inst.append(LUI_INST + "00000" + rs + reg_0[:16])				# inst_lui_rs
		inst.append(ORI_INST + rs + rs + reg_0[16:])					# inst_ori_rs
		inst.append(LUI_INST + "00000" + rt + reg_1[:16])				# inst_lui_rt
		inst.append(ORI_INST + rt + rt + reg_1[16:])					# inst_ori_rt
		inst.append(NOP_INST)											# NOP
		inst.append(op + rs + rt + rd + getRandomBitString(5) + funct)	# inst_add, inst_addu

		return inst

	def templateMULT(self, op, rs, rt, rd, funct):
		inst = []

		reg_seg = random.randint(0, REG_MAX_VALUE//10000)
		reg_val = [random.randint(0, reg_seg), random.randint(0, REG_MAX_VALUE//reg_seg)]
		random.shuffle(reg_val)
		reg_0 = str(decimalTobin(reg_val[0], 32))
		reg_1 = str(decimalTobin(reg_val[1], 32))

		inst.append(LUI_INST + "00000" + rs + reg_0[:16])	# inst_lui_rs
		inst.append(ORI_INST + rs + rs + reg_0[16:])		# inst_ori_rs
		inst.append(LUI_INST + "00000" + rt + reg_1[:16])	# inst_lui_rt
		inst.append(ORI_INST + rt + rt + reg_1[16:])		# inst_ori_rt
		inst.append(NOP_INST)								# NOP
		inst.append(op + rs + rt + '0000000000' + funct)	# inst_mult, inst_multu

		return inst

	def templateBranch(self, op, rs, rt, imm, comp):
		inst = []

		if (op=="000100"): 	# OP_BEQ
			inst.append(LUI_INST + "00000" + rs + comp[:16])	# inst_lui_rs
			inst.append(ORI_INST + rs + rs + comp[16:])			# inst_ori_rs
			inst.append(LUI_INST + "00000" + rt + comp[16:])	# inst_lui_rt
			inst.append(ORI_INST + rt + rt + comp[:16])			# inst_ori_rt
			inst.extend(self.templateNOP(str(1)))				# wait for lui ori
			inst.append(op + rs + rt + imm)						# inst_beq, rs!=rt -> not jump
			inst.extend(self.templateNOP(imm))					# inst_nop
			inst.append(LUI_INST + "00000" + rs + comp[16:])	# inst_lui_rs
			inst.append(ORI_INST + rs + rs + comp[:16])			# inst_ori_rs
			inst.extend(self.templateNOP(str(1)))				# wait for lui ori
			inst.append(op + rs + rt + imm)						# inst_beq, rs=rt -> jump
		else: 				# OP_BNE, (op=="000101")
			inst.append(LUI_INST + "00000" + rs + comp[:16])	# inst_lui_rs
			inst.append(ORI_INST + rs + rs + comp[16:])			# inst_ori_rs
			inst.append(LUI_INST + "00000" + rt + comp[:16])	# inst_lui_rt
			inst.append(ORI_INST + rt + rt + comp[16:])			# inst_ori_rt
			inst.extend(self.templateNOP(str(1)))				# wait for lui ori
			inst.append(op + rs + rt + imm)						# inst_beq, rs=rt -> not jump
			inst.extend(self.templateNOP(imm))					# inst_nop
			inst.append(LUI_INST + "00000" + rs + comp[16:])	# inst_lui_rs
			inst.append(ORI_INST + rs + rs + comp[:16])			# inst_ori_rs
			inst.extend(self.templateNOP(str(1)))				# wait for lui ori
			inst.append(op + rs + rt + imm)						# inst_beq, rs!=rt -> jump
		
		return inst

	def templateJump(self, op, addr, inst_curpos=0, addr_len=26):
		inst = []
		addr_max_range = 5
		bin_addr = binTodecimal(addr)

		while (bin_addr<2 or bin_addr>addr_max_range):
			bin_addr = random.randint(3, addr_max_range)

		else:
			addr_nxt 	= decimalTobin(inst_curpos+bin_addr+1, addr_len)		# pos
			addr_nop	= decimalTobin(bin_addr-2, addr_len)					# len
			addr_return = decimalTobin(inst_curpos+1, addr_len) 				# pos
			#print(inst_curpos, inst_curpos+bin_addr, inst_curpos+bin_addr+1, bin_addr-2, inst_curpos+1)

			inst.append(op + decimalTobin(inst_curpos+bin_addr, addr_len))		# inst_j
			inst.append(op + addr_nxt)											# inst_j+1
			inst.extend(self.templateNOP(addr_nop))								# inst_nop
			inst.append(op + addr_return)										# inst_j_back

		#print(inst)
		#inst = ["0"*self.inst_len]
		return inst

	def templateNOP(self, num=1):	
		return ["0"*32] * int(num, 2)


### Read File
def json_Read(path):
	with open(path+"antares_ISA.json", 'r') as fn_json:
		fn_data = json.loads(fn_json.read())

	return fn_data


def json_Parsing(js_dict):
	isa_type = []
	bit_len_list = []
	op, rs, rt, rd, shamt, funct, imm, addr = [], [], [], [], [], [], [], []
	isa_type.extend(list(js_dict.keys()))

	#print("\n[ISA Formate]")
	for t in range(len(isa_type)):
		isa_list = []
		isa_list.extend(list(js_dict[isa_type[t]].keys())) 

		for f in range(len(isa_list)):
			if (isa_list[f]=='Format'):
				#print(isa_type[t], "Format :", js_dict[isa_type[t]][isa_list[f]])
				bit_pos_list = list(js_dict[isa_type[t]][isa_list[f]].values())
				for bit_len in bit_pos_list:
					bit_len_list.append( int(bit_len.split(":")[0])-int(bit_len.split(":")[1])+1 )
			else:
				if (isa_list[f]=="OP"):
					op.append(js_dict[isa_type[t]][isa_list[f]])
				elif (isa_list[f]=="RS"):
					rs.append(js_dict[isa_type[t]][isa_list[f]])
				elif (isa_list[f]=="RT"):
					rt.append(js_dict[isa_type[t]][isa_list[f]])
				elif (isa_list[f]=="RD"):
					rd.append(js_dict[isa_type[t]][isa_list[f]])
				elif (isa_list[f]=="FUNCT"):
					funct.append(js_dict[isa_type[t]][isa_list[f]])

	return op, rs, rt, rd, funct, bit_len_list


def split_Bits(code_list, bit_len=0):
	row = len(code_list)

	if (code_list[0][0].find("6'b")!=(-1)):
		for r in range(row):
			for c in range(len(code_list[r])):
				code_list[r][c] = code_list[r][c].replace("6'b", "")
				code_list[r][c] = code_list[r][c].replace("_", "")

	elif (code_list[0][0].find(":")!=(-1)):
		for r in range(row):
			for c in range(len(code_list[r])):
				sv, ev = code_list[r][c].split(":")
				code_list[r] = decimalTobin_parser(sv, ev, bit_len)

	return code_list


def preprocess_Inst(inst_list):
	inst_list = [ x for x in inst_list if "'b" in x ]
	inst_list = [ x for x in inst_list if "//" not in x ]
	inst_list = [ x.split("'b")[1] for x in inst_list ]
	inst_list = [ x.replace('_','') for x in inst_list ]
	inst_list = [ x.replace('\n','') for x in inst_list ]
	return inst_list


def read_Inst(path, isa_arc, ban_op_file, ban_func_file):
	op_inst, rs_inst, rt_inst, rd_inst, func_inst = [], [], [], [], []

	print('[Check File]', 										\
		  '\n1. Circuit Architecture File:\t'	, isa_arc, 		\
		  '\n2. Banned OP-code File:\t\t'		, ban_op_file, 	\
		  '\n3. Banned Function-code File:\t'	, ban_func_file )

	if (isa_arc!=path+'false'):
		isa_info 	= json_Read(path)
		op_inst, rs_inst, rt_inst, rd_inst, func_inst, arc_len = json_Parsing(isa_info)

		op_inst     = split_Bits(op_inst)
		rs_inst     = split_Bits(rs_inst, arc_len[1])
		rt_inst     = split_Bits(rt_inst, arc_len[2])
		rd_inst     = split_Bits(rd_inst, arc_len[3])
		func_inst   = split_Bits(func_inst)
	else:
		print("Architecture file open fail.")

	if (ban_op_file!=path+'false'):
		ban_op_inst = []
		with open(ban_op_file, 'r') as ban_op_fn:
			ban_op_inst = ban_op_fn.readlines()
		ban_op_inst = preprocess_Inst(ban_op_inst)
	
	if (ban_func_file!=path+'false'):
		ban_func_inst = []
		with open(ban_func_file, 'r') as ban_func_fn:
			ban_func_inst = ban_func_fn.readlines()
		ban_func_inst = preprocess_Inst(ban_func_inst)

	op_inst = banned_inst(op_inst, ban_op_inst); #print("op_inst=", op_inst)
	func_inst = banned_inst(func_inst, ban_func_inst); #print("func_inst=", func_inst)
	
	return op_inst, rs_inst, rt_inst, rd_inst, func_inst


def banned_inst(code_list, b_code_list):
	tmp_row, tmp_col = [], []
	for row in code_list:
		tmp_col = []
		for col in row:
			if (col not in b_code_list):
				tmp_col.append(col)
		tmp_row.append(tmp_col)
	code_list = tmp_row.copy()
	del tmp_col, tmp_row
	return code_list


def decimalTobin_parser(start, end, bit_num):
	form = '0' + str(int(bit_num)) + 'b'
	valid_code = list(range(int(start), int(end)+1, 1))
	valid_code = [ format(x, form) for x in valid_code ]
	return valid_code


def decimalTobin(val, bit_len):
	return bin(val)[2:].zfill(bit_len)


def binTodecimal(val):
	return int(val, 2)


def binToHex(val, bit_len):
	return hex(int(val, 2))[2:].zfill(bit_len)


def binToHex_list(bin_list, bit_len):
	hex_list = []
	if (len(bin_list[0])!=8):
		for b in bin_list:
			hex_list.append(hex(int(b, 2))[2:].upper().zfill(bit_len))
		return hex_list
	else:
		return bin_list


def getRandomBitString(k):
	return bin(random.getrandbits(k))[2:].zfill(k)


def reverseBit(i):
	if i == '1': 	return '0'
	elif i == '0':	return '1'


### Generate Test Program
def sel_Inst(op_list):
	'''
	Select one Instruction.
		OP_TYPE = ["R-Type", "R2-Type", "I-Type", "J-Type", "CleanUp"]
	'''
	# op_type = random.choices(population=OP_TYPE, weights=[0.74, 0.2, 0.02, 0.01, 0.03], k=1)
	# op_type = random.choices(population=OP_TYPE, weights=[0.8, 0.12, 0.02, 0.02, 0.04], k=1)
	# op_type = random.choices(population=OP_TYPE, weights=[0.45, 0.1, 0.2, 0.05, 0.2], k=1)
	# op_type = random.choices(population=OP_TYPE, weights=[0.7, 0.1, 0.1, 0.02, 0.08], k=1)
	# op_type = random.choices(population=OP_TYPE, weights=[0.55, 0.1, 0.2, 0.05, 0.1], k=1)

	if (Reg_classify):
		op_type = random.choices(population=OP_TYPE, weights=[0, 0, 1, 0, 0], k=1)
	else:
		op_type = random.choices(population=OP_TYPE, weights=[0.55, 0.1, 0.2, 0.05, 0.1], k=1)
		
	
	if (op_type[0]=='R-Type'):
		#print('R-Type')
		return op_type[0], random.choices(op_list[0])
	elif (op_type[0]=='R2-Type'):
		#print('R2-Type')
		return op_type[0], random.choices(op_list[1])
	elif (op_type[0]=='I-Type'):
		#print('I-Type')
		return op_type[0], random.choices(op_list[2])
	elif (op_type[0]=='J-Type'):
		#print('J-Type')
		return op_type[0], random.choices(op_list[3])
	elif (op_type[0]=='CleanUp'):
		#print('CleanUp')
		return op_type[0], []
	else:
		print('Error Type Happened.')


def generate_TP(tc, op_code_list=[]):
	'''
	Generate TP_LENGTH instructions, OP_TYPE = ["R-Type", "R2-Type", "I-Type", "J-Type", "CleanUp"]
	'''
	inst_mem = []
	inst_mem.extend(['00000000', '00000000', '00000000', '00000000', '00000000']) 	# Instruction waiting

	if (Reg_classify):
		TP_LENGTH_in = 1
	else:
		TP_LENGTH_in = TP_LENGTH

	for count in range(TP_LENGTH_in):
		optype, opcode = sel_Inst(op_code_list)
		inst = []
		#print(optype, ":", opcode)

		if (optype==OP_TYPE[0]):	# R-Type
			inst = tc.templateRType(OP_TYPE.index(optype))
		elif (optype==OP_TYPE[1]):	# R2-Type
			inst = tc.templateR2Type(OP_TYPE.index(optype))
		elif (optype==OP_TYPE[2]):	# I-Type
			inst = tc.templateIType(OP_TYPE.index(optype))
		elif (optype==OP_TYPE[3]):	# J-Type
			inst = tc.templateJType(OP_TYPE.index(optype), inst_mem)
		elif (optype==OP_TYPE[4]): 	# CleanUp
			inst = tc.templateCleanUp()
		else:
			print('Error Type Happened.')

		inst_mem.extend(binToHex_list(inst, 8))

	inst_mem.extend(['00000000', '00000000', '00000000', '00000000', '00000000']) 	# Instruction waiting
	inst_mem.extend(['00000000', '00000000', '00000000', '00000000', '00000000']) 	# Instruction waiting
	return inst_mem


def write_mem(path, mem_list):
	print("Test Program len =", len(mem_list))
	with open(path, 'w') as fn:
		fn.write('\n'.join(mem_list))


### Main
def main():
	if not os.path.isdir(PROGRAM_PATH):
		os.mkdir(PROGRAM_PATH)

	## Argument Parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--isa_arc', type=str, default=ISA_PATH+'antares_ISA.json')
	parser.add_argument('--banned_op', type=str, default=ISA_PATH+'ban_opcode_list.txt')
	parser.add_argument('--banned_func', type=str, default=ISA_PATH+'ban_functioncode_list.txt')
	args = parser.parse_args()

	op_inst, rs_inst, rt_inst, rd_inst, func_inst = read_Inst(ISA_PATH, args.isa_arc, args.banned_op, args.banned_func)

	T = Templete(op_inst, rs_inst, rt_inst, rd_inst, [], func_inst, [], [], \
				 bit_len=INST_FILEDS)

	inst_mem = generate_TP(T, op_inst)

	write_mem(MEM_FILE, inst_mem)



if __name__ == '__main__':
	main()
