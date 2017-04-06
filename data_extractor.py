#!/usr/bin/env python

import sys
import os
import csv
user_lim = 50
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/citeulike_a_extended', 'users.dat')
write_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/dummy', 'users.dat')

documents = []
with open(path, "r") as f:
	for i, line in enumerate(f):
		first_word = False
		if i == user_lim:
			break
		for entry in line.split(" "):
			if first_word:
				first_word = False
				continue
			documents.append(int(entry))

old_to_new = {}
new_to_old = {}
sorted_documents = sorted(list(set(documents)))
for i, document in enumerate(sorted_documents):
	old_to_new[document] = i
	new_to_old[i] = document

with open(path, "r") as f:
	with open(write_path, "w") as write_f:
		for user_id, line in enumerate(f):
			if user_id == user_lim:
				break
			first_word = True
			write_line = ""
			for entry in line.split(" "):
				if first_word:
					write_line += entry
					first_word = False
					continue
				write_line += " " + str(old_to_new[int(entry)])
			write_f.write(write_line + "\n")
			# write_f.write(line)

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/citeulike_a_extended', 'raw-data.csv')
write_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/dummy', 'raw-data.csv')
documents_set = set(documents)
citeulike_ids = set()
with open(path, "r") as f:
	with open(write_path, "w") as write_f:
		first_line = True
		write_line = ""
		for i, line in enumerate(f):
			if first_line:
				write_f.write(line)
				first_line = False
				continue
			# print(i - 1)
			# print(documents_set)
			if i in documents_set:
				splitted = line.split(",")
				splitted[0] = old_to_new[int(splitted[0])]
				citeulike_ids.add(int(splitted[2]))
				write_f.write(','.join(map(str, splitted)))

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/citeulike_a_extended', 'paper_info.csv')
write_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/dummy', 'paper_info.csv')
with open(path, "r") as f:
	with open(write_path, "w") as write_f:
		first_line = True
		write_line = ""
		for i, line in enumerate(f):
			if first_line:
				write_f.write(line)
				first_line = False
				continue
			# print(i - 1)
			# print(documents_set)
			splitted = line.split("\t")
			if int(splitted[0]) in documents_set:
				splitted[0] = old_to_new[int(splitted[0])]
				write_f.write('\t'.join(map(str, splitted)))				


path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/citeulike_a_extended', 'mult.dat')
write_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/dummy', 'mult.dat')
with open(path, "r") as f:
	with open(write_path, "w") as write_f:
		first_line = True
		write_line = ""
		for i, line in enumerate(f):
			if first_line:
				write_f.write(line)
				first_line = False
				continue
			# print(i - 1)
			# print(documents_set)
			if i in documents_set:
				write_f.write(line)	


path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/citeulike_a_extended', 'authors.csv')
write_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/Extended_ctr/dummy', 'authors.csv')

with open(path, "r") as f:
	with open(write_path, "w") as write_f:
		delimiter = '\t'
		first_line = True
		for line in f:
			if first_line:
				first_line = False
				write_f.write(line)
				continue
			splitted = line.split(delimiter)
			if int(splitted[0]) in citeulike_ids:
				write_f.write(line)

#svr = SVR()