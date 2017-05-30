import os
import csv
import numpy as np
dataset = 'collected_results/citeulike-a'
machine_name = 'dbisma'
result_folder = 'total'
machines = [machine_name+str(i).zfill(2) for i in range(2,11)]
output_folder = os.path.join(dataset, result_folder)
if not os.path.exists(output_folder):
		os.makedirs(output_folder)

results = []
for result_file in [x[2] for x in os.walk(os.path.join(dataset,'dbisma01'))][0]:
	with open(os.path.join(output_folder,result_file), "w+b") as total_result_file:
		writer = csv.writer(total_result_file)
		total_metrics = []
		total_ids = []
		total_lines =[]
		for machine in sorted(machines):
			path = os.path.join(os.path.join(dataset,machine),result_file)
			with open(path, "rb") as f:
				lines = f.readlines()[:-1]
				for l in lines:
					vec = l.rstrip().split(",")
					total_ids.append(int(float(vec[0])))
					total_metrics.append([float(i) for i in vec[1:]])
		#total_metrics.append(np.mean(np.array(total_metrics,dtype=float),axis=0).tolist())
		with open(os.path.join(os.path.join(dataset,'dbisma01'),result_file), "rb") as f:
			lines = f.readlines()[:-1]
			for l in lines:
				vec = l.rstrip().split(",")
				total_ids.append(int(float(vec[0])))
				total_metrics.append([float(i) for i in vec[1:]])
		for ind in range(len(total_ids)):
			total_lines.append([total_ids[ind]] + total_metrics[ind])
		average_results = ["{0:0.3f}".format(i) for i in np.mean(np.array(total_metrics,dtype=float),axis=0).tolist()]
		total_lines.append([-1]+average_results)
		results.append(result_file.rstrip('.csv')+": "+str(average_results))
		#total_file.write("\n".join(total_lines))
		writer.writerows(total_lines)
results_all = open(os.path.join(output_folder,"results_all.txt"), "a")
results_all.write("\n".join(sorted(results)))
results_all.close()
