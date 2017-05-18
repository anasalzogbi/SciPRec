#!/bin/bash
declare -a machines=("ssh_handle_1" "ssh_handle_2" "ssh_handle3" "ssh_handle4" "ssh_handle5" "ssh_handle_6" "ssh_handle_7" "ssh_handle8" "ssh_handle9" "ssh_handle10")
out_file_name="final_results.csv"
file_name="metrics-results.csv"
for i in "${machines[@]}"
do
	ssh $i  "tail /PATH/$file_name >> /PATH/$out_file_name"
done

