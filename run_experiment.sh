#!/bin/bash

users_count=$1
machines_count=$2
users_per_machine=$((users_count/machines_count))

declare -a machines=("ssh_handle_1" "ssh_handle_2" "ssh_handle3" "ssh_handle4" "ssh_handle5" "ssh_handle_6" "ssh_handle_7" "ssh_handle8" "ssh_handle9" "ssh_handle10")
counter=0
for i in "${machines[@]}"
do
#	ssh "$i"
#	sciprec

	let "start=$users_per_machine*$counter"
	let "end=($counter+1)*$users_per_machine"
	echo "start"
	echo $i
	echo $start
	if ((counter == machines_count-1)); then
		let "end=$users_count"
	fi
	echo $end
	echo $users_per_machine
	python2 runnables.py -s $start -e $end &
	((counter++))
done
echo $users