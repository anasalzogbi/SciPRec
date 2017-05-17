#!/bin/bash

users_count=$1
machines_count=$2
users_per_machine=$((users_count/machines_count))
run_instance() {
	ssh "$1"
	sciprec
	python2 runnables.py -s $2 -e $3 &
}
declare -a machines=("dbisma01" "dbisma02" "dbisma03" "dbisma04" "dbisma05" "dbisma06" "dbisma07" "dbisma08" "dbisma09" "dbisma10")
if [ "${#machines[@]}" -ne "${machines_count}" ]; then
	echo "Machine count is not equal to ssh handles"
	exit
fi
counter=0
for i in "${machines[@]}"
do


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
	run_instance $i $start $end &
	((counter++))
done
echo $users

