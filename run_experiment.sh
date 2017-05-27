#!/bin/bash

users_count=$1
machines_count=$2
dataset=$3
cores=$4
min_sim_start=$5
min_sim_end=$6
min_sim_step=$7
max_sim_start=$8
max_sim_end=$9
max_sim_step=${10}
peers_start=${11}
peers_end=${12}
peers_step=${13}

echo "================Parameters=============="
echo "users_count: $users_count"
echo "machines_count: $machines_count"
echo "dataset: $dataset"
echo "cores: $cores"
echo "min_sim_start:$min_sim_start"
echo "min_sim_end: $min_sim_end"
echo "min_sim_step: $min_sim_step"
echo "max_sim_start:$max_sim_start"
echo "max_sim_end: $max_sim_end"
echo "max_sim_step: $max_sim_step"
echo "peers_start: $peers_start"
echo "peers_end: $peers_end"
echo "peers_step: $peers_step"
echo "================Parameters=============="

users_per_machine=$((users_count/machines_count))

function run_instance() {
        echo "$1: python2 /home/alzoghba/sciprec_cluster_experiment/runnable_cluster_1.py -s $2 -e $3 -d $4 -cores $5 -pe_st $6 -pe_end $7 -pe_stp $8 -mn_st $9 -mn_end ${10} -mn_stp ${11} -mx_st ${12} -mx_end ${13} -mx_stp ${14}  -w $1"
        ssh -tt "$1" "python2 /home/alzoghba/sciprec_cluster_experiment/runnable_cluster_1.py -s $2 -e $3 -d $4 -cores $5 -pe_st $6 -pe_end $7 -pe_stp $8 -mn_st $9 -mn_end ${10} -mn_stp ${11} -mx_st ${12} -mx_end ${13} -mx_stp ${14}  -w $1"

}

declare -a machines=("dbisma02" "dbisma03" "dbisma04" "dbisma05" "dbisma06" "dbisma07" "dbisma08" "dbisma09" "dbisma10")


if [ "${#machines[@]}" -ne "$((machines_count-1))" ]; then
        echo "Machine count is not equal to ssh handles"
        exit
fi

counter=0

for i in "${machines[@]}"
do
    let "start=$users_per_machine*$counter"
    let "end=($counter+1)*$users_per_machine"
    run_instance $i $start $end $dataset $cores $peers_start $peers_end $peers_step $min_sim_start $min_sim_end $min_sim_step $max_sim_start $max_sim_end $max_sim_step &
    ((counter++))
done
python2 runnable_cluster_1.py -s $end -e  $users_count -d $dataset -cores $((cores-2)) -pe_st $peers_start -pe_end $peers_end -pe_stp $peers_step -mn_st $min_sim_start -mn_end $min_sim_end -mn_stp  $min_sim_step -mx_st $max_sim_start -mx_end  $max_sim_end -mx_stp $max_sim_step  -w dbisma01  &


echo "end script"
