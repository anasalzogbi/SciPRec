#!/bin/bash

users_count=$1
machines_count=$2
dataset=$3
cores=$4
peers=$5
min_sim=$6
max_sim=$7
users_per_machine=$((users_count/machines_count))

function run_instance() {
        echo "$1: python2 /home/alzoghba/sciprec_cluster_experiment/runnables_cluster.py -s $2 -e $3 -d $4 -cores $5 -p $6 -mn $7 -mx $8  -w $1"
        ssh -tt "$1" "python2 /home/alzoghba/sciprec_cluster_experiment/runnables_cluster.py -s $2 -e $3 -d $4 -cores $5 -p $6 -mn $7 -mx $8 -w $1"

}

declare -a machines=("dbisma02" "dbisma03" "dbisma04" "dbisma05" "dbisma06" "dbisma07" "dbisma08" "dbisma09" "dbisma10")


if [ "${#machines[@]}" -ne "$((machines_count-1))" ]; then
        echo "Machine count is not equal to ssh handles"
        exit
fi

counter=0

for i in "${machines[@]}"
do
#    echo "================Parameters=============="
#    echo "dataset: $dataset"
#    echo "cores: $cores"
#    echo "peers: $peers"
#    echo "min_sim: $min_sim"
#    echo "max_sim: $max_sim"
#    echo "================Parameters=============="
    let "start=$users_per_machine*$counter"
    let "end=($counter+1)*$users_per_machine"
#    echo "start"
#    echo $i
#    echo $start
#    if ((counter == machines_count-1)); then
#            let "end=$users_count"
#    fi
#    echo $end
#    echo $users_per_machine
    run_instance $i $start $end $dataset $cores $peers $min_sim $max_sim &
    ((counter++))
done
#echo "python2 runnables_cluster.py -s $end -e $users_count -d $dataset -cores $((cores-1)) -p $peers -mn $min_sim -mx $max_sim"
python2 runnables_cluster.py -s $end -e $users_count -d $dataset -cores $((cores-2)) -p $peers -mn $min_sim -mx $max_sim -w dbisma01 &
echo "end script"