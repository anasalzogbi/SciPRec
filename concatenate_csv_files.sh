#!/bin/bash

dataset=$1
declare -a machines=("dbisma02" "dbisma03" "dbisma04" "dbisma05" "dbisma06" "dbisma07" "dbisma08" "dbisma09" "dbisma10")
out_file_name="final_results.csv"
file_name="metrics-results.csv"
for i in "${machines[@]}"
do
	result_file="$i.csv"
	scp -r $i:/home/alzoghba/sciprec_cluster_experiment/results/$dataset collected_results/$dataset/$i
done

FILES=results/$dataset/*
for f in $FILES
do
  echo "Processing $f file..."
  fname="$PATH" | rev | cut -d"/" -f1 | rev
  cat $f >> collected_results/$dataset/$fname
  # take action on each file. $f store current file name
  cat $f
done

cat results/$dataset/$file_name >> collected_results/$out_file_name
for i in "${machines[@]}"
do
	result_file="$i.csv"
	cat collected_results/$result_file >> collected_results/$out_file_name
done