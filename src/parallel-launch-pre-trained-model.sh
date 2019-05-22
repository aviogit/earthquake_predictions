#!/bin/bash

parallel_jobs=4
sequential_runs=5

if [ ! -z "$1" ] ; then
	if [ $1 -gt 1 ] ; then
		parallel_jobs=$1
	fi
fi
if [ ! -z "$2" ] ; then
	if [ $2 -gt 1 ] ; then
		sequential_runs=$2
	fi
fi

echo "Running with $parallel_jobs parallel jobs and $sequential_runs sequential runs, for a total of $((parallel_jobs * sequential_runs)) processes."

parallel_jobs=$((parallel_jobs - 1))

training_set="/tmp/LANL-Earthquake-Prediction-train-csv-gzipped/features-2019-05-15_15.52.59-feature_count-225-batch_size-32-epochs-2000.csv"
test_set="/tmp/LANL-Earthquake-Prediction-train-csv-gzipped/test_set_features-2019-05-16_17.10.36-test_set_feature_count-225-batch_size-8-epochs-10000.csv"

# This one is unable to produce TTF predictions above 9.1 :(
#model="/home/biagio/LANL-Earthquake-Prediction-train-csv-gzip-1/lanl-checkpoints-hopefully-good/lanl-checkpoint-949-0.44984.hdf5"
model="/home/biagio/LANL-Earthquake-Prediction-train-csv-gzip-1/lanl-checkpoints-hopefully-good/lanl-checkpoint-639-0.48996.hdf5"

for j in `seq 1 $sequential_runs`
do
	echo "Starting sequential run no. $j"
	for i in `seq 1 $parallel_jobs`
	do
	
		echo "Launching instance no. $i"
		./main.py "$training_set" "$test_set" "$model" &> /tmp/lanl-pretrained-model-run-$j-$i.log &
	
	done
	echo "Launching blocking (foreground) instance no. $((parallel_jobs + 1))"
	./main.py "$training_set" "$test_set" "$model" &> /tmp/lanl-pretrained-model-run-$j-$((parallel_jobs + 1)).log
done


