# Predicting earthquakes

This is my implementation of the [LANL Earthquake-Prediction challenge](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion) 
The goal of this project is to predict upcoming earthquakes based on a stream of seismic activity. The training data consists of a ~10GB
large csv file which contains a seismic_activity and a time_to_failure column, which represents how many seconds are 
left before the next earthquake:

seismic_activity (V) | time_to_failure (s)
------------ | -------------
12 |	1.4690999832
6	| 1.4690999821
8	| 1.469099981
5	| 1.4690999799
8	| 1.4690999788
8	| 1.4690999777
9	| 1.4690999766
7	| 1.4690999755
-5 | 	1.4690999744
... | ...

The goal of the challenge is to find the "best" correlation between a range of seismic activities and the time remaining before 
the next earthquake, so that we can predict the time_to_failure column given only the seismic_activity column.

In the training data, we can observe an earthquake every time the time_to_failure values jumps from zero up to a higher value, giving us 16 earthquakes in total:

![Seismic activity](https://github.com/phillikus/earthquake_predictions/blob/master/plots/summary.png)

To download the training/test data, please check out the link mentioned above.
Read more about this challenge [on my blog](https://cheesyprogrammer.com/2019/01/25/competing-on-kaggle-com-for-the-first-time/)