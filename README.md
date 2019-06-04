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

Read more about this challenge [on my blog](https://cheesyprogrammer.com/2019/01/25/competing-on-kaggle-com-for-the-first-time/).

#Just a note by "the cloner" of this repo, after the closing of the Kaggle challenge

In this competition the "public" test set, the one used to produce the public leaderboard (LB), was just a small fraction of the whole test set that has been used to produce the private LB when the competition closed. This caused a massive shake up in the leaderboard, with some of the "once top 10" folks catapulted down by 2 THOUSANDS (!) places in the leaderboard (and some other lucky guy with 2 submissions catapulted back up by 4.000 places :joy:).

My last CatBoost-based blended submission scored far worse than what one of my first LSTM-128 submissions would have scored (2.4779 = about 130th place on private LB). All this teaches us (at least) two important things.

1. (and this is bad news) the "real" data scientists, those with sound statistical skills, not only CS skills stays on the top
2. "apprentice data scientists" don't stay on top but can still score high on very hard problems. This is the power of machine learning :wink:

![Shake Up Top 10](https://github.com/phillikus/earthquake_predictions/blob/master/plots/private-LB-shake-up-competition.png)
![Shake Up older Top 10](https://github.com/phillikus/earthquake_predictions/blob/master/plots/shake-up-1.png)
![Shake Up older Top 10](https://github.com/phillikus/earthquake_predictions/blob/master/plots/shake-up-2.png)

Other important lessons learned:

1. never trust the dataset (see the explanatory picture below), in this case anybody had clear indications that the test set was totally a different thing WRT the training set, and people able to clean their dataset from those biases scored higher.
2. I need to learn how to do feature engineering and features selection if not automatically (e.g. through CNNs) at least not "by hand" as I did in this competition.
3. LSTM networks FTW!

![toy dataset vs. real-world datasets](https://github.com/phillikus/earthquake_predictions/blob/master/plots/toy-dataset-vs-real-world-datasets.jpg)
