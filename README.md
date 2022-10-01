# UNIGRAM LANGUAGE MODEL ANALYSIS USING MLE, MAP AND PD ESTIMATES 

## This repository covers three tasks (i) Perplexity analysis, (ii)Log Evidence analysis and (iii)Author identification and it has all the plots pertinent to analysis. 

### There are three important files based on the task question number, namely first task is named `first.py`, second task is named `second.py` and third task as `third.py`.

## To get the plots for the first question

```
python first.py
```
### The resulting plots will be saved in `TRAIN.png` and `TEST.png`. The former plot has the training perplexity for varying train set size and the latter has testing perplexity for the test set. The respective values will be printed in the console. Also, there is a combined plot named `Combined Plot.png`, which combines both of the results.
![img](Combined%20Plot.png)

## For the second question

```
python second.py
```

### The plots will be saved in a single file named `Perpelxity and Log-evidences.png`. Here we will have two subpots stacked on top of each other. The first one is of log evidences and the second one is of test perplexity for training set size of 5000.
![img](Perplexity%20and%20Log-evidences.png)
## For the third question
```
python third.py
```
### It will output the test set perplexity of pg141 and pg1400 in the console and it generate a plotting image named `Author Identification.png`. Here you can see the perplexity of both test sets and compare between them.
![img](Author%20Idenitification.png)