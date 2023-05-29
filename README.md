
<h1> Metric Space Analysis </h1>

This repository hosts algorithms, functions and estimators for clustering adapted to the unique characteristics
 of neuronal entropy and spike timing.

Estimators are wrapped in sklearn.BaseEstimator and sklearn.ClusterMixin classes for integration into sklearn pipelines.

![Python](https://img.shields.io/badge/python-3670A0?style=?style=plastic&logo=python&logoColor=ffdd54)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=plastic&logo=c%2B%2B&logoColor=white)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/NeuroPyPy/CalciumAnalysis/blob/master/LICENSE)

![GitHub language count](https://img.shields.io/github/languages/count/NeuroPyPy/Metric-Space-Analysis?style=plastic)
![Repo Size](https://img.shields.io/github/repo-size/NeuroPyPy/CalciumAnalysis?style=plastic)

<p>
  <img style="float: right"
    width="400"
    height="400"
    src=https://i.imgur.com/7w6mbN9.png
  >
</p>

###### Classification algorith based on:
<a href=https://journals.physiology.org/doi/abs/10.1152/jn.1996.76.2.1310> Nature and precision of temporal coding in visual cortex: a metric-space analysis </a>

###### Most neural activity is represented relative to the spike rate of the individual neurons.

Neurons that *fire significantly faster around the onset of a stimulus* 
is considered a stimulus-active neuron. The activity of these neurons is then used to classify the stimulus. This is a very common method of classification in neuroscience.

### However, the brain is complicated, and this may not tell the entire story.

The metric space analysis is a quantification using spike timings rather than spike rates. Using this method, we can quantify the similarity between two spike trains. 

This allows us to classify stimuli based on the similarity of the spike trains and **how much information** is conveyed. 

+ q-value : precision of temporal coding. q=0 resolves no information in temporal coding, where all information lies in the spike count.
+ h-value : amount of information conveyed by the spike train. h=0 conveys no information, where the spike train is completely random.

The crux of this model is attempting to use **pairwise distances between spike-trains** from different neurons, different animals, or different sessions to classify stimuli.

<p>
  <img style="float: right"
    width="800"
    height="400"
    src=https://i.imgur.com/NSLwRUs.png
  >
</p>

