---
title: "Efficient Labeling Through Representative Samples"
tags: ['NLP', 'Unsupervised Learning']
date: 2022-09-27T11:57:32-08:00
draft: false
---

Efficient Labeling Through Representative Samples
=================================================

Clustering for Semi-Supervised Learning on Disasters Tweets.
------------------------------------------------------------

**Figure 1.** On the left: Wildfire — Photo by [Mike Newbry](https://unsplash.com/@mikenewbry?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText). On the center: Tropical Storm — Photo by [Jeffrey Grospe](https://unsplash.com/@jgrospe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText). On the right: Pandemic Dashboard — Photo by [Martin Sanchez](https://unsplash.com/@martinsanchez?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText). Original images on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText).

Cluster analysis as an unsupervised learning technique is widely implemented throughout many fields of data science. When applied to data suited for hierarchical or partitional clustering, it can provide valuable insights into latent groups of the dataset and further improve your understanding of key features that can describe and classify individuals into meaningful clusters for your use case.

In this article, I will explore an alternative application of partitional clustering to improve the performance  of supervised learning classification tasks of text samples when the **resources to label training data are limited**. By introducing what I call “**representative labeling**” with K-Means clustering, we will see a consistent improvement in classification metrics of Logistic Regression and K-Nearest Neighbors algorithms when compared to naively labeled instances.

Problem Definition
------------------

Today’s studies on unstructured text data that revolve around Natural Language Processing (NLP) techniques are often accompanied by scraping methods that can easily fetch extensive amounts of data to compose large datasets, such as the [Twitter API v2](https://developer.twitter.com/en/docs/twitter-api) or the [MediaWiki API](https://www.mediawiki.org/wiki/API:Get_the_contents_of_a_page). However, manually labeling or annotating a subset of the collected data for training often doesn’t have an equivalently scalable and reliable sample selection method.

Therefore, the approach studied here will aim at maximizing the efficiency of annotating hours by selecting and labeling  the **most representative candidates**, leading up to a training set that is rich and representative of your corpus. Hence, we experiment with another use case of clustering as a semi-supervised learning technique (Géron, A. 2019) \[3\]. The potential audience ranges from machine learning researchers to data science students with limited resources to label training data.

Representative Labeling
-----------------------

I have recently dedicated half a year to studying the complex challenges of using NLP to capture people’s perceptions from social media posts and to understand how the Canadian society is looking at the [well-being of its elderly](https://github.com/erich-hs/Elderly-Wellbeing). On that project, my team and I had access to the powerful [Twitter Academic Research API](https://developer.twitter.com/en/products/twitter-api/academic-research), and after the data collection phase we ended up in this exact situation of having limited resources to label a training set for our classification task. That’s when I studied the effectiveness of this method (exemplified by Aurélien Géron originally on the MNIST dataset in his book _Hands-on machine learning with scikit-learn, keras and TensorFlow_ \[3\] \[4\]) to help us with selecting which samples to label, but now on text data.

Representative labeling proposes the use of K-Means clustering to segment our corpus into K distinct clusters, from which K representative instances will be selected for labeling. Those instances are the ones **closest to the centroid of each cluster**, therefore being the best representatives of their surrounding samples. The figure below illustrates this process.

**Figure 2.** Representative Sampling.

We can then expect that these individuals are important representatives of our data, and, therefore that our models will be more performant during inference when trained on representative instances. We will discuss some key aspects that deserve attention and further details about this approach later on. For now, let’s take a look at how we will prove our point.

Dataset Description
-------------------

The dataset chosen to validate our method is from a Kaggle competition named “**Natural language processing with disaster tweets**” \[6\], where the competitors are tasked with identifying which tweets from the 3,243 public test examples should be defined as a disaster (1) or non-disaster (0). It is a straightforward but challenging **binary classification problem**, where the model needs to capture the intent of the tweeted message as to whether it referred to an actual disaster event or not. The dataset and more information can be found on the [competition website](https://www.kaggle.com/c/nlp-getting-started/overview).

Libraries and Dependencies
--------------------------

This project will make use of [Preprocessor](https://github.com/s/preprocessor), a preprocessing library for tweet data written in Python that will help us with text cleaning. We will also make use of Scikit-Learn KMeans, TF-IDF vectorizer, TruncatedSVD, Normalizer, LogisticRegression, and KNeighborsClassifier.

Other visualization, statistics, machine learning, and data wrangling libraries should be available with a standard [Google Colab Notebook](https://colab.research.google.com/) instance.

Getting Started
---------------

Installing Preprocessor in your working environment or Colab instance.

```
!pip -qq install tweet-preprocessor
```

Importing libraries and dependencies.

```
import os  
import numpy as np  
import pandas as pd  
from time import time  
import preprocessor as p  
import matplotlib.pyplot as plt  
import matplotlib.cm as cm  
import seaborn as sns  
from sklearn.cluster import KMeans  
from sklearn.feature\_extraction.text import TfidfVectorizer  
from sklearn.pipeline import make\_pipeline  
from sklearn.decomposition import TruncatedSVD  
from sklearn.preprocessing import Normalizer  
from sklearn.metrics import silhouette\_score, silhouette\_samples, classification\_report, f1\_score, accuracy\_score  
from sklearn.linear\_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier**\# Seaborn and MatplotLib style settings**  
sns.set\_theme(style = "whitegrid", palette = "colorblind")  
plt.rcParams\["image.cmap"\] = "Paired"
```

Since the goal of our dataset is to validate our proposed methodology, we will use only the training set available on Kaggle. We want to evaluate our approach on a held-out validation set that we will define later on, hence the test set (unlabeled) will be of no use for now.

An easy way to directly download Kaggle datasets on Google Colab instances is through the [Kaggle API](https://github.com/Kaggle/kaggle-api). A [neatly summarized guide](https://www.kaggle.com/general/74235) by Filemon shows you exactly how to do that (Make sure to leave an upvote if you find it useful!). For now, we will simply use the downloaded train.csv file from the competition page and load it as a Pandas DataFrame.

```
**\# Setting up data directory**  
path = ''  
os.mkdir('data/')train\_data\_path = os.path.join(path, "data", "train.csv")raw\_data = pd.read\_csv(train\_data\_path)  
raw\_data
```

**Figure 2.** Disaster Tweets training set.

We can see that the original data comes with one identifier column (id) and three feature columns; keyword, location, and the uncleaned tweet text (text). Only the tweet text will be kept for our analysis.

Preprocessing
-------------

Our preprocessing will consist of three main steps, namely text cleaning, and vectorization, followed by a dimensionality reduction.

During the cleaning process, we will remove from the texts of the tweets all user mentions, URLs, and Unicode emoticon characters. We will also replace all ampersand ‘&’ symbols with an explicit ‘and’ word, and to preserve the maximum amount of information, the hashtags will be kept unchanged. Our chosen form of vectorization will be able to leverage the frequency of important hashtags in the dataset (the ones that appear frequently) and ensure that this information is available to the classification models.

The following code will implement a cleaning function and apply it to our raw dataset’s ‘text’ column, and we will also take a look at some of the original and cleaned tweets.

```
**\# Text cleaning - Removing URLs, mentions, etc using tweet-preprocessor package**  
def tweet\_clean(tweet, replace\_amper = False):  
  """  
  **Clean tweet with tweet-preprocessor p.clean().**  
  """ **# Remove user mentions, symbols and unwanted characters**  
  p.set\_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED,  p.OPT.EMOJI, p.OPT.SMILEY)  
  tweet = p.clean(tweet) **# Replace amper**  
  if replace\_amper:  
    tweet = tweet.replace('&amp;', 'and')  
    tweet = tweet.replace('&AMP;', 'and') return tweet**\# Cleaned tweets**  
raw\_data\['cleaned\_text'\] = raw\_data\['text'\].apply(tweet\_clean, replace\_amper = True)  
for tweet, cleaned\_tweet in zip(raw\_data\['text'\].tail(),  
                                raw\_data\['cleaned\_text'\].tail()):  
  print(f'Original tweet: {tweet}')  
  print(f'Cleaned tweet: {cleaned\_tweet}\\n-')
```

Original tweet examples:

_“Set our hearts ablaze and every city was a gift And every skyline was like a kiss upon the lips @Â‰Ã›\_ https://t.co/cYoMPZ1A0Z“_

_“@PhDSquares #mufc they’ve built so much hype around new acquisitions but I doubt they will set the EPL ablaze this season.“_

Cleaned tweet examples:

_“Set our hearts ablaze and every city was a gift And every skyline was like a kiss upon the lips“_

_“#mufc they’ve built so much hype around new acquisitions but I doubt they will set the EPL ablaze this season.“_

Training and Hold-out Validation Sets
-------------------------------------

Twenty percent (1,523) tweets will be kept aside as a hold-out validation set (Validation set), where we will evaluate our classifiers. The split will be done without shuffling the data, where the top eighty percent (6,090) of the original training set will be kept as our object of study (Training set).

Keeping the original order of the tweets for training and validation split is fundamental not to introduce bias in candidate sample selection.

```
**\# Training and hold-out validation sets****\# 80/20 training/validation split**  
train\_size = int(0.8 \* raw\_data.shape\[0\])train\_index = \[x for x in range(0, train\_size)\]  
val\_index = \[x for x in range(train\_size, raw\_data.shape\[0\])\]X\_train = raw\_data.loc\[train\_index, 'cleaned\_text'\]  
y\_train = raw\_data.loc\[train\_index, 'target'\]X\_val = raw\_data.loc\[val\_index, 'cleaned\_text'\]  
y\_val = raw\_data.loc\[val\_index, 'target'\]print(f"Training set shape: {X\_train.shape}")  
print(f"Training label shape: {y\_train.shape}")  
print(f"Validation set shape: {X\_val.shape}")  
print(f"Validation label shape: {y\_val.shape}")
```

**Figure 3.** Training and Validation set dimensions.

We can now look at the maximum length of each individual tweet in terms of the number of words.

```
\# Training and validation maximum length sequencesprint("Training set maximum length:", max(\[len(row.split()) for row in X\_train\]))print("Validation set maximum length:", max(\[len(row.split()) for row in X\_val\]))
```

**Figure 4.** Training and Validation set maximum tweet lengths.

TF-IDF Vectorization
--------------------

To convert word strings into a vectorized numeric representation, we will use a **Term Frequency, Inverse Document Frequency** (TF-IDF, Sparck Jones, K. 1972) vectorizer \[8\].

As a well-known and widely implemented vectorization form for text data \[4\], it measures the term frequency of every observed word in the corpus vocabulary to assign it a weighted metric that is then penalized by the inverse frequency of the documents that contain the given word. It is an effective form of driving attention from frequently used words, terms, and expressions while keeping relevant the important descriptors of a particular group of documents.

The following equations \[8\] are used to calculate the term frequency (tf), inverse document frequency (idf), and TF-IDF score (tfidf), respectively:

**Formula 1.** Term Frequency.**Formula 2.** Inverse Document Frequency.**Formula 3.** TF-IDF score.

Where:

*   𝑓: Frequency
*   𝑡: Term (word)
*   𝑑: Document
*   𝐷: Corpus
*   𝑁: Number of Documents (d) in Corpus (D)
*   𝑛𝑡: Number of Documents (d) having Term (t)

The TF-IDF vectorization uses a one-hot encoding form of transformation. Each vocabulary word is assigned to a column, and its TF-IDF weighted score is stored in the column when the term is present in the original text. As a result, **each tweet is now represented in a vector space of _m_** **dimensions**, where _m_ is the size of the vocabulary present in the corpus.

We will use Scikit-Learn TfidfVectorizer implementation in our cleaned training and validation sets. It is important to observe that we will call a fit\_transform() method in our training set to generate our vector space. We then vectorize both our training (already transformed via the fit\_transform() method) and validation set via the transform() method.

```
**\# TF-IDF Vectorization**  
max\_features = Nonevectorizer = TfidfVectorizer(max\_features = max\_features)**\# Fitting vectorizer to training set vocabulary**  
X\_train\_tfidf = vectorizer.fit\_transform(X\_train)**\# Transforming the validation set**  
X\_val\_tfidf = vectorizer.transform(X\_val)print(f"TF-IDF normalized training set shape: {X\_train\_tfidf.shape}")  
print(f"TF-IDF normalized validation set shape: {X\_val\_tfidf.shape}")
```

**Figure 5.** TF-IDF Training and Validation set dimensions.

The TF-IDF vectorizer fitted in our 6,090 training tweets identified 13,053 unique words and was then used to transform the 1,523 observations in the validation set.

### A Note on Pre-trained Word Embeddings and Language Models

At this point, you might be thinking that with the rapid recent development in the NLP field we have great proven alternatives to better capture a semantic representation of our tweets, ranging from well-known pre-trained word embeddings such as [GloVe](https://nlp.stanford.edu/projects/glove/) (Pennington et al. 2014), to large language models for word representations, such as the many variations of [BERT embeddings](https://pypi.org/project/bert-embedding/#:~:text=Bert%20Embeddings,from%20BERT's%20pre%2Dtrained%20model.) (Devlin et al. 2019).

However, our goal is to prove the effectiveness of partitional clustering as a technique to sub-sample our data for training. For this reason, we will try to avoid any bias that might unexpectedly emerge from these intricate semantic representations and instead strict to a simple approach derived from our data.

Dimensionality Reduction
------------------------

Optimizing the K-Means, our clustering method of choice, objective function with Lloyd’s algorithm is an **NP-hard problem** \[5\], and even in its implementation on a K-Means standard algorithm with a fixed number _t_ of iterations, **it can be very slow to converge**, as its complexity increases linearly with the product of **O(t\*k\*n\*m)** where _t_ is the number of iterations, _k_ is the number of clusters, _n_ is the number of observations, and _m_ the number of dimensions (_features_) in our data.

You can imagine that our TF-IDF vectorized data with more than 13,000 dimensions and 6,000 tweets can be problematic, to say the least. Hence, to make the dataset suitable for clustering, we will implement a dimensionality reduction technique.

First, through a **Singular Value Decomposition** (SVD), the data will be reduced to its first _r_ components that better capture the variance presence in our TF-IDF vectorized training set. Then, we will **normalize the resulting vectors to unit L2-norm**. When applied to text data, these two techniques together are also known as a **Latent Semantic Analysis** \[1\].

To select the ideal number of _r_ components, we will use a method introduced by Gavish and Donoho, 2014 \[2\]. In summary, we will find the [**optimal Singular Value Hard Threshold**](https://arxiv.org/abs/1305.5870) (SVHT) based on the distribution of the log of Singular Values (captured variance) in the SVD. By truncating the SVD results on this given value, it is said to effectively remove the components that mainly contain noise from the original data \[2\].

Below is a [Python implementation](http://www.pyrunner.com/weblog/2016/08/01/optimal-svht/) of SVHThresholding, followed by a visualization of our selected components _r_.

```
**\# Singular Value Decomposition with Numpy SVD**  
U, S, VT = np.linalg.svd(X\_train\_tfidf.toarray(), full\_matrices=0)**\# Optimal Hard Thresholding (Gavish and Donoho, 2014)**  
def omega\_approx(X):  
  """  
 **Return an approximate omega value for given matrix X. Equation (5)   from Gavish and Donoho, 2014.**  """  
  beta = min(X.shape) / max(X.shape)  
  omega = 0.56 \* beta\*\*3 - 0.95 \* beta\*\*2 + 1.82 \* beta + 1.43 return omega**\# Defining singular value hard threshold**  
def svht(X, sv = None):  
  """  
 **Return the optimal singular value hard threshold (SVHT) value. \`X\` is any m-by-n matrix. \`sigma\` is the standard deviation of the noise, if known. Optionally supply the vector of singular values \`sv\` for the matrix (only necessary when \`sigma\` is unknown). If \`sigma\` is unknown and \`sv\` is not supplied, then the method automatically computes the singular values.**  
  """  
  sv = np.squeeze(sv)  
  if sv.ndim != 1:  
    raise ValueError('vector of singular values must be 1-dimensional')  
  return np.median(sv) \* omega\_approx(X)cutoff = svht(X\_train\_tfidf.toarray(), sv = S) **\# Hard threshold**  
r\_opt = np.max(np.where(S > cutoff)) **\# Keep modes w/ sig > cutoff**print(f"Optimum number of eigen values: {r\_opt}")
```**Figure 6.** SVHT optimum number of components r.```
**\# Plot TF-IDF vectorized dataset singular values**  
N = X\_train\_tfidf.shape\[0\]plt.figure(figsize = (10, 6))  
plt.semilogy(S\[:5400\], '-', color = 'k', LineWidth = 2) **\# Narrowing the visualization to the first relevant components**  
plt.semilogy(np.diag(S\[:(r\_opt + 1)\]), 'o', color = 'blue', LineWidth = 2)  
plt.plot(np.array(\[0, S.shape\[0\]\]),np.array(\[cutoff, cutoff\]), '--', color='r', LineWidth = 2)  
plt.title("TF-IDF training set log of Singular Values", size = 14, fontweight = "bold")  
plt.xlabel("Components")  
plt.ylabel("log of Singular Values")  
plt.show()
```

**Figure 7.** TF-IDF training set log of Singular Values.

We can then truncate our SVD to our r = 652 components using Scikit-Learn TruncatedSVD, and define our LSA pipeline by following up with an L2 normalizer to fit and transform our training and validation sets.

```
**\# Singular Value Decomposition to reduce dimensionality truncated at optimum hard threshold**  
svd = TruncatedSVD(n\_components = r\_opt, random\_state = 123)**\# Standard normalization for spherical K-Means clustering**  
normalizer = Normalizer(copy = True)**\# Latent Semantic Analysis dimensionality reduction pipeline**  
lsa = make\_pipeline(svd, normalizer)X\_train\_lsa = lsa.fit\_transform(X\_train\_tfidf)  
X\_val\_lsa = lsa.transform(X\_val\_tfidf)explained\_variance = svd.explained\_variance\_ratio\_.sum()  
print(f"Explained variance of the SVD step: {explained\_variance \* 100:.2f}%")print(f"LSA training set shape: {X\_train\_lsa.shape}")  
print(f"LSA validation set shape: {X\_val\_lsa.shape}")
```

**Figure 8.** Training and Validation set dimensions after LSA.

We can see that our 652 components found as the optimal _r_ explain 53.75% of the variance in the dataset. This number might seem low, but they are important descriptors of our data, whereas the remaining non-used ones can be mostly attributed to the noise in our dataset \[2\].

It is important to observe that **dimensionality reduction was solely used to reduce computation time during clustering**. Therefore, we are only implementing it in our training set when applying K-Means, from where the representative samples will be extracted for labeling. We will use the original TF-IDF vectorized training and validation sets for training and inference during the classification tasks.

Classification Models
---------------------

To validate our representative sampling method we will use two distinct classification algorithms and task them with classifying our tweets as a disaster (1) or non-disaster (0). **Logistic Regression** and a **K-Nearest-Neighbors Classifier**. Once again, as the focus of this study does not lie on the classification task, but on the effectiveness of partitional clustering as a method to select training samples, we will use these algorithms in their simplest and standard forms.

To set a baseline of the maximum achievable performance we will fit our classifiers to all 6,090 instances of our training set, and evaluate them on the 1,523 samples in our validation set.

After that, we will use a reduced number of samples for training and re-assess our classifiers against the same 1,523 validation observations. We will select these training samples via two methods:

*   **Naively**, taking the **first K observations in the training set**
*   With **representative instances**, picking the K observations closest to the centroids of K groups clustered with a K-Means algorithm.

Let’s code our evaluation routine, where we will:

1.  Fit a LogisticRegression and a KNeighborsClassifier
2.  Evaluate on our validation set and print out a classification report
3.  Calculate and instantiate the resulting F1-score and Accuracy metrics

```
**\# Defining evaluate function**  
def evaluate(X\_train, y\_train, X\_val, classif\_report = True):  
  '''  
 **Plot classification report for given training set using Logistic Regression and K-Nearest Neighbors classifiers.**  
  '''  
  log\_r = LogisticRegression()  
  log\_r.fit(X\_train, y\_train)  
  y\_pred\_log\_r = log\_r.predict(X\_val) knn\_c = KNeighborsClassifier()  
  knn\_c.fit(X\_train, y\_train)  
  y\_pred\_knn\_c = knn\_c.predict(X\_val) if classif\_report:  
    print('Logistic Regression classification report: \\n',  
          classification\_report(y\_val, y\_pred\_log\_r, digits = 3),  
      '\\n=====================================================\\n') print('K-Nearest Neighbors vote classification report: \\n',  
          classification\_report(y\_val, y\_pred\_knn\_c, digits = 3)) log\_r\_f1 = f1\_score(y\_val, y\_pred\_log\_r)  
  log\_r\_acc = accuracy\_score(y\_val, y\_pred\_log\_r) knn\_c\_f1 = f1\_score(y\_val, y\_pred\_knn\_c)  
  knn\_c\_acc = accuracy\_score(y\_val, y\_pred\_knn\_c) return log\_r\_f1, log\_r\_acc, knn\_c\_f1, knn\_c\_acc
```

Baseline Performance
--------------------

We can then evaluate our model on our complete training set.

```
**\# Evaluating performance using the complete training set**  
evaluate(X\_train\_tfidf, y\_train, X\_val\_tfidf)
```

**Figure 9.** Classification report on fully-trained models.

And summarize our baseline metrics for the fully trained models in the following table.

**Table 1.** Baseline metrics on fully-trained models.

Naive vs. Representative Samples
--------------------------------

Let’s set K = 300, which is roughly 5% of our complete training set. You can do a mental exercise and imagine the effort of reading and annotating 300 tweets, and you will start to understand the motivations behind this study.

The following code will select **the first 300 observations** from our dataset and use them to fit our classifiers through the evaluation routine.

```
**\# Evaluating performance with a reduced training set**  
n\_labeled = 300  
evaluate(X\_train\_tfidf\[:n\_labeled\], y\_train\[:n\_labeled\], X\_val\_tfidf)
```

**Figure 10.** Classification reports on the naively reduced training set. K = 300.

As expected, we see an expressive drop in our metrics. We can see that the **Logistic Regression is barely improving on a random guess** with 54.0% accuracy and far-from-workable F1 score and recall metrics.

So let’s cluster our dataset and extract 300 representative samples instead. To do so, we will fit\_transform our LSA training set using Scikit-Learn KMeans with K = 300. We will then map the closest observations to each cluster center to their ground truth representatives in the original data. Remember, the dimension-reduced version is used solely for clustering. We want to train our algorithms on the high-dimensional samples.

```
**\# K-Means++ clustering**  
k = 300kmeans = KMeans(n\_clusters = k, random\_state = 123)  
X\_clustered = kmeans.fit\_transform(X\_train\_lsa)**\# Representative tweets**  
representative\_ids = np.argmin(X\_clustered, axis = 0)  
X\_representative = X\_train\[representative\_ids\]  
X\_representative\_tfidf = X\_train\_tfidf\[representative\_ids\]**\# Representative tweets' labels**  
y\_representative = raw\_data.loc\[X\_representative.index, 'target'\]
```

We can now train on our representative samples and evaluate using our validation set.

```
**\# Evaluating cluster centered representative tweets**  
evaluate(X\_representative\_tfidf, y\_representative, X\_val\_tfidf)
```

**Figure 11.** Classification reports on the representative samples training set. K = 300.

And summarize our results in the following table.

**Table 2.** Evaluation metrics on representative (R) and naive (N) samples. K = 300.

It is possible to observe that the accuracy increased by approximately 7% on both classifiers. As expected, the 300 representative samples provide more meaningful information about our dataset and therefore are highly performant than simply selecting the first 300 observations for training.

But we can improve even further.

Label Propagation
-----------------

Similar to the results seen in (Géron, A. 2019) \[3\], further performance improvement can be achieved by propagating the labels of the selected representative samples to their surrounding observations.

Label propagation assumes that the neighbouring individuals to our representative samples are likely to share the same category. The image below illustrates the idea.

**Figure 12.** Label propagation.

In the following code, we will propagate the labels assigned to our representative samples to their 3% surrounding observations. Since we have the ground truth for the complete dataset, we can also check the propagation accuracy (how many labels were correctly assigned through this method).

```
**\# Propagating cluster labels**  
y\_propagated = np.empty(len(X\_train), dtype = np.int32)  
for i in range(k):  
  y\_propagated\[kmeans.labels\_ == i\] = list(y\_representative)\[i\]**\# Approximation percentile**  
percentile\_closest = 3X\_cluster\_dist = X\_clustered\[np.arange(len(X\_train)), kmeans.labels\_\]**\# Propagating based on approximation percentile**  
for i in range(k):  
  in\_cluster = (kmeans.labels\_ == i)  
  cluster\_dist = X\_cluster\_dist\[in\_cluster\]  
  cutoff\_distance = np.percentile(cluster\_dist, percentile\_closest)  
  above\_cutoff = (X\_cluster\_dist > cutoff\_distance)  
  X\_cluster\_dist\[in\_cluster & above\_cutoff\] = -1partially\_propagated = (X\_cluster\_dist != -1)  
X\_partially\_propagated = X\_train\_tfidf\[partially\_propagated\]  
y\_partially\_propagated = y\_propagated\[partially\_propagated\]print(f'Propagated training set shape: {X\_partially\_propagated.shape}')print(f'Propagation accuracy: {np.mean(y\_partially\_propagated == y\_train\[partially\_propagated\]) \* 100:.2f}%')
```

**Figure 13.** Training set with propagated labels and propagation accuracy.

Label propagation increased our training set to 680 samples, with a propagation accuracy of about 90%. It is important to note that outside of this experimentation scenario you will not have ground truths to verify the quality of your label propagation, and this will always be a trade-off between the number of training samples vs. the quality of labeled data.

Let’s see how it does on our classification algorithms.

```
**\# Evaluating partially propagated set**  
evaluate(X\_partially\_propagated, y\_partially\_propagated, X\_val\_tfidf)
```

**Figure 14.** Classification reports on the partially propagated training set.**Table 3.** Evaluation metrics on the partially propagated training set.

Our accuracy went up to 71.2% on the Logistic Regression (A 9.5% increase), and the F1 Score almost doubled. Meanwhile, our K-Nearest-Neighbors classifier’s accuracy oscillated down to 68.2% (A 0.5% decrease), with no significant change in the F1 Score.

We were able to break 70% accuracy and 0.65 F1 score with **twenty times fewer labeled instances**.

Optimum Number of Clusters
--------------------------

So far, we have worked with an arbitrarily defined number of clusters K = 300. As it is closely related to the amount of resources you have available for labeling, we are going to study the effects of varying K, and investigate where we will start to see diminishing returns for our labeling efforts.

We will also evaluate clustering performance with a Silhouette Width \[7\] metric, as described by the formula:

**Formula 4.** Silhouette Width.

Where:

*   a(i): Average distance between the _ith_ observation and other observations in the same cluster
*   b(i): Average distance between the _ith_ observation and other observations in the nearest cluster
*   S(i): silhouette score of the _ith_ observation.

The silhouette score gives us a quantifiable idea of cohesion within a cluster and is measured individually for each observation. It ranges from -1 to 1, where a higher value indicates that the object is highly similar to its own cluster \[7\].

In the following code we will iterate through a list of increasing numbers for K, and for each configuration, we will fit **ten independent runs of the K-Means algorithm, each with three random initializations**. This will ensure that we minimize any bias due to the random initializations of the cluster centers.

On each run, we will store the initialization number, the mean silhouette score for a given K, and its corresponding cluster characteristics, such as instance labels and distance metrics.

```
clusters = {}  
clusters\['k'\] = \[\]clusters\['init'\] = \[\]  
clusters\['distances'\] = \[\]  
clusters\['labels'\] = \[\]  
clusters\['silhouette\_width'\] = \[\]K\_list = \[20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000\]**\# Setting NumPy random seed for reproducibility**  
np.random.seed(123)  
for k in K\_list: **\# Fitting 10 random initializations of k-means clustering**  
  tic = time()  
  silhouette\_list = \[\]  
    
  for i in range(10):  
    clusters\['k'\].append(k)  
    clusters\['init'\].append(i + 1)  
    minibatch\_kmeans  
    kmeans = KMeans(n\_clusters = k, n\_init = 3)  
    distances = kmeans.fit\_transform(X\_train\_lsa)  
    clusters\['distances'\].append(distances)  
    labels = kmeans.labels\_  
    clusters\['labels'\].append(labels) silhouette\_width = silhouette\_score(X\_train\_lsa, labels)  
    clusters\['silhouette\_width'\].append(silhouette\_width)  
    silhouette\_list.append(silhouette\_width) silhouette\_avg = np.mean(silhouette\_list)  
  toc = time()  
  elapsed = toc - tic print(f"For K = {k}",  
        f"the average silhouette width is: {silhouette\_avg:.4f}.",  
        f"Elapsed time {elapsed:.3f}s")
```

**Figure 15.** Silhouette Width and elapsed time for varying K.

We can see that our **average silhouette width increases consistently with an increasing number of clusters**. This indicates that our dataset holds enough semantic diversity to continue clustering even further, but as noted earlier and made evident above, the time complexity is linearly increasing with the number of clusters K.

So let’s fit our classifiers to the representative samples found within each K-Means initialization. In the code below, we will run our evaluation routine on each individual run of our clustering step and store our performance metrics. Since we had multiple independent runs for each value of K, it will be possible to, later on, visualize a confidence interval for our metrics. A naive selection of samples will also be evaluated on each iteration for comparison purposes.

```
**\# Evaluation routine and comparison with naive labeling method**  
eval = {}  
eval\['k'\] = \[\]  
eval\['init'\] = \[\]  
eval\['classifier'\] = \[\]  
eval\['method'\] = \[\]  
eval\['representatives'\] = \[\]  
eval\['f1\_score'\] = \[\]  
eval\['accuracy'\] = \[\]for k, init, distance in zip(clusters\['k'\], clusters\['init'\], clusters\['distances'\]):  
  eval\['k'\].append(k)  
  eval\['init'\].append(init) representatives = np.argmin(distance, axis = 0)  
  eval\['representatives'\].append(representatives)  
  X\_representative\_tfidf = X\_train\_tfidf\[representatives\]  
  y\_representative = raw\_data.loc\[representatives, 'target'\] **# Representative Labeling - Accuracy and F1 score**  
  log\_r\_f1, log\_r\_acc, knn\_c\_f1, knn\_c\_acc = evaluate(X\_representative\_tfidf,  
         y\_representative,  
         X\_val\_tfidf,  
         classif\_report = False) **# Logistic Regression**  
  eval\['classifier'\].append('Logistic Regression')  
  eval\['method'\].append('Representative Labeling')  
  eval\['f1\_score'\].append(log\_r\_f1)  
  eval\['accuracy'\].append(log\_r\_acc)  
    
 **# KNN Classifier**  
  eval\['k'\].append(k)  
  eval\['init'\].append(init)  
  eval\['representatives'\].append(representatives)  
  eval\['classifier'\].append('KNN Classifier')  
  eval\['method'\].append('Representative Labeling')  
  eval\['f1\_score'\].append(knn\_c\_f1)  
  eval\['accuracy'\].append(knn\_c\_acc) **# Naive Labeling - Accuracy and F1 score**  
  log\_r\_f1\_n, log\_r\_acc\_n, knn\_c\_f1\_n, knn\_c\_acc\_n = evaluate(X\_train\_tfidf\[:k\],  
         y\_train\[:k\],  
         X\_val\_tfidf,  
         classif\_report = False) **# Logistic Regression**  
  eval\['k'\].append(k)  
  eval\['init'\].append(init)  
  eval\['representatives'\].append(representatives)  
  eval\['classifier'\].append('Logistic Regression')  
  eval\['method'\].append('Naive Labeling')  
  eval\['f1\_score'\].append(log\_r\_f1\_n)  
  eval\['accuracy'\].append(log\_r\_acc\_n) **# KNN Classifier**  
  eval\['k'\].append(k)  
  eval\['init'\].append(init)  
  eval\['representatives'\].append(representatives)  
  eval\['classifier'\].append('KNN Classifier')  
  eval\['method'\].append('Naive Labeling')  
  eval\['f1\_score'\].append(knn\_c\_f1\_n)  
  eval\['accuracy'\].append(knn\_c\_acc\_n)**\# clusters and eval as dataframes**  
eval\_df = pd.DataFrame(eval)  
clusters\_df = pd.DataFrame(clusters)
```

We can now start to plot our results to better visualize the effects of K on our model performances.

```
**\# KNN vs. Logistic Regression  
\# Accuracy  
**plt.figure(figsize = (12, 8))  
sns.lineplot(x = "k", y = "accuracy", hue = "classifier",  
             style = "method", data = eval, markers = True)**\# Horizontal line at K = 900**  
plt.axvline(x = 900, color = "k", dashes = (3, 1), linewidth = 2, zorder = 0)  
plt.text(920, 0.455, "K = 900", size = 12, fontweight = "bold")**\# Anotation at K = 1000**  
plt.scatter(1000, 0.735, marker = "v", c = "C0", alpha = 0.7, s = 120, zorder = 3)  
plt.text(1000, 0.743, "K = 1000", size = 11, fontweight = "bold", c = "C0", horizontalalignment = "center")  
plt.suptitle("KNN Classifier vs. Logistic Regression", size = 14, fontweight = 'bold', y = 0.94)  
plt.title("Classification Accuracy")  
plt.ylabel("Accuracy")  
plt.xlabel("Number of clusters K")  
plt.show()
```

**Figure 16.** KNN Classifier vs. Logistic Regression — Classification Accuracy.

```
**\# F-1 Score**  
plt.figure(figsize = (12, 8))  
sns.lineplot(x = "k", y = "f1\_score", hue = "classifier",  
             style = "method", data = eval, markers = True)**\# Anotation at K = 300**  
plt.scatter(300, 0.665, marker = "v", c = "C1", alpha = 0.7, s = 120, zorder = 3)  
plt.text(300, 0.685, "K = 300", size = 11, fontweight = "bold", c = "C1", horizontalalignment = "center")  
plt.suptitle("KNN Classifier vs. Logistic Regression", size = 14, fontweight = 'bold', y = 0.94)  
plt.title("F-1 Score")  
plt.ylabel("F-1 Score")  
plt.xlabel("Number of clusters K")  
plt.show()
```

**Figure 17.** KNN Classifier vs. Logistic Regression — F1 Score.

The plots show the results for both classifiers trained with samples selected naively (dashed lines) and with representative sampling (continuous line). The shaded area around the plot curve illustrates the 95% confidence interval of the ten independent initializations of the clustering algorithm.

It is evident that the representative labeling method results in a **stable performance increase on both metrics**. We can also see that the KNN Classifier has an early build-up of the performance metrics with a smaller number of samples, while the Logistic Regression tends to outperform it after being trained on 900 or more observations.

Another interesting takeaway is the significant drop in performance from 500 to 1000 naively selected training samples. What might be counter-intuitive is illustrated here, showing that you can double your labeling efforts and eventually end up with worse model performance.

Interpreting Clustering Results
-------------------------------

Visualizing and interpreting clustering results might be tricky when working with unstructured data. The high dimension and lack of interpretability of text data in its vectorized form make explainability hard even in simple studies.

In this section, I will use some plots to help us better understand what is being captured by our clustering algorithm, what it means to select representative samples, and what our clusters represent.

Since the coding of some of the following plots can be somewhat extensive, I will simply index them here and discuss the results. Still, if you wish to replicate them, you can have a complete look at the source code on my [GitHub repository](https://github.com/erich-hs/Tweets-Semi-Supervised) for the project.

If you recall from our Silhouette Width definition, it is measured on an _instance level_. As a result, we can visualize where are our best clusters in terms of average silhouette score and look at their distribution through a silhouette width plot. Below is the plot of our training set when K = 50 (therefore, 50 clusters) on its best K-Means run (initialization 7).

**Figure 18.** Silhouette Widths for K = 50, initialization 7.

The clusters are color-coded and enumerated from 1 to 50, and in the chart, you can also see the respective cluster sizes underneath their labels.

When investigated, clusters with silhouette widths closer to 1.0 captured in the same group similar or identical tweets, which are often related to the same topic or a particular event.

On the other hand, large and underperforming clusters (with a negative average silhouette score), such as cluster 20, are holding our “left-overs” tweets that did not fit appropriately with any other well-defined group. When increasing the number of clusters K you will see a systematic improvement in clustering performance, where these marginal observations start to find their own clusters.

Let’s take a look at some of the well-defined examples with higher average silhouette scores.

```
**\# Investigating cluster 12**  
investigate\_df = clusterTweets(k = 50, cluster = 12).head(10)  
for y, tweet in zip(investigate\_df\['ground\_truth'\], investigate\_df\['tweet'\]):  
  print(y, tweet)
```

**Figure 19.** Cluster 12 tweets under K = 50, initialization 7.

With a perfect silhouette score of 1.0, we can see that cluster 12 captured the spam of identical tweets talking about the same disaster event, in this case, a sandstorm hitting an airport.

```
**\# Investigating cluster 15**  
investigate\_df = clusterTweets(k = 50, cluster = 15).head(10)  
for y, tweet in zip(investigate\_df\['ground\_truth'\], investigate\_df\['tweet'\]):  
  print(y, tweet)
```

**Figure 20.** Cluster 15 tweets under K = 50, initialization 7.

Meanwhile, cluster 15’s tweets are mostly about a [train derailment in Madhya Pradesh](https://en.wikipedia.org/wiki/Harda_twin_train_derailment#:~:text=On%204%20August%202015%2C%20two,and%20100%20people%20were%20injured.).

```
**\# Investigating cluster 18**  
investigate\_df = clusterTweets(k = 50, cluster = 18).head(10)  
for y, tweet in zip(investigate\_df\['ground\_truth'\], investigate\_df\['tweet'\]):  
  print(y, tweet)
```

**Figure 21.** Cluster 18 tweets under K = 50, initialization 7.

And cluster 18 grouped the tweets about the finding of the debris of the tragic [Malaysia Airlines Flight 370](https://en.wikipedia.org/wiki/Malaysia_Airlines_Flight_370).

I have also set up a biplot and an auxiliary function to visualize the LSA components where each cluster had been most distinctively represented. Below is the biplot for the LSA components 10 and 13 (Out of our 652), where cluster 18 is noticeably shifted towards the upper right quadrant.

**Figure 22.** Clustering biplot of LSA components (10, 13).

Conclusions
-----------

It is now easy to build an intuition about what our representative sampling method is doing. When capturing the centroid observation of our clustering and labeling them for a training set, we expose our classifiers to the most distinct scenarios in our corpus. From the examples above, our algorithms would have learned from different disaster-related tweets, such as train derailments, aircraft debris, or sandstorms.

In these final bar charts, we can summarize our improvements in performance metrics. For K = 300 and K = 1000, I have plotted a side-by-side comparison between the Representative Labeling and Naive Labeling methods for both classifiers.

**Figure 23.** Representative vs. Naive labeling barplot. Logistic Regression.**Figure 24.** Representative vs. Naive labeling barplot. KNN Classifier.

Besides the evident improvements, it is important to end this article with some considerations. When implementing a similar approach for your own studies, bear in mind that:

1\. Representative labeling **might induce unwanted bias in the classification model** if K is too low for the diversity present in the original dataset.

2\. **Overfitting might occur earlier** during the training and validation process when using representative labeled samples since this method sets aside hard-to-define, diffuse, and less representative samples.

3\. Some of the  [**K-Means assumptions**](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html) for accurate clustering **are not proved during this method** due to the high dimensionality and number of clusters. Hence, unequal variance and anisotropicity problems might result in **inappropriate cluster centers** (and, therefore, inappropriate representative samples).

And that concludes our analysis of clustering for representative labeling with unstructured text data. You can find the development jupyter notebook on the [project repository](https://github.com/erich-hs/Tweets-Semi-Supervised) or directly through this [link](https://github.com/erich-hs/Tweets-Semi-Supervised/blob/master/Tweets_Semi_Supervised.ipynb). I hope you learned as much as I did while going through this study.

References
----------

\[1\] Dumais, S. T. (2005). Latent Semantic Analysis. In _Annual Review of Information Science and Technology_ (Vol. 38, pp. 188–230). essay, Journal of the American Society for Information Science.

\[2\] Gavish, M., & Donoho, D. L. (2014, June 4). _The optimal hard threshold for singular values is 4/sqrt(3)_. arXiv.org. Retrieved April 2, 2022, from [https://arxiv.org/abs/1305.5870](https://arxiv.org/abs/1305.5870)

\[3\] Géron, A. (2019). Chapter 9. Unsupervised Learning Techniques. In _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd Edition). essay, O’Reilly Media, Inc.

\[4\] Géron, A. (2019). _Hands-on machine learning with scikit-learn, keras and tensorflow: Concepts, tools and techniques to build Intelligent Systems._ O’Reilly.

\[5\] Lu, Y., & Zhou, H. H. (2016, December 7). _Statistical and computational guarantees of Lloyd’s algorithm and its variants_. arXiv.org. Retrieved September 20, 2022, from [https://arxiv.org/abs/1612.02099](https://arxiv.org/abs/1612.02099)

\[6\] _Natural language processing with disaster tweets._ Kaggle. (n.d.). Retrieved April 2, 2022, from [https://www.kaggle.com/c/nlp-getting-started/overview](https://www.kaggle.com/c/nlp-getting-started/overview)

\[7\] Rousseeuw, P. J. (2002, April 1). _Silhouettes: A graphical aid to the interpretation and validation of cluster analysis_. Journal of Computational and Applied Mathematics. Retrieved September 22, 2022, from [https://www.sciencedirect.com/science/article/pii/0377042787901257?via%3Dihub](https://www.sciencedirect.com/science/article/pii/0377042787901257?via%3Dihub)

\[8\] SPARCK JONES, K. A. R. E. N. (1972). A statistical interpretation of term specificity and its application in retrieval. _Journal of Documentation, 28_(1), 11–21. [https://doi.org/10.1108/eb026526](https://doi.org/10.1108/eb026526)
