---
title: "Metric Learning for Landmark Image Recognition"
tags: ['Computer Vision', 'Metric Learning']
date: 2022-10-31T11:57:32-08:00
draft: false
---

<!-- Metric Learning for Landmark Image Recognition
============================================== -->
A complete TensorFlow implementation of global descriptors similarity search with local feature re-ranking

![Colosseum](/images/posts/landmark_image/Colosseum_by_Hank_Paul.jpg)
**Figure 1.** Colosseum — Photo by [Hank Paul](https://unsplash.com/@henrypaulphotography?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/@henrypaulphotography?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText")

Metric learning for instance recognition and information retrieval is a technique that has been widely implemented across multiple fields. It is a concept that is highly relevant to novel applications in research, such as the latest AI breakthrough in biology \[2\] with [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) \[11\] by DeepMind, and also mature and well-proven to see vast implementation in the industry, from contextual information retrieval in Google Search \[12\], to image similarity for face recognition \[7\], that you might use every day to unlock your phone. In this article, I will go through a complete example of an image querying architecture that is the foundation of modern solutions to one of today’s challenges in computer vision — **Landmark recognition**.

The Goal of this Article
------------------------

In this article introduction, I listed some examples that should give you an idea of how relevant metric learning for similarity search is to modern machine learning. In the following sections, I will guide you through a sample implementation of the technique through a baseline solution for the **landmark recognition task.** We will use a subset of Google Landmarks Dataset v2 \[8\], which from now on, I will refer to as GLDv2.

The solution to this problem can be divided into two tasks: _Image retrieval_ and _instance recognition_. The task of retrieval is to rank images in an index set according to their relevance to a query image. The recognition task is to identify which specific instance of an object class (_e.g._ the instance “Mona Lisa” of the object class “painting”) is shown in a query image \[8\]. As shown in the benchmarks of the [GLDv2 dataset paper](https://arxiv.org/abs/2004.01804), state-of-the-art approaches use some extent of **global feature similarity search** paired with **local feature matching re-ranking**. Here, I aim to explain, illustrate and implement these concepts, and I hope to give you a clearer idea of how to extend them to your own applications.

Google Landmarks Dataset v2
---------------------------

The dataset was introduced by Google in 2020 and was motivated by the rapid development of deep learning approaches to the landmark recognition task. Previous benchmark datasets, such as the [Oxford5k](https://www.robots.ox.ac.uk/~vgg/publications/papers/philbin07.pdf) and [Paris6k](https://www.robots.ox.ac.uk/~vgg/publications/papers/philbin08.pdf), were struggling to keep up with new solutions \[8\] and are not a great resource to ensure scalability and generalization since they hold few query images of low amounts of instances from single cities.

To define a new challenging benchmark, the GLDv2 was proposed as the largest dataset to date, with over 5,000,000 images and 200,000 distinct instance labels (classes or landmarks). The test set contains around 118,000 query images with ground truth annotations, and, most importantly, only 1% of the images are actually within the target domain of landmarks. The other 99% are out-of-domain, unrelated images \[8\]. Adding to that, it has two core characteristics designed to test model robustness:

1.  **Extremely skewed class distribution**. While famous landmarks might have tens of thousands of image samples, 57% of classes have at most ten images, and 38% of classes have at most five images.
2.  **Intra-class variability**. Landmarks have views from different vantage points and of different details, as well as both indoor and outdoor views of buildings.

![gldv2](/images/posts/landmark_image/gldv2.png)


**Figure 2.** Google Landmarks Dataset v2 long-tailed class distribution \[8\].

While this article, for illustrative purposes, will use a **subset with 75 classes and 11,438 landmark pictures** from the original GLDv2 training set, we will still have to deal with some of the challenges above.

With the release of GLDv2 (and the previous GLDv1), Google sponsored a series of Kaggle competitions, including [the 2020 edition](https://www.kaggle.com/c/landmark-recognition-2020) \[9\], in which top-ranked solutions inspired the architecture illustrated here. If you want to know more about the GLDv2, I recommend going through the [dataset repository](https://github.com/cvdfoundation/google-landmark) and [paper](https://arxiv.org/abs/2004.01804). You can also explore the dataset visually [here](https://storage.googleapis.com/gld-v2/web/index.html).

Baseline Architecture
---------------------

Our model architecture was adapted from the [2020 Recognition challenge winner](https://arxiv.org/abs/2010.01650) \[10\] and [2019 Recognition challenge 2nd place](https://arxiv.org/abs/1906.03990) \[5\] papers and can be seen as a baseline solution to the landmark recognition task. The diagram below illustrates the training and retrieval routines with global feature search with local feature leverage for reranking, which we will cover in detail in the following sections.

![architecture1](/images/posts/landmark_image/Architecture.PNG)


**Figure 3.** Landmark recognition baseline architecture. Diagram by the author.

Google also optimized a similar architecture into the unified model DEep Local and Global features (DELG) \[4\]. I dedicated a short section where you can read more about it later on.

My complete Kaggle notebook with all the code in this article can be found [here](https://www.kaggle.com/code/erichhenrique/gldv2-2020-efficientnet-and-delf-reranking-tf/notebook). Consider leaving an upvote if you find it useful.

Libraries and Dependencies
--------------------------

Our implementation will use TensorFlow 2 and OpenCV as our core libraries. Around that, we will use NumPy and Pandas for data wrangling, SciPy for distance metrics, and matplotlib and seaborn for our visualizations.

Hosted notebook instances at Google Collab and Kaggle Notebooks come with all needed libraries pre-installed. For this mini project, however, I recommend working at a [Kaggle Notebook](https://www.kaggle.com/code) due to the easy access to the complete GLDv2 dataset without the need to download it (Especially if you want to experiment on the complete dataset, which is 105 GB in size).

Getting Started
---------------

To get started, we can confirm that our working environment is set up to run with GPU acceleration. You can verify if it is GPU enabled and check what is the current CUDA version with the bash command `nvida-smi`. With that done, we can start by importing our libraries.

```python
# Importing libraries
import os
import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from absl import logging
from PIL import Image, ImageOps
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
from scipy import spatial
from scipy.spatial import cKDTree
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
```

We can now define our dataset directory and .csv path. If you are working inside the Kaggle environment for the competition, the data will be distributed in a train and test folder, where each image is placed within three subfolders according to the first three characters of the image `id` (i.e. image `abcdef.jpg` is placed in `a/b/c/abcdef.jpg`). The directory also contains a train.csv file with training labels.

![data-exp](/images/posts/landmark_image/data-exp.png)


**Figure 4.** Kaggle GLDv2 folder structure.

We will proceed by reading the train.csv file and defining a column with the training image paths derived from their respective `id`.

```python
# Directories and file paths
TRAIN_DIR = '../input/landmark-recognition-2020/train'
TRAIN_CSV = '../input/landmark-recognition-2020/train.csv'
train_df = pd.read_csv(TRAIN_CSV)

TRAIN_PATHS = [os.path.join(TRAIN_DIR,\
    f'{img[0]}/{img[1]}/{img[2]}/{img}.jpg')\
    for img in train_df['id']]
train_df['path'] = TRAIN_PATHS

train_df
```

![train-df](/images/posts/landmark_image/train_df.png)


**Figure 5.** Train DataFrame with image paths.

We will then define our subset to work with. We will work on a small subset of the data to keep the experiments manageable in a feasible amount of time, both for training and retrieval with cosine similarity search. This subset is defined by landmarks with at least 150 and no more than 155 images per class. We will also assign a new, more interpretable landmark id to each class.

```python
# Subsetting
train_df_grouped = pd.DataFrame(train_df.landmark_id.value_counts())
train_df_grouped.reset_index(inplace=True)
train_df_grouped.columns = ['landmark_id','count']

# Selected landmarks based on inclass frequency
selected_landmarks = train_df_grouped[(train_df_grouped['count'] <= 155) & \
    (train_df_grouped['count'] >= 150)]

train_df_sub=train_df[train_df['landmark_id'].isin(selected_landmarks['landmark_id'])]
new_id = []
current_id = 0
previous_id = int(train_df_sub.head(1)['landmark_id'])
for landmark_id in train_df_sub['landmark_id']:
    if landmark_id == previous_id:
        new_id.append(current_id)
    else:
        current_id += 1
        new_id.append(current_id)
        previous_id = landmark_id

train_df_sub['new_id'] = new_id

NUM_CLASSES = train_df_sub['landmark_id'].nunique()
train_df_sub
```

![subset](/images/posts/landmark_image/subset.png)


**Figure 6.** Subset with 11438 rows and 75 landmark classes.

We went down from 1,580,470 images to 11,438 distributed into 75 distinct landmark classes. If you want to take the challenge head-on and work on the complete dataset, there will be some optimization implementations that I recommend, especially in the cosine similarity search, but we will discuss this in a later section. For now, let's focus on the theory and core implementation of our baseline model.

Training, Test, and Validation Split
------------------------------------

We will do a stratified split to define our training, validation, and test sets. For that, we will use the sciki-learn `train_test_split` method while passing our label to the `stratify` argument. It will ensure that each of the 75 classes in our subset will be present at the training, validation, and test sets after the split.

**Figure 7.** Training, validation, and test shapes.

And we can now confirm that it is evenly distributed with some histograms on each subset.

**Figure 8.** Stratified training, validation, and test split distribution.

So far, we have been working on our DataFrame generated from the train.csv file. Now we have to deal with the actual images from the dataset. We will start by defining a new folder structure that places each of our subset images into a directory named after the landmark id. It is an important step that will allow us to use the TensorFlow `image_dataset_from_directory` function to create `tf.data.Dataset` objects.

We can now check that our new folder structure is in place.

**Figure 9.** New image directories.

And finally, create TensorFlow `tf.data.Dataset` from our training, test, and validation sets. They are objects that make data streaming throughout our pipeline highly efficient and will certainly help with performance moving forward.

The function `image_dataset_from_directory` can preprocess our image by resizing it with the `image_size` argument and pre-set our batch sizes for training and validation. So we will define these parameters as well.

Now our data is ready to work with. You can take a look at one of the training batches with the `.take()` method from the `dataset` object. Remember that our subsets were shuffled during our initial split.

**Figure 10.** Sample batch from the training dataset.

From here, you can see that some of the challenges idealized for the GLDv2 dataset are still present in our subset. With high intra-class variability, we can see pictures from the same landmark that were taken from indoor and outdoor views. There are also photos with indirect relevance to its class, such as the one from _landmark 41_ in the image above, representing a museum piece that is probably inside the actual landmark depicted in the dataset.

Augmentation Layer
------------------

We will use image augmentation to help with generalization and keep the model from overfitting to the training data. With knowledge of the nature of our problem, we can define what works and what does not in this step. An ideal landmark recognition and retrieval model should be able to identify the place on an instance level from non-professional pictures taken from different angles. The following code snippet will define a basic augmentation layer that applies random translation, random rotation, and random zoom to the training image. It is important to notice that the augmentation layer is bypassed during inference, which means it will only preprocess the images during training steps. With that in mind, we can take a look at one augmentation example by passing `training = True` to our augmentation layer.

**Figure 11.** Sample augmentation step.

One common augmentation we will refrain from using here is randomly mirroring or flipping the images on their vertical axis. While we know that our model should be translation and viewpoint invariant (it should be able to identify landmark instances at distinct locations in the picture and under different points of view), it should not be invariant to vertical symmetry. While some landmarks might, indeed, be symmetric around their vertical axis, most of them are not.

On the other hand, one augmentation technique that has proven to improve model performance \[10\] is random cutout. It randomly obfuscates small regions of the original image and, as the examples used above, is also an effective regularization method. You can find a succinct and highly informative Medium article about it [here](https://medium.com/@ombelinelag/cutout-regularization-for-cnns-62670d86bc33), written by [Ombeline Lagé](https://medium.com/u/922346f406cd?source=post_page-----6c1b8e0902bd--------------------------------).

With our augmentation layer in place, we can now define our classification model.

Global Descriptor
-----------------

If you peek back at our model architecture, you will notice one instrumental fact about similarity search with metric learning — It is not directly from our classification model that we infer the landmark label of a queried image. In this section, we will set up a pre-trained EfficientNet and use it to train our **embedding layer**, which we will later use to encode our query and key images into a 512-dimensional feature vector.

**Figure 12.** Global descriptor training architecture.

The embedding layer will be the one responsible for our **global descriptors**, and the diagram above shows the training architecture of our **global retrieval model**.

But what is a global image descriptor? In simple terms, it is an n-dimensional vector that works as an encoded version of our image tailored to a specific use case. The one in our example has 512 dimensions with optimized discriminative power to distinguish one landmark from another. It not only learns from general visual traits from the landmark but also from contextual information, such as background and foreground objects, lighting conditions, and vantage points.

While the following model will not be used for inference, the quality of our global descriptors is directly related to the classification model performance. For the highest-ranked solutions in recognition and retrieval competitions, these embedding layers are often trained on an ensemble of multiple ResNet CNNs. In the following example, we will implement a simple solution with a pre-trained EfficientNetB0, also known as a _backbone block_. Our classification head is then rebuilt on top of an Average Pooling, a Batch Normalization, and a Dropout layer for regularization.

The embedding layer is built right before the Sotfmax classification layer. It is also named `embedding_512` for later use. Notice how we freeze our EfficientNet block to keep the pre-trained weights. For improved performance, you can also fine-tune the top layers in your backbone (there is a section on fine-tuning in this [guide to transfer learning](https://keras.io/guides/transfer_learning/) in Keras documentation). We will keep it simple and proceed to train with an ADAM optimizer and a learning rate scheduler.

**Figure 13.** Train and validation loss.

And evaluate the best model performance.

**Figure 14.** Best model performance in the validation and test set.

With that, we have a trained embedding layer that we can now use to generate our global descriptors. This leads us to our retrieval step.

Cosine Similarity
-----------------

The core concept of metric learning revolves around the optimization of a _parametric distance_ or _similarity function_ over one or more **similarity (or dissimilarity) judgment**. A judgement, in the best definition of the word, is the considerate decision about the task at hand, and is often encoded as a target variable — In our example, the landmark class.

But the concept is not always straightforward. The example below from _“Similarity and Distance Metric Learning with Applications to Computer Vision.”_ (Bellet, A. et al. 2015) \[1\] illustrates a qualitative judgment that is not often directly quantifiable. These are the cases where supervised metric learning can be used to its best.

**Figure 15.** Example of a qualitative judgement fitting to a metric learning task. \[1\]

With the judgement defined, we can then implement a model architecture that is able to solve this optimization problem. In our example, we will use a **cosine similarity** as our distance metric, and as mentioned previously, the optimization occurs during the training of our global descriptor. The better (most descriptive) our embedding layers is, the closer we get to an optimum solution to our problem.

**Figure 16.** Similarity search with global features in the architecture diagram.

Cosine similarity (or angular cosine distance) is the measurement of the cosine of the angle between two vectors and can be described mathematically as the division between the dot product of vectors and the product of their lengths (or the product of their euclidean norm) \[3\].

**Figure 17.** Cosine Similarity mathematic representation \[3\].

In the formula above, _x_ and _y_ are the components of two independent vectors, and _n_ is their dimension.

In the following snippet, we implement a series of auxiliar functions to load and preprocess individual images, retrieve image embeddings with a given model (to which we will pass our `embedding_512` layer later on), calculate pairwise similarity between a query and a key set, and some visualization functions to plot our most similar candidate images.

> **Note on performance**: The cosine similarity implementation above is based on a simple SciPy sequential CPU run and is intended to be performed individually between a single vectorized query image against all vectorized key samples.
> 
> When working with a large-scale scenario (such as the complete dataset), you want to efficiently compute the cosine distance between a matrix of query vectors and a matrix of key samples. For that, you can implement a batch GPU run with TensorFlow tensors for a highly parallelized execution. You can look at [keras Cosine Similarity loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity) source code to have an idea on how to perform this operation with tensors.

For the next examples, we will use our training set as our key images, and from our validation set we will get some query samples. The following code will pass both subsets through out `embedding_layer` with the function `get_embeddings()`. It will generate our 512-dimensional global descriptor for each image.

So let’s use our `query_top()` function to query some images and visualize the results.

**Figure 18.** Queried image and highest similarity candidates.

We can see from the very first image in our query set that our global descriptor is effective and the results are highly relevant to our query image. They not only returned the correct landmark in all top five candidates but also images with similar vantage points and lighting conditions.

The following function performs a similar search but returns the results as a Pandas DataFrame.

That we can use to look at our cosine similarity scores.

**Figure 19.** Highest similarity candidates scores.

And we can repeat the process with other query image examples.

**Figure 20.** Queried image and highest similarity candidates example.**Figure 21.** Queried image and highest similarity candidates example.**Figure 22.** Queried image and highest similarity candidates example.

With that, we conclude our similarity search with global features. We have achieved excellent results so far, but let's take a look at an example where global descriptors lack in performance.

Reranking with Local Features
-----------------------------

Let’s take a look at an example of object occlusion.

**Figure 23.** Landmark occlusion example.

The landmark is not only occluded but is also occupying a fairly small region in the picture, which is dominated by the tree in the foreground. We can see from the results that it played a major role in the similarity search, and in most top results we also see a big tree in the image.

This is an example where our reranking with local features will play a major role.

**Figure 24.** Reranking with local features in the architecture diagram.

We will use a **DEep Local Feature (DELF)** \[13\]  module to extract attentive local feature descriptors from our query image, and compare it to the highest-ranked candidate images selected previously.

DELF is a convolutional neural network model that was trained on images of landmarks. It enables accurate feature matching and geometric verification, and with the model paper, Google announced the first Google-Landmarks dataset (GLDv1) \[13\]. You can read more in the [paper](https://arxiv.org/abs/1612.06321).

The following implementation was adapted from the [TensorFlow hub tutorial for the DELF module](https://www.tensorflow.org/hub/tutorials/tf_hub_delf_module). As it is a refinement step, we will set our image size to 600 x 600. The following code will load the pre-trained DELF model from the TensorFlow hub and define some functions to look for inliers (feature matches) between image pairs.

We can now loop through our previous results to look for inliers.

**Figure 25.** DELF correspondences on candidate images.

Local attentive features, as proposed by the DELF architecture, are powerful descriptors for similarity matching based on geometric attributes. You can see from the results above that, despite the difference in scale and resolution, it was able to identify the relevant features in the building architecture between the correct image pair.

So that leads us to our last step. We will rerank our candidate images from the global similarity search using the number of DELF correspondences found. The following function will recalculate the confidence index (which was previously the cosine similarity) by multiplying its current value by the square root of the number of inliers found using local features.

We can finally look at the reranked results for our example above.

**Figure 26.** Reranked confidence index.

**Figure 27.** Reranked candidate landmarks.

With that, we have covered the complete metric learning architecture for landmark recognition with global similarity search and reranking with local features.

I will leave below additional reranking examples and relevant supplementary reading in the following sections.

Reranking Examples
------------------

**Figure 28.** Queried image and highest similarity candidates without reranking.**Figure 29.** Reranked results with local features.**Figure 30.** Queried image and highest similarity candidates without reranking.**Figure 31.** Reranked results with local features.**Figure 32.** Queried image and highest similarity candidates without reranking.**Figure 33.** Reranked results with local features.

ArcFace Loss
------------

One of the challenges of working with a large-scale classification problem, such as the one in the GLDv2 dataset, is that a large amount of classes (that can increase over time in a real-world scenario) leads to a regular softmax loss function that lacks interclass separability. To solve a similar problem with face recognition algorithms, the ArcFace Loss was proposed in 2018, and today is highly adopted in landmark recognition algorithms.

ArcFace margin \[6\] is **an additive angular margin loss function that enforces smaller intra-class variance**. As opposed to the softmax loss function used here, it has proven to produce better global feature descriptors due to its enhanced discriminative power.

If you want to learn more about it, I highly recommend reading the [paper](https://arxiv.org/pdf/1801.07698.pdf) (Deng, J. et al. 2018) \[6\] and the excellent [kernel](https://www.kaggle.com/code/slawekbiel/arcface-explained) by Slawek Biel that provides visual intuition of ArcFace learned embeddings compared to usual softmax loss ones.

Google’s Unified DELG Model
---------------------------

DEep Local and Global Features (DELG — Cao, B. et al. 2020) \[4\] is Google’s proposed unified model that incorporates DELF’s attention module with a much simpler training pipeline that is integrated with global feature retrieval under the same architecture. It is essentially a single-model implementation of the architecture exemplified in this article.

References
----------

\[1\] Bellet, Aurélien, and Matthieu Cord. “Similarity and Distance Metric Learning with Applications to Computer Vision.” Lille University Research, September 7, 2015. [http://researchers.lille.inria.fr/abellet/talks/metric\_learning\_tutorial\_CIL.pdf.](http://researchers.lille.inria.fr/abellet/talks/metric_learning_tutorial_CIL.pdf.)

\[2\] Callaway, Ewen. “‘It Will Change Everything’: Deepmind’s Ai Makes Gigantic Leap in Solving Protein Structures.” Nature News. Nature Publishing Group, November 30, 2020. [https://www.nature.com/articles/d41586-020-03348-4.](https://www.nature.com/articles/d41586-020-03348-4.)

\[3\] “Cosine Distance Cosine Similarity Angular Cosine Distance Angular Cosine Similarity.” COSINE DISTANCE, COSINE SIMILARITY, ANGULAR COSINE DISTANCE, ANGULAR COSINE SIMILARITY. Accessed October 29, 2022. [https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/cosdist.htm.](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/cosdist.htm.)

\[4\] Cao, Bingyi, Araujo, Andre, and Jack Sim. “Unifying Deep Local and Global Features for Image Search.” _arXiv_, (2020). [https://doi.org/10.48550/arXiv.2001.05027.](https://doi.org/10.48550/arXiv.2001.05027.)

\[5\] Chen, Kaibing, Cui, Cheng, Du, Yuning, Meng, Xianglong, and Hui Ren. “2nd Place and 2nd Place Solution to Kaggle Landmark Recognition and Retrieval Competition 2019.” _arXiv_, (2019). [https://doi.org/10.48550/arXiv.1906.03990.](https://doi.org/10.48550/arXiv.1906.03990.)

\[6\] Deng, Jiankang, Guo, Jia, Yang, Jing, Xue, Niannan, Kotsia, Irene, and Stefanos Zafeiriou. “ArcFace: Additive Angular Margin Loss for Deep Face Recognition.” _arXiv_, (2018). [https://doi.org/10.1109/TPAMI.2021.3087709.](https://doi.org/10.1109/TPAMI.2021.3087709.)

\[7\] “Facial Recognition Is Everywhere. Here’s What We Can Do about It.” The New York Times. The New York Times, July 15, 2020. [https://www.nytimes.com/wirecutter/blog/how-facial-recognition-works/.](https://www.nytimes.com/wirecutter/blog/how-facial-recognition-works/.)

\[8\] “Google Landmarks Dataset v2 — A Large-Scale Benchmark for Instance-Level Recognition and Retrieval”, T. Weyand, A. Araujo, B. Cao and J. Sim, Proc. CVPR’20

\[9\] “Google Landmark Recognition 2020.” Kaggle. Google. Accessed October 28, 2022. [https://www.kaggle.com/c/landmark-recognition-2020.](https://www.kaggle.com/c/landmark-recognition-2020.)

\[10\] Henkel, Christof, and Philipp Singer. “Supporting large-scale image recognition with out-of-domain samples.” _arXiv_, (2020). [https://doi.org/10.48550/arXiv.2010.01650.](https://doi.org/10.48550/arXiv.2010.01650.)

\[11\] Jumper, J., Evans, R., Pritzel, A. _et al._ “Highly accurate protein structure prediction with AlphaFold.” _Nature_ **596**, 583–589 (2021). [https://doi.org/10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

\[12\] Nayak, Pandu. “Understanding Searches Better than Ever Before.” Google. Google, October 25, 2019. [https://blog.google/products/search/search-language-understanding-bert/.](https://blog.google/products/search/search-language-understanding-bert/.)

\[13\] Noh, Hyeonwoo, Araujo, Andre, Sim, Jack, Weyand, Tobias, and Bohyung Han. “Large-Scale Image Retrieval with Attentive Deep Local Features.” _arXiv_, (2016). [https://doi.org/10.48550/arXiv.1612.06321.](https://doi.org/10.48550/arXiv.1612.06321.)