---
title: "Table Data from Images"
subtitle: "Clustering for Layout Matching"
categories: ['Data Science']
tags: ['Computer Vision', 'OCR', 'Web Scraping', 'Python']
date: 2022-09-15T11:57:32-08:00
draft: false

featuredImagePreview: "/images/posts/table_data_from_images/fig1.png"
---
{{< figure src="/images/posts/table_data_from_images/fig1.png" >}}
_On the left: [Canada Rent Rankings — May 2022](https://rentals.ca/blog/rentals-ca-may-2022-rent-report). Report summary by [Rentals.ca](https://rentals.ca/). On the right: Preprocessed image with cluster-defined table layout._

---
A crucial step in document parsing and recognition tasks, extracting table data from image and pdf files has been a widely explored problem with its own challenges. While working on a small personal project, I dived deep into it to discover a wide range of solutions with varying complexity. What, at first, seemed to be a simple task turned out to be an exciting learning opportunity.

Typical table parsing and recognition approaches use [R-CNNs](https://arxiv.org/pdf/1311.2524.pdf) (Such as [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet) and [RetinaNet](https://github.com/jabhinav/RetinaNet-for-Table-Detection)) that can leverage large public datasets such as [TableBank](https://github.com/doc-analysis/TableBank) or those made available during [ICDAR competitions](https://icdar2023.org/). Most successful frameworks often lead to a **precise table detection and layout recognition step**, followed by an **optical character recognition** process.

In this article, I document an **unsupervised implementation** of the problem, focusing on applying a simple yet robust layout matching step. Links to the references and the complete jupyter-notebook can be found in the latter sections.

Problem Definition
------------------

To analyze Canada’s historical rent prices, I gathered summary reports on Rentals.ca’s blog [6], published monthly by [Rentals.ca](https://rentals.ca/) since January 2019. The platform doesn’t allow web scraping, so the summary charts were manually downloaded from the publications. Each blog post is extensive and contains a comprehensive EDA on its’ monthly data, while the charts contain average rent prices per city, which is the core data we are looking for.

The goal is to compose a dataset to analyze trends and seasonality, and ultimately define optimum rent opportunity windows, but that is suitable for another article. For now, we need to extract the table data contained in each image.

Dataset Description
-------------------

The dataset contains **40 image files** with varying resolutions and table layouts, each holding information on monthly average rent prices for 1 and 2-bedroom units within different cities in Canada.

In this article, I will refer to some of the images used during development, and the goal is to implement a robust solution able to generalize on all cases with minimum hyperparameters tweaking.

Libraries and Dependencies
--------------------------

This project will make use of the latest available versions of **OpenCV** [2] and **PyTesseract** [3] at the moment of writing:

*   OpenCV 4.6.0
*   PyTesseract 0.3.10

[OpenCV](https://github.com/opencv/opencv) is a commonly used open source computer vision library [2] and will be used for image preprocessing. [PyTesseract](https://github.com/madmaze/pytesseract) is an open source Optical Character Recognition (OCR) engine for Python [3] and will be used to extract the text elements, bounding boxes, and confidences from our images.

Other visualization, statistics, machine learning, and data wrangling libraries should be available with a standard [Google Colab Notebook](https://colab.research.google.com/) instance.

Getting Started
---------------

Installing PyTesseract in your working environment. For installation from source or with conda, refer to their GitHub [installation instruction section](https://github.com/madmaze/pytesseract#installation).

```bash
!sudo apt install tesseract-ocr  
!pip install pytesseract
```

Importing libraries and dependencies.

```python
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
import pytesseract, cv2, os  
from glob import glob  
from tqdm import tqdm  
from statistics import mean, median, stdev  
from sklearn.cluster import AgglomerativeClustering

# Google colab cv2.imshow() alternative implementation  
from google.colab.patches import cv2_imshow

# Seaborn style settings
sns.set_theme(style = "ticks", palette = sns.dark_palette("seagreen", reverse=True))

# OpenCV and PyTesseract versions
print(f"OpenCV version: {cv2.__version__}")  
print(f"PyTesseract version: {pytesseract.__version__}")
```

Setting up data and output directories.

```python
# Input and output directories
os.mkdir('data/')  
os.mkdir('output/')  
os.mkdir('output/png/')  
os.mkdir('output/csv/')

# Reading images' paths from data directory
images = glob('data/*.png')
images
```

{{< figure src="/images/posts/table_data_from_images/fig2.png" title="" >}}
**Figure 2.** Monthly report images dataset.

The sample image used to illustrate this article is hosted on this [link](https://images.rentals.ca/images/Rent_Report_-_May_2022_.width-720.png), and the [complete dataset](https://github.com/erich-hs/Canada-Rents/tree/main/data/rentals.ca/monthly-reports) can be found on my GitHub repository. _The images were manually collected from the public blog posts at Rentals.ca/blog_ [6]_. I hold no proprietary rights to its content, and they are intended for personal use only._

Below is a preview of the original summary chart for the month of May 2022.

```python
# Displaying sample image  
cv2_imshow(cv2.imread(images[0]))
```

{{< figure src="/images/posts/table_data_from_images/fig3.png" title="" >}}
**Figure 3.** National Rent Rankings — May 2022, by [Rentals.ca](https://rentals.ca/) [4].

Image Preprocessing
-------------------

PyTesseract can handle RGB images, but we will introduce a preprocessing step using OpenCV to improve our OCR results. The following function will:

*   Resize the input image to a pre-defined width while preserving its original aspect ratio
*   Gray-scale the input image to a single color channel
*   Threshold the gray-scaled image with a binary threshold operation

The function also implements an optional blurring with a Gaussian Blur filter for experimentation.

For the purpose of this article, I will not dive deep into autoeach step described above, but the goal is to achieve a crisp black-and-white image from the source while standardizing image size to a higher resolution. It is a fundamental process to improve PyTesseract results, especially on lower-resolution images.

```python
def preprocess(image,  
               resize = False,  
               preserve_ar = True,  
               grayscale = False,  
               gaussian_blur = False,  
               thresholding = False,  
               thresh_value = 127,  
               verbose = True):
  '''  
  Preprocess image object input with:
  image: image input file path;  
  resize: Resize to desired width and height dimensions.
    Takes arguments tuple (width, height), single Integer as target width or false boolean.
    Will inforce aspect ratio based on passed target width if preserve_ar argument is set to True.
    Default = False.
    Default = True if resize argument is integer;  
  preserve_ar: Boolean argument to preserve original image's Aspect Ratio
    or redefine based on 'resize' input.
    Default = True;  
  grayscale: OpenCV grayscaling.
    Takes argument boolean = True or False. Default = False;  
  gaussian_blur: Smooth image input with a gaussian blurring method.
    Takes arguments Integer kernel size or false boolean.
    Default = False;  
  thresholding: OpenCV simple thresholding.
    Takes arguments [binary, binary_inv] or false boolean.
    Default = False;  
  thresh_value: OpenCV threshold value.
    Takes argument Int. Default = 127;
  '''
  # Image load and input dimensions
  input_file = image  
  image = cv2.imread(image)  
  input_height = int(image.shape[0])  
  input_width = int(image.shape[1])  
  aspect_ratio = input_height/input_width if verbose:  
    print(f"Processing input file: {input_file}...") 
  
  # Resizing  
  if type(resize) == int:  
    resize = (resize,) if resize:  
    if preserve_ar:  
      image = cv2.resize(image, (resize[0], int(resize[0]\*aspect_ratio)))  
    else:  
      image = cv2.resize(image, (resize[0], input_height)) output_height = int(image.shape[0])  
  output_width = int(image.shape[1]) 
  
  # Gray-scaling
  if grayscale:  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
  # Blurring
  if gaussian_blur:  
    image = cv2.GaussianBlur(image, (5, 5), gaussian_blur)
    
  # Thresholding
  if thresholding:  
    if thresholding == "binary":  
      image = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY_INV)[1]  
    elif thresholding == "binary_inv":  
      image = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY_INV)[1]  
    else:  
      print("Invalid thresholding argument!") if verbose:  
    print(f"Image input dimensions: {(input_width, input_height)}\n"\  
    f"Image output dimensions: {(output_width, output_height)}\n") return image
```

Below are the optimum preprocessing parameters for our use case and the corresponding resulting image.

```python
# Preprocessing parameters
preprocess_args = {  
      "resize": 1000,  
      "grayscale": True,  
      "thresholding": "binary",  
      "thresh_value": 165,  
      "verbose": False  
}

preprocessed_image = preprocess(images[0], **preprocess_args)  
cv2_imshow(preprocessed_image)
```

{{< figure src="/images/posts/table_data_from_images/fig4.png" title="" >}}
**Figure 4.** Preprocessed sample image.

OCR with PyTesseract
--------------------

We will use PyTesseract ‘image_to_data’ method [3] to extract the text elements and their corresponding bounding boxes and confidence levels. We can define the OCR settings according to what better suits our needs. We will look specifically at its page Segmentation Modes ( — psm).

```bash
!tesseract --help-psmPage segmentation modes:  
  0    Orientation and script detection (OSD) only.  
  1    Automatic page segmentation with OSD.  
  2    Automatic page segmentation, but no OSD, or OCR.  
  3    Fully automatic page segmentation, but no OSD. (Default)  
  4    Assume a single column of text of variable sizes.  
  5    Assume a single uniform block of vertically aligned text.  
  6    Assume a single uniform block of text.  
  7    Treat the image as a single text line.  
  8    Treat the image as a single word.  
  9    Treat the image as a single word in a circle.  
  10   Treat the image as a single character.  
  11   Sparse text. Find as much text as possible in no particular order.  
  12   Sparse text with OSD.  
  13   Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
```

After experimenting with the suitable options, psm 3, 4, and 11 are the most consistent for our use case. You can explore further available settings within the PyTesseract engine with the following help commands.

```bash
# Available OCR engine modes
!tesseract --help-oem

# Currently supported languages
!tesseract --list-langs

# List of all available parameters
!tesseract --print-parameters
```

Under all available parameters, you might find helpful arguments such as blacklisting and whitelisting characters, words to debug, and many other tweakable model settings.

For now, we will proceed by defining our desired language as English and setting our page segmentation mode to 4. We will also specify that we want our output as a Python dictionary.

```python
OCRdict = pytesseract.image_to_data(
                  images,
                  lang = 'eng',  
                  output_type = pytesseract.Output.DICT,  
                  config = "--psm 4"
  )
```

From our output we will use the parsed text elements and their corresponding bounding boxes and confidence levels. The following function will extract that information for each pair of occurrences to calculate vertical and horizontal gaps. We will also store only the observations with a positive confidence level (A negative confidence level of -1 means that the interpreted text is probably incorrect, where values from 0 to 100 quantify how confident the model is for the output prediction).

Another common debug step is disregarding all predictions where the OCR outputs a bounding box with height and width equal to the input image size. In these cases, it either tried to predict a single text element for the whole image or was unsure about given element’s dimensions.

```python
def draw_table(image,  
               pytesseract_config = "--psm 4",  
               conf_thresh = 0):  
  '''  
  Parsing image input with PyTesseract and extracting coordinates,
  bounding boxes, parsed text, and confidence levels.
  pytesseract_config: Pytesseract OCR config argument.
    String.  
    Default = "--psm 4";
  conf_thresh: Minimum confidence value for thresholding OCR results.
    Positive integer.
    Default = 0;  
  '''
  
  # Pytesseract image_to_data method on input image
  OCRdict = pytesseract.image_to_data(image,   
                    lang = 'eng',   
                    output_type = pytesseract.Output.DICT,   
                    config = pytesseract_config)

  # Initializing coords, gaps, and OCR text list
  coords = []  
  h_gaps = []  
  v_gaps = []  
  OCRtext = []  
  confs = []
  
  for i in range(0, len(OCRdict["text"])):  
    # Retrieving current text and bounding box coordinates
    x0 = OCRdict["left"][i]  
    y0 = OCRdict["top"][i]  
    w0 = OCRdict["width"][i]  
    h0 = OCRdict["height"][i]  
    text0 = OCRdict["text"][i]  
    conf0 = OCRdict["conf"][i]
    
    # Retrieving following text and bounding box coordinates
    try:  
      x1 = OCRdict["left"][i+1]  
      y1 = OCRdict["top"][i+1]  
      w1 = OCRdict["width"][i+1]  
      h1 = OCRdict["height"][i+1]  
    except:  
      pass
    
    # Calculating vertical and horizontal gaps to next element*
    h_gap = x1 - (x0 + w0)  
    v_gap = y1 - (y0 + h0)
    # Filtering out characters with confidence level below predefined threshold  
    # Filtering out undefined bounding boxes where OCR height and width are
    # higher than half input image height and width  
    if (conf0 > conf_thresh) and (h0 < image.shape[0]/2) and (w0 < image.shape[1]/2):  
      coords.append((x0, y0, w0, h0))  
      h_gaps.append(h_gap)  
      v_gaps.append(v_gap)  
      OCRtext.append(text0)  
      confs.append(conf0)
  
  table = {}  
  table['coords'] = coords  
  table['h_gaps'] = h_gaps  
  table['v_gaps'] = v_gaps  
  table['OCRtext'] = OCRtext  
  table['confs'] = confs  
    
  return table
```

Interpreting and visualizing our OCR output for the sample image.

```python
extracted_table = draw_table(preprocessed_image)

OCRdf = {  
    'Coordinates': extracted_table['coords'],  
    'h_gaps': extracted_table['h_gaps'],  
    'v_gaps': extracted_table['v_gaps'],  
    'Text': extracted_table['OCRtext'],  
    'Conf:': extracted_table['confs']  
}

pd.DataFrame(OCRdf)[25:45]
```

{{< figure src="/images/posts/table_data_from_images/fig5.png" title="" >}}
**Figure 5.** Partial OCR results for the sample image.

We can observe the partial alignment for texts that share the same row or columns. Negative vertical gaps show bounding boxes on the same table line, whereas positive ones indicate that the following text is probably starting a new line.

It is easier to visualize these alignments by plotting a histogram for both X and Y coordinates of our text elements.

```python
# Visualizing X Coordinates distribution
plt.figure(figsize = (8, 3))  
hist = sns.histplot(x = [x[0] for x in extracted_table['coords']],  
                    element = "step",  
                    binwidth = 25)  
hist.set_title("X Coordinate Distribution", size = 12, weight = "bold")  
hist.set(xlabel = "X Coordinate", ylabel = "Text Elements Count")  
plt.show()
```

{{< figure src="/images/posts/table_data_from_images/fig6.png" title="" >}}
**Figure 6.** X Coordinate histogram.

The plot hints at 8 distinct columns, which matches our expected result given the sample image.

```python
# Visualizing Y Coordinates distribution
plt.figure(figsize = (4, 10))  
hist = sns.histplot(y = [y[1] for y in extracted_table['coords']],  
                    element = "step",  
                    binwidth = 15)  
hist.invert_yaxis()  
hist.set_title("Y Coordinate Distribution", size = 12, weight = "bold")  
hist.set(xlabel = "Text Elements Count", ylabel = "Y Coordinate")  
plt.show()
```

{{< figure src="/images/posts/table_data_from_images/fig7.png" title="" >}}
**Figure 7.** Y Coordinate histogram.

We can also see our table lines here, with a somewhat cluttered first set of rows.

For a more intuitive interpretation of these plots, we can align them with the output image coordinates.

{{< figure src="/images/posts/table_data_from_images/fig8.png" title="" >}}
**Figure 8.** Preprocessed sample image with bounding boxes histograms.

Hierarchical Clustering
-----------------------

This section was inspired by an excellent [blog post](https://pyimagesearch.com/2022/02/28/multi-column-table-ocr/) [7] by Dr. @Adrian Rosebrock earlier this year on his blog [pyimagesearch.com](https://pyimagesearch.com/).

In his post, Adrian shows how we can use **hierarchical clustering [5] to define the table columns** based on their X coordinates [7]. To avoid misalignments (that happen way too often when table rows are not properly defined — either due to inaccurate bounding boxes or where the OCR model simply failed to interpret a text element), we will improve on his method by applying it to both X and Y coordinates.

As seen above, it is often easier to cluster columns rather than rows coordinates, as they hold a sparser and more evident alignment, with more bounding box instances on long format tables (where we have more _observations_ — rows than _features_ — columns). Therefore, we will use independent tweaks for X and Y coordinates clustering, while implementing an auxiliary statistical step to refine our rows selection and determine the marginal coordinates of our table in the image.

Before starting, we should first understand why Hierarchical Clustering and, more specifically, **Hierarchical Agglomerative Clustering** (HAC) [5] is the best option in our case.

Having our goal in mind of generalizing the results to distinct table layouts, we want a clustering algorithm that does not take a predefined number of clusters as input, such as most centroid and distribution-based algorithms. We must also keep in mind that **we have a univariate case, where each coordinate (X and Y) will be independently clustered**.

Agglomerative Clustering will do a “bottom-up” search for hierarchical structure in our data using a pre-defined distance metric. Each observation starts on its individual cluster to be paired up with its closest sample until a complete hierarchical tree is formed or a minimum distance threshold is reached.

We will use Scikit-Learn AgglomerativeClustering implementation. Getting started with the X coordinate to define columns, we set up our metric to use the manhattan distance (due to the univariate nature of our use case), a complete linkage method, and a distance threshold of 3.

SKLearn AgglomerativeClustering [1] also expects a feature space of at least two dimensions. So we will input 2D sets with our X coordinates and a dummy Y coordinate set to 0.

```python
# Clustering X coordinates
x_coords = [(x[0], 0) for x in extracted_table['coords']]

clustering = AgglomerativeClustering(n_clusters = None,  
                                     affinity = "manhattan",  
                                     linkage = "complete",  
                                     distance_threshold = 3)

clustering.fit(x_coords)
```

We can then start to define our table’s vertical lines bascoordinatesed on clustering results. For that, we iterate through each cluster, **average their X coordinates** and subtract a predefined horizontal padding (A 0 padding means that our lines will be tangent to each text’s bounding box). We will also filter out clusters with less than 10 observations.

```python
# Initializing vertical lines list and defining horizontal padding
v_lines = []  
h_padding = 10

# Iterating through X coordinates clusters
for cluster in np.unique(clustering.labels_):
  ids = np.where(clustering.labels_ == cluster)[0]
  
  # Filtering out outlier clusters with less than 10 observations
  if len(ids) > 10:  
    # Averaging X coordinates within clusters
    avg_x = np.average([extracted_table['coords'][i][0] for i in ids])  
    v_lines.append(int(avg_x) - h_padding)

# Sorting vertical lines on ascending order
v_lines.sort()  
n_columns = len(v_lines)
```

For the horizontal lines, we will cluster our Y coordinates with a distance threshold of 25.

```python
# Clustering Y coordinates
y_coords = [(0, y[1]) for y in extracted_table['coords']]

clustering = AgglomerativeClustering(n_clusters = None,  
                                     affinity = "manhattan",  
                                     linkage = "complete",  
                                     distance_threshold = 25)
                                     
clustering.fit(y_coords)
```

And proceed with a similar iteration through the resulting cluster **averaging the Y coordinates**. This time we will **filter out clusters smaller than half our number of columns rounded up**, meaning that we want only rows where at least half the columns contain text elements.

```python
# Initializing horizontal lines list and defining horizontal padding
h_lines = []  
v_padding = 10

# Iterating through Y coordinates clusters
for cluster in np.unique(clustering.labels_):  
  ids = np.where(clustering.labels_ == cluster)[0]
  
  # Filtering out clusters smaller than half n_columns rounded up
  if len(ids) > (int(n_columns / 2) + 1):  
    # Averaging Y coordinates within clusters
    avg_y = np.average([extracted_table['coords'][i][1] for i in ids])  
    h_lines.append(int(avg_y) - v_padding)
    
# Sorting horizontal lines on ascending order
h_lines.sort()
```

We can then start to visualize the resulting layout by plotting our cluster-defined vertical and horizontal lines.

```python
# Redefining preprocessed sample as a BGR OpenCV image
color_preprocessed = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

# Table lines color
lines_color = [76, 153, 0] # mild pale green

# Plotting resulting vertical lines
for v_line in v_lines:  
  cv2.line(color_preprocessed,  
           (v_line, 0),  
           (v_line, color_preprocessed.shape[0]),  
           color = lines_color,  
           thickness = 2)
           
# Plotting resulting horizontal lines
for h_line in h_lines:  
  cv2.line(color_preprocessed,  
           (0, h_line),  
           (color_preprocessed.shape[1], h_line),  
           color = lines_color,  
           thickness = 2)

cv2_imshow(color_preprocessed)
```

{{< figure src="/images/posts/table_data_from_images/fig9.png" title="" >}}
**Figure 9.** Preprocessed sample image with cluster-defined horizontal and vertical lines.

Vertical Gaps Smoothening
-------------------------

We can see that our clustering steps easily capture our table structure, but we still need to define where the table starts and finishes in the image.

As noted earlier, our column coordinates were easily clustered, but we must refine the resulting rows. We will do that by analyzing the distribution of the new gaps between lines. The following code section redefines our vertical gaps as the distance between each subsequent cluster-defined horizontal line and plots them into a density plot.

```python
# Vertical line gaps distribution
v_gaps = [h_lines[i+1] - h_lines[i] for i in range(len(h_lines) - 1)]

plt.figure(figsize = (7, 3))  
hist = sns.kdeplot(x = v_gaps, shade = True)  
hist.vlines(median(v_gaps), 0, 0.08, color = "green")  
hist.text(median(v_gaps) + 3, 0.075,  
          f"Median: {median(v_gaps)}",  
          color = "green")  
hist.set_title("Vertical gaps", size = 12, weight = "bold")  
hist.set(xlabel = "Vertical gaps")  
plt.show()
```

{{< figure src="/images/posts/table_data_from_images/fig10.png" title="" >}}
**Figure 10.** Vertical gaps density plot.

As in most images from our dataset, the vertical gaps are right-skewed with outliers on the higher end. We will work with a natural log of gaps to approximate it to a normal distribution (assuming that our skewed samples tend to a log-normal distribution). That transformation will allow us to use the new **distribution mean** and **standard deviation** values to create a **vertical gap range** to help define our table lines.

```python
# Log of vertical gaps to approximate to a normal distribution
log_v_gaps = np.log(v_gaps)plt.figure(figsize = (7, 3))  

hist = sns.kdeplot(x = log_v_gaps, shade = True)  
hist.set_title("log-Vertical gaps", size = 12, weight = "bold")  
hist.set(xlabel = "log of Vertical gaps")  
plt.show()
```

{{< figure src="/images/posts/table_data_from_images/fig11.png" title="" >}}
**Figure 11.** Log of Vertical gaps density plot.

We can now threshold our acceptable vertical gaps within a statistically defined range. To calculate this range, we will use the distribution mean and standard deviation and a formula that resembles the one used to determine confidence intervals on a standard normal distribution. However, instead of utilizing a Z critical value, we will implement our new hyperparameter: **smoothening vertical factor**.

The formula for our smoothening vertical increment is:

\begin{equation}
  S_{i} = S_{f} \times (\frac{\sigma}{\sqrt{n}})
\end{equation}
**Formula 1.** Smoothening Vertical Increment.

Where:

*   Si: Log of smoothening vertical increment
*   Sf: Smoothening vertical factor
*   σ: Log of vertical gaps’ standard deviation
*   n: Number of vertical gaps

The following segment calculates our smoothened vertical interval with a smoothening factor of 3 and converts it to a range object.

```python
# Vertical gaps smoothening factor
smooth_v_factor = 3

stdev_v_gaps = stdev(log_v_gaps)  
mean_v_gaps = mean(log_v_gaps)  
smooth_v_increment = smooth_v_factor * (stdev_v_gaps / np.sqrt(len(v_gaps)))  
smooth_v_interval = (mean_v_gaps - smooth_v_increment, mean_v_gaps + smooth_v_increment)

# Converting back to original scale
smooth_v_interval = np.exp(smooth_v_interval).astype("int8")

# Converting to a range interval
smooth_v_interval = range(smooth_v_interval[0], smooth_v_interval[1])
```

We can then plot our newly defined acceptable gap range over our vertical gap distribution.

```python
# Vertical line gaps distribution with smoothened interval
v_gaps = [h_lines[i+1] - h_lines[i] for i in range(len(h_lines) - 1)]

plt.figure(figsize = (7, 3))  
hist = sns.kdeplot(x = v_gaps, shade = False)  
hist.text(median(v_gaps) + 6, 0.06,  
          f"Smoothened gap:\n {smooth_v_interval}",  
          color = "green")  
kdeline = hist.lines[0]  
xs = kdeline.get_xdata()  
ys = kdeline.get_ydata()  
left = smooth_v_interval[0]  
right = smooth_v_interval[-1]  
hist.fill_between(xs, 0, ys,  
                  facecolor = 'darkgreen',  
                  alpha = 0.2)  
hist.fill_between(xs, 0, ys,  
                  where = (left <= xs) & (xs <= right),  
                  interpolate = True,  
                  facecolor = 'darkgreen',  
                  alpha = 0.4)  
hist.set_title("Vertical gaps", size = 12, weight = "bold")  
hist.set(xlabel = "Vertical gaps")  
plt.show()
```

{{< figure src="/images/posts/table_data_from_images/fig12.png" title="" >}}
**Figure 12.** Vertical gaps density plot with smoothened vertical interval.

To threshold the horizontal lines previously defined in our clustering step, we will iterate over them by doing a forward and backward search. The goal is to keep only the lines that define rows with gaps that fall within our smoothened vertical interval.

```python
# Updating horizontal lines based on smoothened vertical interval
smooth_h_lines = []

for i, line in enumerate(h_lines):  
  try:  
    # Look forward for at least 2 gaps in a row within smoothened interval  
    if h_lines[i+2] - h_lines[i+1] in smooth_v_interval:  
      if h_lines[i+1] - h_lines[i] in smooth_v_interval:  
        smooth_h_lines.append(line)  
    # Look backward for at least 2 gaps in a row within smoothened interval
      elif h_lines[i-1] - h_lines[i-2] in smooth_v_interval:  
        if h_lines[i] - h_lines[i-1] in smooth_v_interval:  
          smooth_h_lines.append(line)  
  except:  
    pass
```

With that, we should have well-defined columns and rows. To enclose them in our table, we are missing a final step: Calculating the last row height and the last column width.

For that, we will set the last row height as the mean value of the preceding rows minus the average height of text elements within the table. For the last column width, we will look for its widest text element and add some horizontal padding to it.

```python
# Defining external borders  
# Calculating mean vertical spacing within table
v_spacings = []

for i in range(0, len(smooth_h_lines) - 1):  
  v_spacings.append(smooth_h_lines[i+1] - smooth_h_lines[i])
  
# Subtracting mean
v_spacing = int(mean(v_spacings) - mean([h[3] for h in extracted_table['coords']]))

# Coordinates on last column
last_column_widths = []  
for id in np.where([x[0] for x in extracted_table['coords']] > np.max(v_lines))[0]:  
  last_column_widths.append(extracted_table['coords'][id][2])
  last_column_width = np.max(last_column_widths) + 2*h_padding
```

We can now draw our resulting table layout.

```python
# Drawing final table layout
color_preprocessed = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

border_color = [51, 102, 0] # dark green
lines_color = [76, 153, 0] # mild pale green

# Table corners
x_min = np.min(v_lines)  
y_min = np.min(smooth_h_lines)  
x_max = np.max(v_lines) + last_column_width  
y_max = np.max(smooth_h_lines) + v_spacing + v_padding

# Drawing external borders
cv2.rectangle(color_preprocessed,  
              (x_min, y_min),  
              (x_max, y_max),  
              color = border_color,  
              thickness = 3)
              
# Table lines and columns
for v_line in v_lines:  
  if v_line != x_min:  
    cv2.line(color_preprocessed,  
             (v_line, y_min),  
             (v_line, y_max),  
             color = lines_color,  
             thickness = 2)

for h_line in smooth_h_lines:  
  if h_line != y_min:  
    cv2.line(color_preprocessed,  
             (x_min, h_line),  
             (x_max, h_line),  
             color = lines_color,  
             thickness = 2)

cv2_imshow(color_preprocessed)
```

{{< figure src="/images/posts/table_data_from_images/fig13.png" title="" >}}
**Figure 13.** Preprocessed sample image with final table layout.

We now have concluded the most critical step to a robust multi-column table OCR. Layout matching. You can experiment with the other images in the dataset, where most of them will output similarly accurate results using the same hyperparameters. You can also feed other table images (appropriately preprocessed) and play with the parameters we have defined:

*   Vertical cluster size threshold
*   Horizontal/vertical clustering distance thresholds
*   Smoothening vertical factor
*   Horizontal/vertical padding

Let’s proceed to the final step.

Writing a DataFrame
-------------------

To capture our text content into a Pandas DataFrame, we will initialize an empty NumPy array with our corresponding table dimensions.

You will notice that some table cells contain more than one text element detected by PyTesseract. We will define columns and row ranges to determine when to concatenate these elements into a single string and allocate them to their appropriate table cell.

```python
# Columns ranges
columns = []  
for i in range(0, len(v_lines) -1):  
  columns.append(range(v_lines[i], v_lines[i+1]))
  
# Appending last column
columns.append(range(v_lines[-1], x_max))

# Rows ranges
rows = []  
for i in range(0, len(smooth_h_lines) -1):  
  rows.append(range(smooth_h_lines[i], smooth_h_lines[i+1]))
  
# Appending last row
rows.append(range(smooth_h_lines[-1], y_max))
```

We can now map our OCR texts to our NumPy array. We will use the centroid of each bounding box to avoid losing text elements that might clip through the table lines. The following code initializes our NumPy array, populates it and converts it to a pandas DataFrame.

```python
# Final table dimensions
n_rows = len(smooth_h_lines)  
table_dim = (n_rows, n_columns)

# Initializing an empy NumPy array to store table data
table = np.empty(table_dim, dtype = 'object')

for coord, text in zip(extracted_table['coords'], extracted_table['OCRtext']):  
  for j in range(0, len(columns)):  
    for i in range(0, len(rows)):  
      if (int(coord[0] + coord[2]/2) in columns[j]) and (int(coord[1] + coord[3]/2) in rows[i]):  
        # Check for cells with existing text element and concatenate
        if table[i, j] is not None:  
          table[i, j] += f" {text}"  
        # Or populate it with a new element
        else:  
          table[i, j] = textpd.DataFrame(table[1:], columns = table[0])
```

{{< figure src="/images/posts/table_data_from_images/fig14.png" title="" >}}
**Figure 14.** Final DataFrame.

And here is our table. We can see that our current configuration for PyTesseract struggled a bit on percentages decimal cases at the Year over Year and Month over Month columns. Nevertheless, we should not worry about refining results here since they are calculated columns, and we can always derive them from our core data; the average rent prices.

Our output is now a cleaning step away from being ready for exploratory analysis and machine learning. We can strip away inconsistent spacings, dollar signs, and thousands separators using regular expressions. City names and provinces misspellings can also be easily revised, especially after aggregating the result of our complete dataset.

So let’s process all the images. The following code uses an updated version of the draw_table function that includes all our steps so far and outputs a dictionary with the extracted DataFrame and the corresponding preprocessed image with the predicted table layout. We will then export our results as csv files and png images.

```python
# Preprocessing parameters
preprocess_args = {  
        "resize": 1000,  
        "grayscale": True,  
        "thresholding": "binary",  
        "thresh_value": 165  
}

# OCR table extraction parameters
draw_table_args = {  
    "pytesseract_config": "--psm 4",  
    "h_padding": 10,  
    "v_padding": 10,  
    "h_distance_threshold": 3,  
    "v_distance_threshold": 25,  
    "smooth_v_factor": 3  
}

# Extracting table for images on input folder
tables = {}for image in tqdm(images):  
  match = re.search("[A-Z][a-z]\*.[0-9][0-9][0-9][0-9]", image)  
  if match is not None:  
    table_name = match.group(0).replace("-", "_")  
    preprocessed = preprocess(image, **preprocess_args, verbose = False)  
    tables[table_name] = draw_table(preprocessed, **draw_table_args)

# Writting png and csv files
for table in tables:  
  cv2.imwrite(f"output/png/{str(table)}.png", tables[table]['image'])  
  tables[table]['df'].to_csv(f"output/csv/{str(table)}.csv")

# Creating output zip folder
shutil.make_archive("output", "zip", 'output/')
```

{{< figure src="/images/posts/table_data_from_images/fig15.png" title="" >}}
**Figure 15.** Processed dataset samples.

Most of the 40 images from our dataset returned a perfectly accurate match for the table layout, while a minority defined an extra column or failed to close the table boundaries. We can always revisit these individuals with updated parameters.

And that puts a check on our table parsing from images with cluster-defined layouts. You can find the development **jupyter notebook** on the [project repository](https://github.com/erich-hs/Canada-Rents) or directly through this [link](https://github.com/erich-hs/Canada-Rents/blob/main/rent_scraping_development.ipynb). I hope you learned as much as I did while going through this article.

References
----------

[1] _2.3. clustering_. Scikit-Learn. (n.d.). Retrieved September 15, 2022, from [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)

[2] _About_. OpenCV. (2020, November 4). Retrieved September 15, 2022, from [https://opencv.org/about/](https://opencv.org/about/)

[3] Madmaze. (n.d.). _Madmaze/pytesseract: A Python wrapper for google tesseract_. GitHub. Retrieved September 15, 2022, from [https://github.com/madmaze/pytesseract](https://github.com/madmaze/pytesseract)

[4] Myers, B. (2022, June 13). _Rentals.ca May 2022 Rent Report_. [Rentals.ca](https://rentals.ca/) Blog. Retrieved September 15, 2022, from [https://rentals.ca/blog/rentals-ca-may-2022-rent-report](https://rentals.ca/blog/rentals-ca-may-2022-rent-report)

[5] Nielsen, Frank (2016). [“8. Hierarchical Clustering”](https://www.researchgate.net/publication/314700681). [_Introduction to HPC with MPI for Data Science_](https://www.springer.com/gp/book/9783319219028). Springer. pp. 195–211. ISBN 978–3–319–21903–5.

[6] [Rentals.ca](https://rentals.ca/) Blog. (n.d.). Retrieved September 15, 2022, from [https://rentals.ca/blog/](https://rentals.ca/blog/)

[7] Rosebrock, A. (2022, February 24). _Multi-column table OCR_. PyImageSearch. Retrieved September 15, 2022, from [https://pyimagesearch.com/2022/02/28/multi-column-table-ocr/](https://pyimagesearch.com/2022/02/28/multi-column-table-ocr/)
