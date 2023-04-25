---
title: "Comprehensive Time Series Exploratory Analysis"
subtitle: "A deep dive into air quality data"
categories: ['Data Analytics', 'Data Science']
tags: ['Time Series', 'EDA', 'Python', 'Visualization']
date: 2023-01-24T16:14:33-07:00
draft: false

featuredImagePreview: "/images/posts/time_series_eda/fig0.jpg"
---
{{< figure src="/images/posts/time_series_eda/fig0.jpg" >}}

_Photo by [Jason Blackeye](https://unsplash.com/@jeisblack?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/collections/55366/my-first-collection/981603704225affe48a9007fc5094d84?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)_

---
Here you are with a dataset indexed by time stamps. Your data might be about storage demand and supply, and you are tasked with predicting the ideal replenishment intervals for a strategic product. Or maybe you need to translate historical sales information into key actionable insights for your team. Perhaps your data is financial, with information about historical interest rates and a selection of stock prices. Maybe you are tasked with modelling market volatility and need to quantify monetary risk over an investment horizon. From social sciences and energy distribution or from healthcare to environmental studies. The examples are numerous. But what do these scenarios have in common? One, you have a time series task at hand. And two, you will certainly benefit from starting with a **succinct yet comprehensive exploratory analysis**.

---
## The Goal of this Article
But what does it mean to perform an **exploratory time series analysis**? Different from other data science problems, gathering insights from time series data can be tricky and everything but straightforward. Your data might have important underlying trends and seasons or be suitable for nested forecasting within its intricate cyclical patterns. Differentiating abnormal outliers caused by a failure in your data generation process from actual anomalies that hold key information can be challenging. And dealing with missing values might not be as simple as you expect.

This article will outline a process that has worked for me when studying time series datasets. You will follow me along as I explore the measurements of fine particulate matter, also known as PM 2.5, one of the main contributors to air pollution and air quality indices. I will focus on laying out some best practices with specific attention to detail to generate sharp and highly informative visualizations and statistical summaries.

---
## Dataset Description
The data studied here are from four monitoring stations in the city of Vancouver, British Columbia, Canada. They hold one-hour average measurements of fine particulate matter PM 2.5 (fine particles with diameters of 2.5 microns and smaller) in µg/m3 (micrograms per cubic meter) with values from January 1st, 2016, to July 3rd, 2022.

PM 2.5 primarily comes from the burn of fossil fuels, and in cities, it normally originates from car traffic and construction sites. Another major source of the pollutant are forest and grass fires, and they are easily carried away by the wind [1].

The image below shows the approximate location of the stations we will explore.

{{< figure src="/images/posts/time_series_eda/fig1.png" title="Vancouver map with air monitoring stations.">}}

The dataset is from the [British Columbia Data Catalogue](https://catalogue.data.gov.bc.ca/), and as stated by the publisher, it has not been quality-assured [5]. For the version you will see here, I have preprocessed some minor problems, such as assigning negative measurements (only 6 out of 57k observations) as missing values and generating a master DataFrame with the stations of our choice.

---
## Libraries and Dependencies
We will use Python 3.9 and the plotting libraries [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for our visualizations. For statistical tests and data exploration, we will work with the [statsmodels](https://www.statsmodels.org/stable/index.html) Python module and the [SciPy](https://scipy.org/) library. All our data manipulation and auxiliary tasks will be handled with [Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/).

These packages are natively available in popular Python distributions and hosted notebooks, such as Anaconda and Miniconda, Google Collab or Kaggle Notebooks. So every code example here should be easily reproducible in your environment of choice.

---
## Getting Started
Starting by importing our libraries, we will call ```matplotlib.dates.mdates``` , and the ```datetime``` module to help us work with our DateTime index. To generate consistent visualizations, I also like to start by defining our plot styles and the color palette. So let's begin with.

{{< gist erich-hs 478ad553dd9d174c8d1b5557df725fb7 >}}

{{< figure src="/images/posts/time_series_eda/fig2.png" title="Seaborn mako color palette.">}}

After reading the .csv file, we will define the timestamp ```DATE_PS``` as a NumPy ```datetime64``` object and set it as our DataFrame index. This common step will enable some Pandas time series functionalities, such as the one used below to create datepart features in our dataset.

{{< gist erich-hs 72bd8781d9eaa553ef49d47e09433414 >}}

| DATE_PST | North_Vancouver_Second_Narrows_PM25 | Month | Day_of_Week | Hour |
|--:|--:|--:|--:|--:|
| **2016-01-01 01:00:00** | NaN | 1 | Friday | 1 |
| **2016-01-01 02:00:00** | 6.341111 | 1 | Friday | 2 |
| **2016-01-01 03:00:00** | 5.340278 | 1 | Friday | 3 |
| **2016-01-01 04:00:00** | 3.907917 | 1 | Friday | 4 |
| **2016-01-01 05:00:00** | 7.235889 | 1 | Friday | 5 |

---
## The Big Picture
Here is an ample view of where we will dive in - This is where we spend some time to get a first grasp of our data.

For this visualization, we will use Seaborn relational plots that will read from an aggregated long version of our DataFrame. For that, we will use Pandas ```melt``` and ```resample``` methods with a mean aggregation in 24 hours intervals. This will reduce our data granularity from hourly to daily average measurements, and it is done to reduce the time it takes to generate the plot.

{{< gist erich-hs 431c2a4279e8c64d2d94825bf3eb51ff >}}

{{< figure src="/images/posts/time_series_eda/fig4.png" title="PM 2.5 time series plot of monitoring stations.">}}

With a clear picture of all four stations along the entire span of our time series, it is already possible to start taking some notes:

* There are some major anomalies, and they seem to be prevalent during summer and early autumn.
* These anomalies appear to result from large-scale events, as they affect all four stations approximately during the same periods.
* If you look carefully, **I have included a faded gray scatter plot of all stations within every chart**. With this subtle detail, it is possible to see that the anomaly present in 2017, for example, had a major effect in both North Vancouver stations (they reached higher PM 2.5 values), while the opposite is true for the 2018 event. This technique also ensures that all four line charts are within the same Y-axis range.

Some good practices that you can take from this first plot:
* Matplotlib allows for highly customizable axis ticker locators and formatters. In this example, I have used a ```MonthLocator``` from the ```mdates``` module to create **minor month locators** in the X-axis, which help with overall readability.
* The exact expected dates range from our plot is passed to the plot title or subtitle. "Expected" because our visualization might be truncated due to missing values on either end of the plotted period, and this practice can help identify those problems. It is also a good documentation practice to report your findings.

Great start, but let's narrow down our view slightly. In the next section, we will  look at shorter periods of time, but now with our original, hourly granularity.

---
## A Detailed View
From now on, we will start defining some functions we can quickly call to generate tailored visualizations. You can think of it as a means of setting up an analytical toolset that will be extremely helpful moving forward.

This first function will help us look at individual time series within a specific period in time. We will start by looking at the year 2017 for the North Vancouver Mahon Park Station.

{{< gist erich-hs d11a61723421e8009844db811a85bdcb >}}

{{< figure src="/images/posts/time_series_eda/fig5.png" title="North Vancouver Mahon Park PM 2.5 plot for 2017.">}}

We can see here that there are localized smaller spikes in addition to the major anomalies. There were also periods of higher volatility (where variance increases during a short time span) at the beginning of the year and in December 2017.

Let's dive in a bit further and look outside the anomaly range so we can analyze our values within a narrower span in the Y-axis.

{{< gist erich-hs 78222f4a8e670f4fc053f5f00399d24c >}}

{{< figure src="/images/posts/time_series_eda/fig6.png" title="North Vancouver Mahon Park PM 2.5 plot for 15 Apr to 1 Jul 2017.">}}

We can see some missing values here. The ```fill=True``` parameter of our function helps identify that and is a good way to give visual emphasis to missingness in our data. Those small interruptions that are initially hard to notice are now clearly visible.

Another detail you might have noticed is the X-axis's custom date format. For the plot above, I have enhanced our ```plot_sequence()``` function with custom major and minor locators and formatters. This new functionality now adapts our plot to the visualization's time span and formats the X-axis accordingly. Below is the code snipped that was included in the function.

{{< gist erich-hs 6a19dd3d7bf09c57b81c07310d35f54a >}}

Now we know that our dataset has interruptions, so let's take a better look at that.

---
## Missing Values

For tabular data problems, in this section, we would probably be focused on defining MAR (Missing At Random) from MNAR (Missing Not At Random). Instead, knowing the nature of our data (sensorial temporal measurements), we know that interruptions in the data stream are probably not intended. Hence, in these cases, it is more important to distinguish isolated from continuously missing values and missing values from completely missing samples. The possibilities here are vast, so I will dedicate an article solely to that aspect in the future.

For now, let's start by looking at a **missing values heatmap**. Again we will define a function for that.

{{< gist erich-hs 6c7339b446321022f05ca88aaef0376d >}}

{{< figure src="/images/posts/time_series_eda/fig7.png" title="Missing values heatmap.">}}

Heatmaps are great as they allow us to quantify missingness and **localize them in the time axis**. From here, we can denote:
* We don't have completely missing samples (where missing values occur simultaneously for a time period). That's expected as the data streaming from the monitoring stations happens independently.
* There are long sequences of missing values early on in our timeline, and data availability seems to improve as time goes by.

Some statistical analysis in the latter sections will be problematic with missing values. Therefore we will use simple techniques to treat them as we see fit, such as:
* Pandas ```ffill()``` and ```bfill()``` methods. They are used to carry forward or backwards, respectively, the nearest available value.
* Linear or spline interpolation with Pandas ```interpolate()``` method. It uses neighbouring observations to draw a curve to fill in a missing interval.

---
## Intermittency
From the nature of our data, we should not expect negative values. As mentioned in the beginning, I treated them as missing when preprocessing the data. Let's call our summary statistics to confirm that.

{{< gist erich-hs 79bdd03e0ae8aa7aa6e0203be8074bfc >}}

|   | count | mean | std | min | 25% | 50% | 75% | max |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Vancouver_Clark_Drive_PM25**              | 55487.0 | 7.45 | 9.60 | 0.0 | 3.48 | 5.56 | 8.55 | 214.87 |
| **Vancouver_International_Airport_#2_PM25** | 56467.0 | 5.57 | 8.33 | 0.0 | 2.30 | 3.94 | 6.48 | 209.59 |
| **North_Vancouver_Mahon_Park_PM25**         | 56381.0 | 5.11 | 8.98 | 0.0 | 2.04 | 3.51 | 5.67 | 203.03 |
| **North_Vancouver_Second_Narrows_PM25**     | 54763.0 | 6.68 | 8.64 | 0.0 | 3.12 | 4.95 | 7.69 | 188.93 |

We see that our minimum measurements are zeros for each station, which leads us to the next question. **Is our time series intermittent?**

Intermittency is characterized when your data has a large number of values that are exactly zero. This behaviour poses specific challenges and must be taken into account during model selection. So how often do we see zero values in our data?

{{< gist erich-hs 701de5c6af39c8c0da4eebce918bc2d0 >}}

|   | Zeroes | Zeroes % |
|--:|--:|--:|
| **Vancouver_Clark_Drive_PM25**              |  19 | 0.03 |
| **Vancouver_International_Airport_#2_PM25** | 136 | 0.24 |
| **North_Vancouver_Mahon_Park_PM25**         | 408 | 0.72 |
| **North_Vancouver_Second_Narrows_PM25**     |  10 | 0.02 |

We can see that the amount of zeros is negligible, so we don't have  intermittent series. This is an easy but crucial check, especially if your goal is forecasting. It might be hard for some models to predict an absolute zero, and that can be a problem if you want to forecast demand, for example. You don't want to plan out the delivery of, let's say, three products to your client if, in fact, he is expecting none.

---
## Seasonality

Understanding the cycles in your time series is fundamental to planning the modeling phase. You might be losing key information of smaller cycles if you decide to aggregate your data too much, or it can help you determine the feasibility of forecasting on a smaller granularity.

We will use some box plots to start looking into that. But first, we will temporarily remove the top 5% percentile so we can look at the data on a better scale.

{{< gist erich-hs 64d35a40e70fb116591a828e3f1744f7 >}}

In this next function, we will use a series of boxplots to investigate the cycles within our data. We will also map our color palette to the median values so it can serve as another neat visual clue.

{{< gist erich-hs 16ed19bd9050a8ee0b4e38ffd8a21f51 >}}

{{< figure src="/images/posts/time_series_eda/fig10.png" title="PM 2.5 hourly values.">}}

This first plot returns hourly measurements. Here we can see:
* Consistently higher values for PM 2.5 from 9 AM to 2 PM.
* Stations outside North Vancouver also show a peak from 8 PM to 10 PM.
* Early mornings hold the lowest values for PM 2.5 from 2 AM to 5 AM.

Now looking at weekly seasonality and the difference in values across the week.

{{< gist erich-hs 4a213524f43a998e235f05c7507954ef >}}

{{< figure src="/images/posts/time_series_eda/fig11.png" title="PM 2.5 daily values.">}}

From here, we see:
* Lower PM 2.5 values during weekends.
* A slightly higher trend for pollution levels on Tuesdays.

And finally, looking at the month-to-month trend.

{{< gist erich-hs 7a40aabffa1d407b3b304d602369bcfd >}}

{{< figure src="/images/posts/time_series_eda/fig12.png" title="PM 2.5 monthly values.">}}

Where we can observe:
* Consistently higher values for PM 2.5 in August for all years.
* The southern stations have lower PM 2.5 values in June and July, while the North Vancouver ones show lower measurements in January.

Finally, more good practices from these plots:
* Do not make naive use of your color palette, as they might mislead you to equivocate interpretations. Had we simply passed pallette="mako" to our boxplots, it would have been mapped to our X-axis and not to our variable of interest.
* Grid plots are powerful containers of information for low-dimensional data, and they can be quickly set up with Seaborn relplot() or Matplotlib subplots().
* You can make use of the Seaborn boxplot() order parameter to reorder your X-axis accordingly. I used it to reorganize the day-of-week X labels in a meaningful order.

A more elaborate view of seasonalities can be attained from a trend-season decomposition from our time series. This, however, will be left for a future article where we can dive deeper into time series similarity and model selection.

For now, let's try a quick look at one of our well-known statistical coefficients to investigate the **linear relationship** between our four stations.

---
## Pearson Correlation
R programmers might be familiar with the following plot. A correlogram is a concise and highly informative visualization that is implemented at multiple R libraries, such as the ```ggpairs()``` in the ```GGally``` package. The upper diagonal of a correlogram shows us bivariate correlations, or Pearson correlation coefficients between numeric variables in our data. In the lower diagonal, we see scatterplots with regression curves fitted to our data. Finally, in the main diagonal, we have histograms and a density curve of each variable.

The following code is an adapted implementation using the Seaborn ```PairGrid()``` plot and yet another function for our analytical toolset.

{{< gist erich-hs 16eab8b5a8f0735f0c3d54795773b101 >}}

{{< figure src="/images/posts/time_series_eda/fig13.png" title="PM 2.5 Correlogram on all four stations.">}}

As expected, our stations are highly correlated, especially the ones closer to each other, such as both in North Vancouver. 

It is important to note that to alleviate the computational time, our data was aggregated in 6-hours periods. If you experiment with this plot with bigger aggregation periods, you will see an increase in the correlation coefficients, as mean aggregations tend to smoothen out the outliers present in the data.

If you were already introduced to time series analysis, you might now be thinking about other kinds of correlation that are worth checking. But first, we need to test our time series for **stationarity**.

---
## Stationarity
A stationary time series is one whose statistical properties do not change over time. In other words, it has a constant mean, variance, and autocorrelation independent of time [4].

**Several forecasting models rely on time series stationarity**, hence the importance of testing for it in this exploratory phase. Our next function will make use of statsmodels implementation of two commonly used tests for stationarity, the _Augmented Dickey-Fuller_ ("ADF") test and the _Kwiatkowski-Phillips-Schmidt-Shin_ ("KPSS") test.

I will leave both tests' hypotheses below. Note that they have opposing null hypotheses, so we will create a "Decision" column for an easy interpretation of their results. You can read more about both tests in the [statsmodels documentation](https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html).

***Augmented Dickey-Fuller (ADF)*** test hypothesis:
>* **H0**: A unit root is present in the time series sample (**Non-stationary**)
>* **Ha**: There is no root unit present in the time series sample (**Stationary**) 

***Kwiatkowski-Phillips-Schmidt-Shin (KPSS)*** test hypothesis:
>* **H0**: The data is stationary around a constant (**Stationary**)
>* **Ha**: A unit root is present in the time series sample (**Non-stationary**)

Now an opportune question to ask is on which scale we should check for stationarity. The answer will highly depend on how you will model your data, and one of the goals of a comprehensive exploratory analysis is exactly to help you with that decision.

For illustration purposes, in the following example, we will take a look at the months of January 2016 and January 2022 for the Vancouver International Airport Station and see if there was a change in behavior from 2016 to 2022 in the data.

{{< gist erich-hs d9f9d164964c337a238b95154d64ec54 >}}

You might remember from our Missing Values section that we can use Pandas ```ffill()```, ```bfill()```, and ```interpolate()``` methods to quickly impute interruptions in the series. You can see that I have defined a dedicated argument ```fillna``` to our function to select from either of these methods to quickly work around missing values, as both tests only accept complete samples.

Now coming back to our results.

|   | adf | kpss |
|--:|--:|--:|
| **Test Statistic** | -2.7768        | 3.079          |
| **p-value**        | 0.0617         | 0.01           |
| **Lags Used**      | 20             | 15             |
| **Decision**       | Non-Stationary | Non-Stationary |

{{< gist erich-hs c09ecfc47d1536d962fd995df50ac6e8 >}}

|   | adf | kpss |
|--:|--:|--:|
| **Test Statistic** | -3.2505        | 0.4676         |
| **p-value**        | 0.0172         | 0.0049         |
| **Lags Used**      | 20             | 15             |
| **Decision**       | Stationary     | Non-Stationary |

We can see that for 2016 both tests indicated Non-Stationarity, but for 2022 the results diverged. The statsmodels documentation clearly lists the interpretations for the results when the ADF and KPSS tests are performed together [6]:

>* **Case 1**: **Both tests conclude that the series is not stationary** - The series is not stationary
>* **Case 2**: **Both tests conclude that the series is stationary** - The series is stationary
>* **Case 3**: **KPSS indicates stationarity** and **ADF indicates non-stationarity** - The series is trend stationary. **Trend needs to be removed** to make series strict stationary. The detrended series is checked for stationarity.
>* **Case 4**: **KPSS indicates non-stationarity** and **ADF indicates stationarity** - The series is difference stationary. **Differencing is to be used** to make series stationary. The differenced series is checked for stationarity.

If you repeat this operation for all four stations across multiple months, you will see that Case 4 is predominant in the data. This leads us to our next section about **first-order differencing to make our data stationary**.

---
## First-Order Differencing
As one of the most common transformation techniques, applying first- or second-order differencing to a time series is widely used to make the data suitable for statistical models that can only be used on stationary time series. Here we will look at the technique applied to one of the previous examples in the month of January 2016. But first, let's take a look at the original data before the transformation with our ```plot_sequence()``` function.

{{< gist erich-hs edba50f969c547c4cdfdbd36b9f26b0d >}}

{{< figure src="/images/posts/time_series_eda/fig16.png" title="Vancouver International Airport PM 2.5 plot for Jan 2016.">}}

We can see that the variance in the period changes significantly from the beginning to the end of the month. The mean PM 2.5 also seems to go from a higher to a lower, more stable value. These are some of the characteristics that confirm the non-stationarity of the series.

Again, Pandas has a quite convenient method to differentiate our data. We will call ```.diff()``` to our DataFrame and instantiate it as a first-order differentiated version of our data. So let's plot the same period again.

{{< gist erich-hs 34ab66a930c195de6b26e5de5c2a7fa9 >}}

{{< figure src="/images/posts/time_series_eda/fig17.png" title="Vancouver International Airport differentiated PM 2.5 plot for Jan 2016.">}}

Besides the still oscillating variance, the data is now clearly more stable around a mean value. We can once again call our stationarity_test() function to check for stationarity on the differentiated data.

{{< gist erich-hs a4f6a2a4e434310998b35f68a46e007b >}}

|   | adf | kpss |
|--:|--:|--:|
| **Test Statistic** | -12.153        | 0.1035         |
| **p-value**        | 0.0            | 0.1            |
| **Lags Used**      | 20             | 24             |
| **Decision**       | Stationary     | Stationary     |

There we have it. We can put another check on our comprehensive exploratory time series analysis, as we have now confirmed that:
* We are dealing with non-stationary time series.
* First-order differencing is an appropriate transformation technique to make it stationary.

And that finally leads us to our last section.

---
## Autocorrelation
Once our data is stationary, we can investigate other key time series attributes: partial autocorrelation and autocorrelation. In formal terms:

>The **autocorrelation function (ACF)** measures the linear relationship between lagged values of a time series. In other words, it measures the correlation of the time series with itself. [2]

>The **partial autocorrelation function (PACF)** measures the correlation between lagged values in a time series when we remove the influence of correlated lagged values in between. Those are known as confounding variables. [3]

Both metrics can be visualized with statistical plots known as correlograms. But first, it is important to develop a better understanding of them.

Since this article is focused on exploratory analysis and these concepts are fundamental to statistical forecasting models, I will keep the explanation brief, but bear in mind that these are highly important ideas to build a solid intuition upon when working with time series. For a comprehensive read, I recommend the great kernel [Time Series: Interpreting ACF and PACF](https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf) by the Kaggle Notebooks Grandmaster Leonie Monigatti.

As noted above, autocorrelation measures how the time series correlates with itself on previous _q_ lags. You can think of it as a measurement of the linear relationship of a subset of your data with a copy of itself shifted back by _q_ periods. **Autocorrelation, or ACF, is an important metric to determine the order q of Moving Average (MA) models**.

On the other hand, partial autocorrelation is the correlation of the time series with its _p_ lagged version, but now solely regarding its **direct effects**. For example, if I want to check the partial autocorrelation of the _t-3_ to _t-1_ time period with my current _t0_ value, I won't care about how _t-3_ influences _t-2_ and _t-1_ or how _t-2_ influences _t-1_. I'll be exclusively focused on the direct effects of _t-3_, _t-2_, and _t-1_ on my current time stamp, _t0_. **Partial autocorrelation, or PACF, is an important metric to determine the order _p_ of Autoregressive (AR) models**.

With these concepts cleared out, we can now come back to our data. Since the two metrics are often analyzed together, our last function will combine the PACF and ACF plots in a grid plot that will return correlograms for multiple variables. It will make use of statsmodels ```plot_pacf()``` and ```plot_acf()``` functions, and map them to a Matplotlib ```subplots()``` grid.

{{< gist erich-hs 8fe341623adb0b38d8ebc979cf2610a2 >}}

Notice how both statsmodels functions use the same arguments, except for the ```method``` parameter that is exclusive to the ```plot_pacf()``` plot.

Now you can experiment with different aggregations of your data, but remember that when resampling the time series, each lag will then represent a different jump back in time. For illustrative purposes, let's analyze the PACF and ACF for all four stations in the month of January 2016, with a 6-hours aggregated dataset.

{{< gist erich-hs fbc4d04c97e4380cfc501309057a496e >}}

{{< figure src="/images/posts/time_series_eda/fig19.png" title="PACF and ACF Correlograms for Jan 2016.">}}

Correlograms return the correlation coefficients ranging from -1.0 to 1.0 and a shaded area indicating the significance threshold. Any value that extends beyond that should be considered statistically significant.

From the results above, we can finally conclude that on a 6-hours aggregation:

* Lags 1, 2, 3 (t-6h, t-12h, and t-18h) and sometimes 4 (t-24h) have significant PACF.
* Lags 1 and 4 (t-6h and t-24h) show significant ACF for most cases.

And take note of some final good practices:
* Plotting correlograms for large periods of time series with high granularity (For example, plotting a whole-year correlogram for a dataset with hourly measurements) should be avoided, as the significance threshold narrows down to zero with increasingly higher sample sizes.
* I defined an ```x_label``` parameter to our function to make it easy to annotate the X-axis with the time period represented by each lag. It is common to see correlograms without that information, but having easy access to it can avoid misinterpretations of the results.
* Statsmodels ```plot_acf()``` and ```plot_pacf()``` default values are set to include the 0-lag correlation coefficient in the plot. Since the correlation of a number with itself is always one, I have set our plots to start from the first lag with the parameter ```zero=False```. It also improves the scale of the Y-axis, making the lags we actually need to analyze more readable.

---

With that, we have thoroughly explored our time series. With a toolset of visualizations and analytical functions, we could draw a comprehensive understanding of our data. You have also learned some of the best practices when exploring time series datasets and how to present them succinctly and polishedly with high-quality plots.

---
## References
[1] "Department of Health - Fine Particles (PM 2.5) Questions and Answers." Accessed October 14, 2022. https://www.health.ny.gov/environmental/indoors/air/pmq_a.htm.

[2] Peixeiro, Marco. "3. Going on a Random Walk." Essay. In Time Series Forecasting in Python, 30–58. O'Reilly Media, 2022.

[3] Peixeiro, Marco. "5. Modeling an Autoregressive Process." Essay. In Time Series Forecasting in Python, 81–100. O'Reilly Media, 2022.

[4] Peixeiro, Marco. "8. Accounting for Seasonality." Essay. In Time Series Forecasting in Python, 156–79. O'Reilly Media, 2022.

[5] Services, Ministry of Citizens'. "The BC Data Catalogue." Province of British Columbia. Province of British Columbia, February 2, 2022. https://www2.gov.bc.ca/gov/content/data/bc-data-catalogue.

[6] "Stationarity and Detrending (ADF/KPSS)." statsmodels. Accessed October 17, 2022. https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html.