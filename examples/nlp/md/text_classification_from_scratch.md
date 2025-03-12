# Text classification from scratch

**Authors:** Mark Omernick, Francois Chollet<br>
**Date created:** 2019/11/06<br>
**Last modified:** 2025/03/05<br>
**Description:** Text sentiment classification starting from raw text files.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_from_scratch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_from_scratch.py)



---
## Introduction

This example shows how to do text classification starting from raw text (as
a set of text files on disk). We demonstrate the workflow on the IMDB sentiment
classification dataset (unprocessed version). We use the `TextVectorization` layer for
 word splitting & indexing.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow

import keras
import numpy as np
from keras import layers
```

---
## Load the data: IMDB movie review sentiment classification

Let's download the data and inspect its structure.


```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
```
</div>
    
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

    
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

    
  0     0    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0

    
  0 80.2M    0 16384    0     0   6024      0  3:52:45  0:00:02  3:52:43  6023

    
  0 80.2M    0 49152    0     0  12905      0  1:48:38  0:00:03  1:48:35 12904

    
  0 80.2M    0 98304    0     0  20783      0  1:07:27  0:00:04  1:07:23 20783

    
  0 80.2M    0  192k    0     0  34750      0  0:40:20  0:00:05  0:40:15 38183

    
  0 80.2M    0  320k    0     0  47626      0  0:29:26  0:00:06  0:29:20 63466

    
  0 80.2M    0  384k    0     0  52465      0  0:26:43  0:00:07  0:26:36 78917

    
  0 80.2M    0  528k    0     0  61978      0  0:22:37  0:00:08  0:22:29 99983

    
  0 80.2M    0  688k    0     0  70788      0  0:19:48  0:00:09  0:19:39  113k

    
  0 80.2M    0  768k    0     0  74425      0  0:18:50  0:00:10  0:18:40  117k

    
  1 80.2M    1  960k    0     0  83339      0  0:16:49  0:00:11  0:16:38  130k

    
  1 80.2M    1 1184k    0     0  93088      0  0:15:03  0:00:13  0:14:50  144k

    
  1 80.2M    1 1440k    0     0   105k      0  0:12:58  0:00:13  0:12:45  185k

    
  2 80.2M    2 1760k    0     0   120k      0  0:11:20  0:00:14  0:11:06  231k

    
  2 80.2M    2 1872k    0     0   118k      0  0:11:32  0:00:15  0:11:17  211k

    
  2 80.2M    2 2016k    0     0   118k      0  0:11:33  0:00:17  0:11:16  202k

    
  2 80.2M    2 2112k    0     0   119k      0  0:11:25  0:00:17  0:11:08  201k

    
  2 80.2M    2 2320k    0     0   125k      0  0:10:55  0:00:18  0:10:37  180k

    
  3 80.2M    3 2592k    0     0   131k      0  0:10:23  0:00:19  0:10:04  163k

    
  3 80.2M    3 2880k    0     0   137k      0  0:09:56  0:00:20  0:09:36  196k

    
  3 80.2M    3 2960k    0     0   137k      0  0:09:57  0:00:21  0:09:36  208k

    
  4 80.2M    4 3296k    0     0   144k      0  0:09:27  0:00:22  0:09:05  231k

    
  4 80.2M    4 3552k    0     0   148k      0  0:09:14  0:00:23  0:08:51  225k

    
  4 80.2M    4 3680k    0     0   149k      0  0:09:09  0:00:24  0:08:45  221k

    
  4 80.2M    4 3968k    0     0   153k      0  0:08:54  0:00:25  0:08:29  221k

    
  5 80.2M    5 4112k    0     0   155k      0  0:08:48  0:00:26  0:08:22  235k

    
  5 80.2M    5 4384k    0     0   158k      0  0:08:38  0:00:27  0:08:11  221k

    
  5 80.2M    5 4672k    0     0   161k      0  0:08:28  0:00:28  0:08:00  227k

    
  5 80.2M    5 4816k    0     0   163k      0  0:08:23  0:00:29  0:07:54  231k

    
  6 80.2M    6 5344k    0     0   172k      0  0:07:57  0:00:31  0:07:26  263k

    
  6 80.2M    6 5472k    0     0   172k      0  0:07:55  0:00:31  0:07:24  260k

    
  7 80.2M    7 5760k    0     0   175k      0  0:07:49  0:00:32  0:07:17  263k

    
  7 80.2M    7 5904k    0     0   176k      0  0:07:46  0:00:33  0:07:13  267k

    
  7 80.2M    7 6224k    0     0   179k      0  0:07:38  0:00:34  0:07:04  269k

    
  7 80.2M    7 6400k    0     0   178k      0  0:07:39  0:00:35  0:07:04  223k

    
  8 80.2M    8 6720k    0     0   183k      0  0:07:27  0:00:36  0:06:51  253k

    
  8 80.2M    8 6992k    0     0   184k      0  0:07:24  0:00:37  0:06:47  250k

    
  8 80.2M    8 7248k    0     0   188k      0  0:07:15  0:00:38  0:06:37  272k

    
  9 80.2M    9 7456k    0     0   188k      0  0:07:16  0:00:39  0:06:37  250k

    
  9 80.2M    9 7776k    0     0   190k      0  0:07:11  0:00:40  0:06:31  269k

    
  9 80.2M    9 7936k    0     0   191k      0  0:07:09  0:00:41  0:06:28  247k

    
 10 80.2M   10 8272k    0     0   193k      0  0:07:04  0:00:42  0:06:22  260k

    
 10 80.2M   10 8608k    0     0   195k      0  0:06:59  0:00:43  0:06:16  246k

    
 10 80.2M   10 8768k    0     0   196k      0  0:06:57  0:00:44  0:06:13  266k

    
 10 80.2M   10 8944k    0     0   196k      0  0:06:58  0:00:45  0:06:13  247k

    
 11 80.2M   11 9408k    0     0   200k      0  0:06:50  0:00:47  0:06:03  266k

    
 11 80.2M   11 9568k    0     0   200k      0  0:06:49  0:00:47  0:06:02  263k

    
 12 80.2M   12 9904k    0     0   202k      0  0:06:45  0:00:48  0:05:57  263k

    
 12 80.2M   12  9.8M    0     0   201k      0  0:06:47  0:00:49  0:05:58  243k

    
 12 80.2M   12 10.1M    0     0   205k      0  0:06:40  0:00:50  0:05:50  287k

    
 12 80.2M   12 10.4M    0     0   205k      0  0:06:39  0:00:51  0:05:48  259k

    
 13 80.2M   13 10.5M    0     0   206k      0  0:06:38  0:00:52  0:05:46  259k

    
 13 80.2M   13 10.8M    0     0   207k      0  0:06:35  0:00:53  0:05:42  255k

    
 13 80.2M   13 11.1M    0     0   208k      0  0:06:33  0:00:54  0:05:39  277k

    
 14 80.2M   14 11.3M    0     0   209k      0  0:06:32  0:00:55  0:05:37  250k

    
 14 80.2M   14 11.6M    0     0   210k      0  0:06:30  0:00:56  0:05:34  260k

    
 14 80.2M   14 12.0M    0     0   212k      0  0:06:27  0:00:57  0:05:30  269k

    
 15 80.2M   15 12.3M    0     0   214k      0  0:06:22  0:00:58  0:05:24  286k

    
 15 80.2M   15 12.6M    0     0   215k      0  0:06:20  0:00:59  0:05:21  296k

    
 15 80.2M   15 12.7M    0     0   215k      0  0:06:21  0:01:00  0:05:21  281k

    
 16 80.2M   16 12.8M    0     0   212k      0  0:06:26  0:01:01  0:05:25  238k

    
 16 80.2M   16 12.9M    0     0   211k      0  0:06:28  0:01:02  0:05:26  204k

    
 16 80.2M   16 13.1M    0     0   210k      0  0:06:30  0:01:03  0:05:27  160k

    
 16 80.2M   16 13.3M    0     0   209k      0  0:06:31  0:01:04  0:05:27  140k

    
 16 80.2M   16 13.4M    0     0   209k      0  0:06:31  0:01:05  0:05:26  136k

    
 17 80.2M   17 13.6M    0     0   209k      0  0:06:32  0:01:06  0:05:26  162k

    
 17 80.2M   17 13.8M    0     0   208k      0  0:06:33  0:01:08  0:05:25  176k

    
 17 80.2M   17 14.0M    0   

<div class="k-default-codeblock">
```
  0   208k      0  0:06:33  0:01:08  0:05:25  189k

```
</div>
    
 17 80.2M   17 14.2M    0     0   208k      0  0:06:33  0:01:09  0:05:24  192k

    
 17 80.2M   17 14.3M    0     0   208k      0  0:06:33  0:01:10  0:05:23  195k

    
 18 80.2M   18 14.6M    0     0   209k      0  0:06:32  0:01:11  0:05:21  211k

    
 18 80.2M   18 14.9M    0     0   210k      0  0:06:30  0:01:12  0:05:18  231k

    
 18 80.2M   18 15.1M    0     0   210k      0  0:06:29  0:01:13  0:05:16  239k

    
 19 80.2M   19 15.6M    0     0   214k      0  0:06:22  0:01:14  0:05:08  304k

    
 19 80.2M   19 15.6M    0     0   212k      0  0:06:27  0:01:15  0:05:12  260k

    
 19 80.2M   19 15.7M    0     0   210k      0  0:06:30  0:01:16  0:05:14  223k

    
 19 80.2M   19 15.8M    0     0   209k      0  0:06:32  0:01:17  0:05:15  191k

    
 19 80.2M   19 16.0M    0     0   209k      0  0:06:32  0:01:18  0:05:14  182k

    
 20 80.2M   20 16.3M    0     0   210k      0  0:06:30  0:01:19  0:05:11  147k

    
 20 80.2M   20 16.6M    0     0   209k      0  0:06:31  0:01:21  0:05:10  177k

    
 20 80.2M   20 16.7M    0     0   209k      0  0:06:31  0:01:21  0:05:10  203k

    
 21 80.2M   21 16.9M    0     0   209k      0  0:06:31  0:01:22  0:05:09  219k

    
 21 80.2M   21 17.1M    0     0   209k      0  0:06:31  0:01:23  0:05:08  220k

    
 21 80.2M   21 17.3M    0     0   2

<div class="k-default-codeblock">
```
10k      0  0:06:31  0:01:24  0:05:07  208k

```
</div>
    
 22 80.2M   22 17.6M    0     0   210k      0  0:06:30  0:01:25  0:05:05  218k

    
 22 80.2M   22 17.7M    0     0   210k      0  0:06:30  0:01:26  0:05:04  221k

    
 22 80.2M   22 18.1M    0     0   210k      0  0:06:30  0:01:28  0:05:02  219k

    
 22 80.2M   22 18.4M    0     0   212k      0  0:06:27  0:01:29  0:04:58  245k

    
 23 80.2M   23 18.5M    0     0   212k      0  0:06:27  0:01:29  0:04:58  250k

    
 23 80.2M   23 18.8M    0     0   212k      0  0:06:25  0:01:30  0:04:55  257k

    
 23 80.2M   23 19.0M    0     0   213k      0  0:06:24  0:01:31  0:04:53  263k

    
 23 80.2M   23 19.2M    0     0   212k      0  0:06:26  0:01:32  0:04:54  260k

    
 24 80.2M   24 19.6M    0     0   214k      0  0:06:23  0:01:33  0:04:50  253k

    
 24 80.2M   24 19.7M    0     0   214k      0  0:06:23  0:01:34  0:04:49  253k

    
 25 80.2M   25 20.0M    0     0   214k      0  0:06:22  0:01:35  0:04:47  251k

    
 25 80.2M   25 20.4M    0     0   215k      0  0:06:21  0:01:36  0:04:45  248k

    
 25 80.2M   25 20.5M    0     0   215k      0  0:06:20  0:01:37  0:04:43  268k

    
 25 80.2M   25 20.8M    0     0   216k      0  0:06:20  0:01:38  0:04:42  250k

    
 26 80.2M   26 21.0M    0     0   216k      0  0:06:19  0:01:39  0:04:40  257k

    
 26 80.2M   26 21.3M    0     0   217k      0  0:06:18  0:01:40  0:04:38  262k

    
 27 80.2M   27 21.7M    0     0   217k      0  0:06:18  0:01:42  0:04:36  249k

    
 27 80.2M   27 21.8M    0     0   218k      0  0:06:15  0:01:42  0:04:33  276k

    
 27 80.2M   27 22.1M    0     0   218k      0  0:06:15  0:01:43  0:04:32  270k

    
 28 80.2M   28 22.4M    0     0   219k      0  0:06:14  0:01:44  0:04:30  266k

    
 28 80.2M   28 22.6M    0     0   219k      0  0:06:14  0:01:45  0:04:29  266k

    
 28 80.2M   28 22.7M    0     0   218k      0  0:06:15  0:01:46  0:04:29  260k

    
 28 80.2M   28 23.1M    0     0   220k      0  0:06:12  0:01:47  0:04:25  260k

    
 29 80.2M   29 23.3M    0     0   220k      0  0:06:12  0:01:48  0:04:24  253k

    
 29 80.2M   29 23.6M    0     0   220k      0  0:06:12  0:01:49  0:04:23  250k

    
 29 80.2M   29 23.8M    0     0   220k      0  0:06:12  0:01:50  0:04:22  244k

    
 30 80.2M   30 24.1M    0     0   221k      0  0:06:11  0:01:51  0:04:20  269k

    
 30 80.2M   30 24.4M    0     0   221k      0  0:06:10  0:01:52  0:04:18  243k

    
 30 80.2M   30 24.6M    0     0   222k      0  0:06:10  0:01:53  0:04:17  260k

    
 31 80.2M   31 24.9M    0     0   222k      0  0:06:09  0:01:54  0:04:15  266k

    
 31 80.2M   31 25.1M    0     0   223k      0  0:06:07  0:01:55  0:04:12  285k

    
 31 80.2M   31 25.2M    0     0   222k      0  0:06:09  0:01:56  0:04:13  250k

    
 32 80.2M   32 25.7M    0     0   223k      0  0:06:07  0:01:57  0:04:10  266k

    
 32 80.2M   32 25.8M    0     0   223k      0  0:06:07  0:01:58  0:04:09  263k

    
 32 80.2M   32 26.3M    0     0   225k      0  0:06:03  0:01:59  0:04:04  299k

    
 33 80.2M   33 26.5M    0     0   225k      0  0:06:04  0:02:00  0:04:04  270k

    
 33 80.2M   33 26.7M    0     0   225k      0  0:06:04  0:02:01  0:04:03  289k

    
 33 80.2M   33 27.0M    0     0   226k      0  0:06:03  0:02:02  0:04:01  287k

    
 34 80.2M   34 27.3M    0     0   226k      0  0:06:02  0:02:03  0:03:59  294k

    
 34 80.2M   34 27.7M    0     0   227k      0  0:06:01  0:02:04  0:03:57  267k

    
 34 80.2M   34 28.0M    0     0   228k      0  0:06:00  0:02:05  0:03:55  290k

    
 35 80.2M   35 28.3M    0     0   228k      0  0:05:59  0:02:06  0:03:53  303k

    
 35 80.2M   35 28.5M    0     0   228k      0  0:05:59  0:02:07  0:03:52  297k

    
 35 80.2M   35 28.8M    0     0   228k      0  0:05:59  0:02:09  0:03:50  268k

    
 36 80.2M   36 29.0M    0     0   229k      0  0:05:57  0:02:09  0:03:48  288k

    
 36 80.2M   36 29.2M    0     0   229k      0  0:05:57  0:02:10  0:03:47  266k

    
 36 80.2M   36 29.5M    0     0   229k      0  0:05:57  0:02:11  0:03:46  257k

    
 37 80.2M   37 29.7M    0     0   229k      0  0:05:57  0:02:12  0:03:45  257k

    
 37 80.2M   37 30.2M    0     0   231k      0  0:05:55  0:02:14  0:03:41  306k

    
 37 80.2M   37 30.4M    0     0   231k      0  0:05:55  0:02:14  0:03:41  269k

    
 38 80.2M   38 30.6M    0     0   231k      0  0:05:55  0:02:15  0:03:40  272k

    
 38 80.2M   38 30.8M    0     0   231k      0  0:05:55  0:02:16  0:03:39  277k

    
 38 80.2M   38 31.1M    0     0   231k      0  0:05:54  0:02:17  0:03:37  272k

    
 39 80.2M   39 31.4M    0     0   232k      0  0:05:53  0:02:18  0:03:35  269k

    
 39 80.2M   39 31.7M    0     0   233k      0  0:05:52  0:02:19  0:03:33  292k

    
 39 80.2M   39 31.9M    0     0   232k      0  0:05:53  0:02:20  0:03:33  271k

    
 39 80.2M   39 32.0M    0     0   231k      0  0:05:55  0:02:21  0:03:34  232k

    
 40 80.2M   40 32.1M    0     0   230k      0  0:05:56  0:02:22  0:03:34  209k

    
 40 80.2M   40 32.2M    0     0   229k      0  0:05:57  0:02:23  0:03:34  154k

    
 40 80.2M   40 32.4M    0     0   228k      0  0:05:58  0:02:25  0:03:33  118k

    
 40 80.2M   40 32.5M    0     0   228k      0  0:05:59  0:02:25  0:03:34  115k

    
 40 80.2M   40 32.7M    0     0   228k      0  0:05:59  0:02:26  0:03:33  146k

    
 41 80.2M   41 33.0M    0     0   228k      0  0:05:58  0:02:27  0:03:31  180k

    
 41 80.2M   41 33.3M    0     0   229k      0  0:05:58  0:02:29  0:03:29  220k

    
 41 80.2M   41 33.5M    0     0   229k      0  0:05:58  0:02:29  0:03:29  246k

    
 42 80.2M   42 33.8M    0     0   229k      0  0:05:57  0:02:30  0:03:27  257k

    
 42 80.2M   42 34.0M    0     0   229k      0  0:05:57  0:02:31  0:03:26  277k

    
 42 80.2M   42 34.3M    0     0   230k      0  0:05:56  0:02:32  0:03:24  266k

    
 43 80.2M   43 34.6M    0     0   230k      0  0:05:57  0:02:34  0:03:23  249k

    
 43 80.2M   43 34.8M    0     0   230k      0  0:05:55  0:02:34  0:03:21  273k

    
 43 80.2M   43 35.0M    0     0   230k      0  0:05:56  0:02:35  0:03:21  260k

    
 44 80.2M   44 35.4M    0     0   230k      0  0:05:55  0:02:36  0:03:19  257k

    
 44 80.2M   44 35.5M    0     0   231k      0  0:05:55  0:02:37  0:03:18  257k

    
 44 80.2M   44 35.8M    0     0   230k      0  0:05:56  0:02:39  0:03:17  253k

    
 44 80.2M   44 35.9M    0     0   230k      0  0:05:56  0:02:39  0:03:17  221k

    
 45 80.2M   45 36.3M    0     0   231k      0  0:05:54  0:02:40  0:03:14  260k

    
 45 80.2M   45 36.6M    0     0   231k      0  0:05:54  0:02:41  0:03:13  253k

    
 45 80.2M   45 36.7M    0     0   231k      0  0:05:54  0:02:42  0:03:12  250k

    
 46 80.2M   46 37.0M    0     0   232k      0  0:05:53  0:02:43  0:03:10  294k

    
 46 80.2M   46 37.4M    0     0   232k      0  0:05:53  0:02:44  0:03:09  294k

    
 47 80.2M   47 37.7M    0     0   232k      0  0:05:52  0:02:45  0:03:07  272k

    
 47 80.2M   47 37.8M    0     0   233k      0  0:05:52  0:02:46  0:03:06  281k

    
 47 80.2M   47 38.2M    0     0   233k      0  0:05:51  0:02:47  0:03:04  287k

    
 47 80.2M   47 38.3M    0     0   232k      0  0:05:52  0:02:48  0:03:04  249k

    
 48 80.2M   48 38.7M    0     0   233k      0  0:05:51  0:02:49  0:03:02  270k

    
 48 80.2M   48 38.9M    0     0   233k      0  0:05:51  0:02:50  0:03:01  260k

    
 48 80.2M   48 39.1M    0     0   233k      0  0:05:51  0:02:51  0:03:00  266k

    
 49 80.2M   49 39.4M    0     0   234k      0  0:05:50  0:02:52  0:02:58  257k

    
 49 80.2M   49 39.6M    0     0   233k      0  0:05:51  0:02:53  0:02:58  265k

    
 49 80.2M   49 40.0M    0     0   234k      0  0:05:49  0:02:54  0:02:55  275k

    
 50 80.2M   50 40.3M    0     0   234k      0  0:05:49  0:02:55  0:02:54  278k

    
 50 80.2M   50 40.5M    0     0   234k      0  0:05:49  0:02:56  0:02:53  269k

    
 50 80.2M   50 40.8M    0     0   235k  

<div class="k-default-codeblock">
```
    0  0:05:48  0:02:57  0:02:51  289k

```
</div>
    
 51 80.2M   51 41.1M    0     0   235k      0  0:05:48  0:02:58  0:02:50  306k

    
 51 80.2M   51 41.4M    0     0   236k      0  0:05:47  0:02:59  0:02:48  292k

    
 52 80.2M   52 41.8M    0     0   237k      0  0:05:46  0:03:00  0:02:46  319k

    
 52 80.2M   52 42.1M    0     0   237k      0  0:05:46  0:03:01  0:02:45  318k

    
 52 80.2M   52 42.3M    0     0   237k      0  0:05:46  0:03:02  0:02:44  285k

    
 53 80.2M   53 42.6M    0     0   237k      0  0:05:45  0:03:03  0:02:42  304k

    
 53 80.2M   53 42.9M    0     0   237k      0  0:05:45  0:03:04  0:02:41  284k

    
 53 80.2M   53 43.2M    0     0   238k      0  0:05:44  0:03:05  0:02:39  271k

    
 54 80.2M   54 43.3M    0     0   237k      0  0:05:45  0:03:06  0:02:39  251k

    
 54 80.2M   54 43.7M    0     0   238k      0  0:05:44  0:03:07  0:02:37  302k

    
 54 80.2M   54 43.9M    0     0   238k      0  0:05:44  0:03:08  0:02:36  268k

    
 55 80.2M   55 44.2M    0     0   238k      0  0:05:44  0:03:09  0:02:35  263k

    
 55 80.2M   55 44.4M    0     0   238k      0  0:05:44  0:03:10  0:02:34  256k

    
 55 80.2M   55 44.7M    0     0   238k      0  0:05:44  0:03:11  0:02:33  275k

    
 55 80.2M   55 44.9M    0     0   239k      0  0:05:43  0:03:12  0:02:31  253k

    
 56 80.2M   56 45.1M    0     0   238k      0  0:05:43  0:03:13  0:02:30  253k

    
 56 80.2M   56 45.5M    0     0   239k      0  0:05:43  0:03:14  0:02:29  263k

    
 56 80.2M   56 45.7M    0     0   239k      0  0:05:43  0:03:15  0:02:28  270k

    
 57 80.2M   57 45.8M    0     0   238k      0  0:05:43  0:03:16  0:02:27  247k

    
 57 80.2M   57 46.3M    0     0   239k      0  0:05:42  0:03:17  0:02:25  273k

    
 57 80.2M   57 46.5M    0     0   239k      0  0:05:42  0:03:18  0:02:24  276k

    
 58 80.2M   58 46.6M    0     0   239k      0  0:05:43  0:03:19  0:02:24  244k

    
 58 80.2M   58 46.8M    0     0   238k      0  0:05:44  0:03:20  0:02:24  213k

    
 58 80.2M   58 46.9M    0     0   238k      0  0:05:44  0:03:21  0:02:23  221k

    
 58 80.2M   58 47.1M    0     0   238k      0  0:05:45  0:03:22  0:02:23  165k

    
 59 80.2M   59 47.3M    0     0   237k      0  0:05:45  0:03:23  0:02:22  162k

    
 59 80.2M   59 47.4M    0     0   237k      0  0:05:45  0:03:24  0:02:21  175k

    
 59 80.2M   59 47.7M    0     0   237k      0  0:05:45  0:03:25  0:02:20  192k

    
 59 80.2M   59 47.9M    0     0   237k      0  0:05:46  0:03:26  0:02:20  196k

    
 59 80.2M   59 48.1M    0     0   237k      0  0:05:46  0:03:27  0:02:19  205k

    
 60 80.2M   60 48.3M    0     0   237k      0  0:05:46  0:03:28  0:02:18  214k

    
 60 80.2M   60 48.6M    0     0   237k      0  0:05:45  0:03:29  0:02:16  236k

    
 60 80.2M   60 48.8M    0     0   237k      0  0:05:45  0:03:30  0:02:15  237k

    
 61 80.2M   61 49.0M    0     0   237k      0  0:05:46  0:03:31  0:02:15  227k

    
 61 80.2M   61 49.2M    0     0   237k      0  0:05:46  0:03:32  0:02:14  234k

    
 61 80.2M   61 49.6M    0     0   237k      0  0:05:45  0:03:33  0:02:12  263k

    
 62 80.2M   62 49.9M    0     0   238k      0  0:05:45  0:03:34  0:02:11  252k

    
 62 80.2M   62 50.1M    0     0   237k      0  0:05:45  0:03:35  0:02:10  243k

    
 62 80.2M   62 50.4M    0     0   238k      0  0:05:44  0:03:36  0:02:08  284k

    
 63 80.2M   63 50.7M    0     0   238k      0  0:05:44  0:03:38  0:02:06  274k

    
 63 80.2M   63 50.8M    0     0   238k      0  0:05:44  0:03:38  0:02:06  253k

    
 63 80.2M   63 51.1M    0     0   238k      0  0:05:44  0:03:39  0:02:05  267k

    
 64 80.2M   64 51.3M    0     0   238k      0  0:05:44  0:03:40  0:02:04  277k

    
 64 80.2M   64 51.4M    0     0   237k      0  0:05:45  0:03:41  0:02:04  211k

    
 64 80.2M   64 51.5M    0     0   236k      0  0:05:46  0:03:42  0:02:04  182k

    
 64 80.2M   64 51.6M    0     0   236k      0  0:05:47  0:03:43  0:02:04  169k

    
 64 80.2M   64 51.8M    0     0   236k      0  0:05:47  0:03:44  0:02:03  134k

    
 64 80.2M   64 52.1M    0     0   236k      0  0:05:47  0:03:46  0:02:01  138k

    
 65 80.2M   65 52.2M    0     0   236k      0  0:05:47  0:03:46  0:02:01  162k

    
 65 80.2M   65 52.5M    0     0   236k      0  0:05:48  0:03:47  0:02:01  192k

    
 65 80.2M   65 52.6M    0     0   236k      0  0:05:48  0:03:48  0:02:00  201k

    
 65 80.2M   65 52.9M    0     0   235k      0  0:05:48  0:03:49  0:01:59  218k

    
 66 80.2M   66 53.2M    0     0   235k      0  0:05:48  0:03:50  0:01:58  229k

    
 66 80.2M   66 53.5M    0     0   236k      0  0:05:47  0:03:51  0:01:56  245k

    
 66 80.2M   66 53.6M    0     0   236k      0  0:05:47  0:03:52  0:01:55  244k

    
 67 80.2M   67 53.7M    0     0   235k      0  0:05:49  0:03:54  0:01:55  208k

    
 67 80.2M   67 53.8M    0     0   234k      0  0:05:49  0:03:54  0:01:55  188k

    
 67 80.2M   67 53.9M    0     0   234k      0  0:05:50  0:03:55  0:01:55  158k

    
 67 80.2M   67 54.0M    0     0   234k      0  0:05:50  0:03:56  0:01:54  125k

    
 67 80.2M   67 54.3M    0     0   234k      0  0:05:51  0:03:57  0:01:54  137k

    
 67 80.2M   67 54.5M    0     0   234k      0  0:05:51  0:03:58  0:01:53  166k

    
 68 80.2M   68 54.8M    0     0   234k      0  0:05:50  0:03:59  0:01:51  208k

    
 68 80.2M   68 55.1M    0     0   234k      0  0:05:50  0:04:00  0:01:50  249k

    
 69 80.2M   69 55.4M    0     0   234k      0  0:05:49  0:04:01  0:01:48  273k

    
 69 80.2M   69 55.7M    0     0   235k      0  0:05:49  0:04:02  0:01:47  304k

    
 69 80.2M   69 56.1M    0     0   235k      0  0:05:48  0:04:03  0:01:45  320k

    
 70 80.2M   70 56.4M    0     0   236k      0  0:05:47  0:04:04  0:01:43  325k

    
 70 80.2M   70 56.6M    0     0   236k      0  0:05:48  0:04:05  0:01:43  302k

    
 70 80.2M   70 56.9M    0     0   236k      0  0:05:47  0:04:06  0:01:41  314k

    
 71 80.2M   71 57.2M    0     0   236k      0  0:05:47  0:04:07  0:01:40  291k

    
 71 80.2M   71 57.5M    0     0   236k      0  0:05:47  0:04:08  0:01:39  274k

    
 71 80.2M   71 57.7M    0     0   236k      0  0:05:46  0:04:09  0:01:37  260k

    
 72 80.2M   72 57.8M    0     0   236k      0  0:05:47  0:04:10  0:01:37  260k

    
 72 80.2M   72 58.3M    0     0   236k      0  0:05:46  0:04:12  0:01:34  257k

    
 72 80.2M   72 58.4M    0     0   236k      0  0:05:46  0:04:12  0:01:34  257k

    
 73 80.2M   73 58.7M    0     0   237k      0  0:05:46  0:04:13  0:01:33  270k

    
 73 80.2M   73 59.0M    0     0   237k      0  0:05:45  0:04:14  0:01:31  283k

    
 73 80.2M   73 59.1M    0     0   236k      0  0:05:46  0:04:15  0:01:31  253k

    
 73 80.2M   73 59.2M    0     0   236k      0  0:05:47  0:04:16  0:01:31  195k

    
 73 80.2M   73 59.3M    0     0   235k      0  0:05:48  0:04:17  0:01:31  185k

    
 74 80.2M   74 59.5M    0     0   235k      0  0:05:48  0:04:18  0:01:30  165k

    
 74 80.2M   74 59.7M    0     0   235k      0  0:05:48  0:04:19  0:01:29  146k

    
 74 80.2M   74 60.0M    0     0   235k      0  0:05:48  0:04:20  0:01:28  184k

    
 75 80.2M   75 60.3M    0     0   235k      0  0:05:48  0:04:21  0:01:27  213k

    
 75 80.2M   75 60.4M    0     0   235k      0  0:05:48  0:04:22  0:01:26  223k

    
 75 80.2M   75 60.7M    0     0   235k      0  0:05:48  0:04:23  0:01:25  222k

    
 75 80.2M   75 60.9M    0     0   23

<div class="k-default-codeblock">
```
6k      0  0:05:47  0:04:24  0:01:23  246k

```
</div>
    
 76 80.2M   76 61.1M    0     0   235k      0  0:05:48  0:04:25  0:01:23  224k

    
 76 80.2M   76 61.4M    0     0   235k      0  0:05:48  0:04:26  0:01:22  237k

    
 77 80.2M   77 61.7M    0     0   236k      0  0:05:47  0:04:27  0:01:20  265k

    
 77 80.2M   77 62.0M    0     0   236k      0  0:05:47  0:04:28  0:01:19  271k

    
 77 80.2M   77 62.2M    0     0   236k      0  0:05:47  0:04:29  0:01:18  248k

    
 77 80.2M   77 62.5M    0     0   236k      0  0:05:47  0:04:30  0:01:17  278k

    
 78 80.2M   78 62.8M    0     0   236k      0  0:05:47  0:04:32  0:01:15  260k

    
 78 80.2M   78 62.8M    0     0   236k      0  0:05:47  0:04:32  0:01:15  227k

    
 78 80.2M   78 63.3M    0     0   236k      0  0:05:47  0:04:33  0:01:14  260k

    
 79 80.2M   79 63.4M    0     0   236k      0  0:05:47  0:04:34  0:01:13  260k

    
 79 80.2M   79 63.7M    0     0   236k      0  0:05:46  0:04:35  0:01:11  257k

    
 79 80.2M   79 64.0M    0     0   237k      0  0:05:46  0:04:36  0:01:10  293k

    
 80 80.2M   80 64.2M    0     0   237k      0  0:05:46  0:04:37  0:01:09  289k

    
 80 80.2M   80 64.3M    0     0   236k      0  0:05:47  0:04:38  0:01:09  211k

    
 80 80.2M   80 64.4M    0     0   235k      0  0:05:48  0:04:39  0:01:09  188k

    
 80 80.2M   80 64.5M    0     0   235k      0  0:05:48  0:04:40  0:01:08  166k

    
 80 80.2M   80 64.7M    0     0   235k      0  0:05:49  0:04:41  0:01:08  131k

    
 80 80.2M   80 64.9M    0     0   235k      0  0:05:49  0:04:42  0:01:07  132k

    
 81 80.2M   81 65.0M    0     0   235k      0  0:05:49  0:04:43  0:01:06  162k

    
 81 80.2M   81 65.3M    0     0   235k      0  0:05:49  0:04:44  0:01:05  195k

    
 81 80.2M   81 65.7M    0     0   235k      0  0:05:49  0:04:46  0:01:03  217k

    
 82 80.2M   82 65.8M    0     0   235k      0  0:05:49  0:04:46  0:01:03  237k

    
 82 80.2M   82 66.1M    0     0   235k      0  0:05:48  0:04:47  0:01:01  257k

    
 82 80.2M   82 66.3M    0     0   235k      0  0:05:48  0:04:48  0:01:00  263k

    
 83 80.2M   83 66.7M    0     0   235k      0  0:05:48  0:04:49  0:00:59  273k

    
 83 80.2M   83 66.8M    0     0   235k      0  0:05:48  0:04:50  0:00:58  251k

    
 83 80.2M   83 67.0M    0     0   235k      0  0:05:48  0:04:51  0:00:57  250k

    
 84 80.2M   84 67.4M    0     0   235k      0  0:05:48  0:04:52  0:00:56  266k

    
 84 80.2M   84 67.7M    0     0   236k      0  0:05:47  0:04:54  0:00:53  266k

    
 84 80.2M   84 67.9M    0     0   235k      0  0:05:48  0:04:55  0:00:53  240k

    
 85 80.2M   85 68.2M    0     0   236k      0  0:05:47  0:04:55  0:00:52  281k

    
 85 80.2M   85 68.3M    0     0   236k      0  0:05:47  0:04:56  0:00:51  276k

    
 85 80.2M   85 68.8M    0     0   236k      0  0:05:47  0:04:57  0:00:50  283k

    
 86 80.2M   86 69.1M    0     0   237k      0  0:05:46  0:04:58  0:00:48  303k

    
 86 80.2M   86 69.6M    0     0   237k      0  0:05:45  0:04:59  0:00:46  348k

    
 86 80.2M   86 69.7M    0     0   237k      0  0:05:45  0:05:00  0:00:45  329k

    
 87 80.2M   87 70.1M    0     0   237k      0  0:05:45  0:05:01  0:00:44  330k

    
 87 80.2M   87 70.4M    0     0   238k      0  0:05:45  0:05:03  0:00:42  312k

    
 88 80.2M   88 70.6M    0     0   238k      0  0:05:44  0:05:03  0:00:41  300k

    
 88 80.2M   88 70.9M    0     0   238k      0  0:05:44  0:05:04  0:00:40  302k

    
 88 80.2M   88 71.2M    0     0   238k      0  0:05:44  0:05:05  0:00:39  291k

    
 89 80.2M   89 71.5M    0     0   238k      0  0:05:44  0:05:07  0:00:37  291k

    
 89 80.2M   89 71.7M    0     0   238k      0  0:05:44  0:05:08  0:00:36  264k

    
 89 80.2M   89 72.0M    0     0   238k      0  0:05:43  0:05:08  0:00:35  280k

    
 89 80.2M   89 72.1M    0     0   238k      0  0:05:43  0:05:09  0:00:34  250k

    
 90 80.2M   90 72.4M    0     0   238k      0  0:05:43  0:05:10  0:00:33  250k

    
 90 80.2M   90 72.7M    0     0   238k      0  0:05:43  0:05:11  0:00:32  244k

    
 90 80.2M   90 72.9M    0     0   238k      0  0:05:43  0:05:12  0:00:31  266k

    
 91 80.2M   91 73.3M    0     0   239k      0  0:05:42  0:05:13  0:00:29  281k

    
 91 80.2M   91 73.7M    0     0   239k      0  0:05:42  0:05:14  0:00:28  305k

    
 91 80.2M   91 73.7M    0     0   239k      0  0:05:43  0:05:15  0:00:28  265k

    
 92 80.2M   92 73.8M    0     0   239k      0  0:05:43  0:05:16  0:00:27  255k

    
 92 80.2M   92 73.9M    0     0   238k      0  0:05:44  0:05:17  0:00:27  209k

    
 92 80.2M   92 74.1M    0     0   238k      0  0:05:44  0:05:18  0:00:26  159k

    
 92 80.2M   92 74.3M    0     0   238k      0  0:05:44  0:05:19  0:00:25  131k

    
 93 80.2M   93 74.6M    0     0   238k      0  0:05:44  0:05:20  0:00:24  168k

    
 93 80.2M   93 74.7M    0     0   238k      0  0:05:44  0:05:21  0:00:23  178k

    
 93 80.2M   93 75.0M    0     0   238k      0  0:05:44  0:05:22  0:00:22  225k

    
 93 80.2M   93 75.4M    0     0   238k      0  0:05:44  0:05:24  0:00:20  235k

    
 94 80.2M   94 75.5M    0     0   238k      0  0:05:44  0:05:24  0:00:20  250k

    
 94 80.2M   94 75.8M    0     0   238k      0  0:05:44  0:05:25  0:00:19  260k

    
 94 80.2M   94 76.0M    0     0   238k      0  0:05:44  0:05:26  0:00:18  266k

    
 95 80.2M   95 76.4M    0     0   238k      0  0:05:44  0:05:28  0:00:16  252k

    
 95 80.2M   95 76.7M    0     0   238k      0  0:05:43  0:05:28  0:00:15  273k

    
 95 80.2M   95 76.8M    0     0   238k      0  0:05:44  0:05:29  0:00:15  263k

    
 96 80.2M   96 77.1M    0     0   238k      0  0:05:44  0:05:30  0:00:14  260k

    
 96 80.2M   96 77.4M    0     0   238k      0  0:05:43  0:05:31  0:00:12  263k

    
 96 80.2M   96 77.6M    0     0   238k      0  0:05:44  0:05:33  0:00:11  257k

    
 97 80.2M   97 77.9M    0     0   239k      0  0:05:43  0:05:33  0:00:10  260k

    
 97 80.2M   97 78.0M    0     0   239k      0  0:05:43  0:05:34  0:00:09  260k

    
 97 80.2M   97 78.3M    0     0   239k      0  0:05:43  0:05:35  0:00:08  260k

    
 98 80.2M   98 78.7M    0     0   239k      0  0:05:43  0:05:36  0:00:07  253k

    
 98 80.2M   98 78.8M    0     0   239k      0  0:05:43  0:05:37  0:00:06  277k

    
 98 80.2M   98 79.1M    0     0   239k      0  0:05:43  0:05:38  0:00:05  253k

    
 99 80.2M   99 79.5M    0     0   239k      0  0:05:43  0:05:39  0:00:04  263k

    
 99 80.2M   99 79.6M    0     0   239k      0  0:05:42  0:05:40  0:00:02  266k

    
 99 80.2M   99 80.0M    0     0   239k      0  0:05:43  0:05:42  0:00:01  249k

    
 99 80.2M   99 80.1M    0     0   239k      0  0:05:42  0:05:42 --:--:--  276k

    
100 80.2M  100 80.2M    0     0   239k      0  0:05:43  0:05:43 --:--:--  247k


The `aclImdb` folder contains a `train` and `test` subfolder:


```python
!ls aclImdb
```

```python
!ls aclImdb/test
```
```python
!ls aclImdb/train
```
<div class="k-default-codeblock">
```
imdbEr.txt  imdb.vocab	README	test  train

labeledBow.feat  neg  pos  urls_neg.txt  urls_pos.txt

labeledBow.feat  pos	unsupBow.feat  urls_pos.txt
neg		 unsup	urls_neg.txt   urls_unsup.txt

```
</div>
The `aclImdb/train/pos` and `aclImdb/train/neg` folders contain text files, each of
 which represents one review (either positive or negative):


```python
!cat aclImdb/train/pos/6248_7.txt
```

<div class="k-default-codeblock">
```
Being an Austrian myself this has been a straight knock in my face. Fortunately I don't live nowhere near the place where this movie takes place but unfortunately it portrays everything that the rest of Austria hates about Viennese people (or people close to that region). And it is very easy to read that this is exactly the directors intention: to let your head sink into your hands and say "Oh my god, how can THAT be possible!". No, not with me, the (in my opinion) totally exaggerated uncensored swinger club scene is not necessary, I watch porn, sure, but in this context I was rather disgusted than put in the right context.<br /><br />This movie tells a story about how misled people who suffer from lack of education or bad company try to survive and live in a world of redundancy and boring horizons. A girl who is treated like a whore by her super-jealous boyfriend (and still keeps coming back), a female teacher who discovers her masochism by putting the life of her super-cruel "lover" on the line, an old couple who has an almost mathematical daily cycle (she is the "official replacement" of his ex wife), a couple that has just divorced and has the ex husband suffer under the acts of his former wife obviously having a relationship with her masseuse and finally a crazy hitchhiker who asks her drivers the most unusual questions and stretches their nerves by just being super-annoying.<br /><br />After having seen it you feel almost nothing. You're not even shocked, sad, depressed or feel like doing anything... Maybe that's why I gave it 7 points, it made me react in a way I never reacted before. If that's good or bad is up to you!

```
</div>
We are only interested in the `pos` and `neg` subfolders, so let's delete the other subfolder that has text files in it:


```python
!rm -r aclImdb/train/unsup
```

You can use the utility `keras.utils.text_dataset_from_directory` to
generate a labeled `tf.data.Dataset` object from a set of text files on disk filed
 into class-specific folders.

Let's use it to generate the training, validation, and test datasets. The validation
and training datasets are generated from two subsets of the `train` directory, with 20%
of samples going to the validation dataset and 80% going to the training dataset.

Having a validation dataset in addition to the test dataset is useful for tuning
hyperparameters, such as the model architecture, for which the test dataset should not
be used.

Before putting the model out into the real world however, it should be retrained using all
available training data (without creating a validation dataset), so its performance is maximized.

When using the `validation_split` & `subset` arguments, make sure to either specify a
random seed, or to pass `shuffle=False`, so that the validation & training splits you
get have no overlap.


```python
batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
```

<div class="k-default-codeblock">
```
Found 25000 files belonging to 2 classes.

Using 20000 files for training.

Found 25000 files belonging to 2 classes.

Using 5000 files for validation.

Found 25000 files belonging to 2 classes.

Number of batches in raw_train_ds: 625
Number of batches in raw_val_ds: 157
Number of batches in raw_test_ds: 782

```
</div>
Let's preview a few samples:


```python
# It's important to take a look at your raw data to ensure your normalization
# and tokenization will work as expected. We can do that by taking a few
# examples from the training set and looking at them.
# This is one of the places where eager execution shines:
# we can just evaluate these tensors using .numpy()
# instead of needing to evaluate them in a Session/Graph context.
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
```

<div class="k-default-codeblock">
```
b'I\'ve seen tons of science fiction from the 70s; some horrendously bad, and others thought provoking and truly frightening. Soylent Green fits into the latter category. Yes, at times it\'s a little campy, and yes, the furniture is good for a giggle or two, but some of the film seems awfully prescient. Here we have a film, 9 years before Blade Runner, that dares to imagine the future as somthing dark, scary, and nihilistic. Both Charlton Heston and Edward G. Robinson fare far better in this than The Ten Commandments, and Robinson\'s assisted-suicide scene is creepily prescient of Kevorkian and his ilk. Some of the attitudes are dated (can you imagine a filmmaker getting away with the "women as furniture" concept in our oh-so-politically-correct-90s?), but it\'s rare to find a film from the Me Decade that actually can make you think. This is one I\'d love to see on the big screen, because even in a widescreen presentation, I don\'t think the overall scope of this film would receive its due. Check it out.'
1
b'First than anything, I\'m not going to praise I\xc3\xb1arritu\'s short film, even I\'m Mexican and proud of his success in mainstream Hollywood.<br /><br />In another hand, I see most of the reviews focuses on their favorite (and not so) short films; but we are forgetting that there is a subtle bottom line that circles the whole compilation, and maybe it will not be so pleasant for American people. (Even if that was not the main purpose of the producers) <br /><br />What i\'m talking about is that most of the short films does not show the suffering that WASP people went through because the terrorist attack on September 11th, but the suffering of the Other people.<br /><br />Do you need proofs about what i\'m saying? Look, in the Bosnia short film, the message is: "You cry because of the people who died in the Towers, but we (The Others = East Europeans) are crying long ago for the crimes committed against our women and nobody pay attention to us like the whole world has done to you".<br /><br />Even though the Burkina Fasso story is more in comedy, there is a the same thought: "You are angry because Osama Bin Laden punched you in an evil way, but we (The Others = Africans) should be more angry, because our people is dying of hunger, poverty and AIDS long time ago, and nobody pay attention to us like the whole world has done to you".<br /><br />Look now at the Sean Penn short: The fall of the Twin Towers makes happy to a lonely (and alienated) man. So the message is that the Power and the Greed (symbolized by the Towers) must fall for letting the people see the sun rise and the flowers blossom? It is remarkable that this terrible bottom line has been proposed by an American. There is so much irony in this short film that it is close to be subversive.<br /><br />Well, the Ken Loach (very know because his anti-capitalism ideology) is much more clearly and shameless in going straight to the point: "You are angry because your country has been attacked by evil forces, but we (The Others = Latin Americans) suffered at a similar date something worst, and nobody remembers our grief as the whole world has done to you".<br /><br />It is like if the creative of this project wanted to say to Americans: "You see now, America? You are not the only that have become victim of the world violence, you are not alone in your pain and by the way, we (the Others = the Non Americans) have been suffering a lot more than you from long time ago; so, we are in solidarity with you in your pain... and by the way, we are sorry because you have had some taste of your own medicine" Only the Mexican and the French short films showed some compassion and sympathy for American people; the others are like a slap on the face for the American State, that is not equal to American People.'
1
b'Blood Castle (aka Scream of the Demon Lover, Altar of Blood, Ivanna--the best, but least exploitation cinema-sounding title, and so on) is a very traditional Gothic Romance film. That means that it has big, creepy castles, a headstrong young woman, a mysterious older man, hints of horror and the supernatural, and romance elements in the contemporary sense of that genre term. It also means that it is very deliberately paced, and that the film will work best for horror mavens who are big fans of understatement. If you love films like Robert Wise\'s The Haunting (1963), but you also have a taste for late 1960s/early 1970s Spanish and Italian horror, you may love Blood Castle, as well.<br /><br />Baron Janos Dalmar (Carlos Quiney) lives in a large castle on the outskirts of a traditional, unspecified European village. The locals fear him because legend has it that whenever he beds a woman, she soon after ends up dead--the consensus is that he sets his ferocious dogs on them. This is quite a problem because the Baron has a very healthy appetite for women. At the beginning of the film, yet another woman has turned up dead and mutilated.<br /><br />Meanwhile, Dr. Ivanna Rakowsky (Erna Sch\xc3\xbcrer) has appeared in the center of the village, asking to be taken to Baron Dalmar\'s castle. She\'s an out-of-towner who has been hired by the Baron for her expertise in chemistry. Of course, no one wants to go near the castle. Finally, Ivanna finds a shady individual (who becomes even shadier) to take her. Once there, an odd woman who lives in the castle, Olga (Cristiana Galloni), rejects Ivanna and says that she shouldn\'t be there since she\'s a woman. Baron Dalmar vacillates over whether she should stay. She ends up staying, but somewhat reluctantly. The Baron has hired her to try to reverse the effects of severe burns, which the Baron\'s brother, Igor, is suffering from.<br /><br />Unfortunately, the Baron\'s brother appears to be just a lump of decomposing flesh in a vat of bizarre, blackish liquid. And furthermore, Ivanna is having bizarre, hallucinatory dreams. Just what is going on at the castle? Is the Baron responsible for the crimes? Is he insane? <br /><br />I wanted to like Blood Castle more than I did. As I mentioned, the film is very deliberate in its pacing, and most of it is very understated. I can go either way on material like that. I don\'t care for The Haunting (yes, I\'m in a very small minority there), but I\'m a big fan of 1960s and 1970s European horror. One of my favorite directors is Mario Bava. I also love Dario Argento\'s work from that period. But occasionally, Blood Castle moved a bit too slow for me at times. There are large chunks that amount to scenes of not very exciting talking alternated with scenes of Ivanna slowly walking the corridors of the castle.<br /><br />But the atmosphere of the film is decent. Director Jos\xc3\xa9 Luis Merino managed more than passable sets and locations, and they\'re shot fairly well by Emanuele Di Cola. However, Blood Castle feels relatively low budget, and this is a Roger Corman-produced film, after all (which usually means a low-budget, though often surprisingly high quality "quickie"). So while there is a hint of the lushness of Bava\'s colors and complex set decoration, everything is much more minimalist. Of course, it doesn\'t help that the Retromedia print I watched looks like a 30-year old photograph that\'s been left out in the sun too long. It appears "washed out", with compromised contrast.<br /><br />Still, Merino and Di Cola occasionally set up fantastic visuals. For example, a scene of Ivanna walking in a darkened hallway that\'s shot from an exaggerated angle, and where an important plot element is revealed through shadows on a wall only. There are also a couple Ingmar Bergmanesque shots, where actors are exquisitely blocked to imply complex relationships, besides just being visually attractive and pulling your eye deep into the frame.<br /><br />The performances are fairly good, and the women--especially Sch\xc3\xbcrer--are very attractive. Merino exploits this fact by incorporating a decent amount of nudity. Sch\xc3\xbcrer went on to do a number of films that were as much soft corn porn as they were other genres, with English titles such as Sex Life in a Woman\'s Prison (1974), Naked and Lustful (1974), Strip Nude for Your Killer (1975) and Erotic Exploits of a Sexy Seducer (1977). Blood Castle is much tamer, but in addition to the nudity, there are still mild scenes suggesting rape and bondage, and of course the scenes mixing sex and death.<br /><br />The primary attraction here, though, is probably the story, which is much a slow-burning romance as anything else. The horror elements, the mystery elements, and a somewhat unexpected twist near the end are bonuses, but in the end, Blood Castle is a love story, about a couple overcoming various difficulties and antagonisms (often with physical threats or harms) to be together.'
1
b"I was talked into watching this movie by a friend who blubbered on about what a cute story this was.<br /><br />Yuck.<br /><br />I want my two hours back, as I could have done SO many more productive things with my time...like, for instance, twiddling my thumbs. I see nothing redeeming about this film at all, save for the eye-candy aspect of it...<br /><br />3/10 (and that's being generous)"
0
b"Michelle Rodriguez is the defining actress who could be the charging force for other actresses to look out for. She has the audacity to place herself in a rarely seen tough-girl role very early in her career (and pull it off), which is a feat that should be recognized. Although her later films pigeonhole her to that same role, this film was made for her ruggedness.<br /><br />Her character is a romanticized student/fighter/lover, struggling to overcome her disenchanted existence in the projects, which is a little overdone in film...but not by a girl. That aspect of this film isn't very original, but the story goes in depth when the heated relationships that this girl has to deal with come to a boil and her primal rage takes over.<br /><br />I haven't seen an actress take such an aggressive stance in movie-making yet, and I'm glad that she's getting that original twist out there in Hollywood. This film got a 7 from me because of the average story of ghetto youth, but it has such a great actress portraying a rarely-seen role in a minimal budget movie. Great work."
1

```
</div>
---
## Prepare the data

In particular, we remove `<br />` tags.


```python
import string
import re
import tensorflow as tf


# Having looked at our data above, we see that the raw text contains HTML break
# tags of the form '<br />'. These tags will not be removed by the default
# standardizer (which doesn't strip HTML). Because of this, we will need to
# create a custom standardization function.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Now that the vectorize_layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)
```

---
## Two options to vectorize the data

There are 2 ways we can use our text vectorization layer:

**Option 1: Make it part of the model**, so as to obtain a model that processes raw
 strings, like this:

```python
text_input = keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorize_layer(text_input)
x = layers.Embedding(max_features + 1, embedding_dim)(x)
...
```

**Option 2: Apply it to the text dataset** to obtain a dataset of word indices, then
 feed it into a model that expects integer sequences as inputs.

An important difference between the two is that option 2 enables you to do
**asynchronous CPU processing and buffering** of your data when training on GPU.
So if you're training the model on GPU, you probably want to go with this option to get
 the best performance. This is what we will do below.

If we were to export our model to production, we'd ship a model that accepts raw
strings as input, like in the code snippet for option 1 above. This can be done after
 training. We do this in the last section.



```python

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
```

---
## Build a model

We choose a simple 1D convnet starting with an `Embedding` layer.


```python
# A integer input for vocab indices.
inputs = keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
```

---
## Train the model


```python
epochs = 5

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
```

    
   1/625 [37m━━━━━━━━━━━━━━━━━━━━  1:41 163ms/step - accuracy: 0.4062 - loss: 0.6957

<div class="k-default-codeblock">
```

```
</div>
   2/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 104ms/step - accuracy: 0.4297 - loss: 0.6964

<div class="k-default-codeblock">
```

```
</div>
   3/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 104ms/step - accuracy: 0.4392 - loss: 0.6971

<div class="k-default-codeblock">
```

```
</div>
   4/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 103ms/step - accuracy: 0.4544 - loss: 0.6961

<div class="k-default-codeblock">
```

```
</div>
   5/625 [37m━━━━━━━━━━━━━━━━━━━━  1:03 103ms/step - accuracy: 0.4660 - loss: 0.6952

<div class="k-default-codeblock">
```

```
</div>
   6/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 103ms/step - accuracy: 0.4726 - loss: 0.6952

<div class="k-default-codeblock">
```

```
</div>
   7/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 104ms/step - accuracy: 0.4816 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
   8/625 [37m━━━━━━━━━━━━━━━━━━━━  1:03 103ms/step - accuracy: 0.4893 - loss: 0.6938

<div class="k-default-codeblock">
```

```
</div>
   9/625 [37m━━━━━━━━━━━━━━━━━━━━  1:03 103ms/step - accuracy: 0.4943 - loss: 0.6935

<div class="k-default-codeblock">
```

```
</div>
  10/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 106ms/step - accuracy: 0.4974 - loss: 0.6935

<div class="k-default-codeblock">
```

```
</div>
  11/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 106ms/step - accuracy: 0.4989 - loss: 0.6940

<div class="k-default-codeblock">
```

```
</div>
  12/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 106ms/step - accuracy: 0.5005 - loss: 0.6943

<div class="k-default-codeblock">
```

```
</div>
  13/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 106ms/step - accuracy: 0.5020 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
  14/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 106ms/step - accuracy: 0.5033 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
  15/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 107ms/step - accuracy: 0.5048 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  16/625 [37m━━━━━━━━━━━━━━━━━━━━  1:06 109ms/step - accuracy: 0.5058 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  17/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 108ms/step - accuracy: 0.5069 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  18/625 [37m━━━━━━━━━━━━━━━━━━━━  1:06 109ms/step - accuracy: 0.5079 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  19/625 [37m━━━━━━━━━━━━━━━━━━━━  1:06 109ms/step - accuracy: 0.5092 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  20/625 [37m━━━━━━━━━━━━━━━━━━━━  1:06 110ms/step - accuracy: 0.5099 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  21/625 [37m━━━━━━━━━━━━━━━━━━━━  1:06 110ms/step - accuracy: 0.5106 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  22/625 [37m━━━━━━━━━━━━━━━━━━━━  1:06 110ms/step - accuracy: 0.5111 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  23/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 110ms/step - accuracy: 0.5114 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  24/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 109ms/step - accuracy: 0.5118 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  25/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 109ms/step - accuracy: 0.5123 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  26/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 109ms/step - accuracy: 0.5127 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
  27/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 109ms/step - accuracy: 0.5128 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
  28/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 110ms/step - accuracy: 0.5128 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
  29/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 109ms/step - accuracy: 0.5128 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
  30/625 [37m━━━━━━━━━━━━━━━━━━━━  1:05 109ms/step - accuracy: 0.5128 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  31/625 [37m━━━━━━━━━━━━━━━━━━━━  1:04 109ms/step - accuracy: 0.5128 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  32/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 109ms/step - accuracy: 0.5128 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  33/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5128 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  34/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5128 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  35/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5129 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  36/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 109ms/step - accuracy: 0.5128 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  37/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5127 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
  38/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5125 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  39/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5124 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  40/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 110ms/step - accuracy: 0.5122 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  41/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5121 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  42/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5119 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  43/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5117 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  44/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5115 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  45/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5113 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  46/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5111 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
  47/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5110 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  48/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5108 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  49/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 111ms/step - accuracy: 0.5107 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  50/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 112ms/step - accuracy: 0.5105 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  51/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:03 111ms/step - accuracy: 0.5103 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  52/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 112ms/step - accuracy: 0.5102 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  53/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 113ms/step - accuracy: 0.5100 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  54/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 113ms/step - accuracy: 0.5099 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  55/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 113ms/step - accuracy: 0.5097 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  56/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 113ms/step - accuracy: 0.5096 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  57/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 113ms/step - accuracy: 0.5094 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  58/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 113ms/step - accuracy: 0.5093 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  59/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 114ms/step - accuracy: 0.5092 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  60/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 114ms/step - accuracy: 0.5091 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  61/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5090 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  62/625 ━[37m━━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5089 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  63/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5088 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  64/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5087 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  65/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5086 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  66/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5085 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  67/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 115ms/step - accuracy: 0.5084 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  68/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:04 115ms/step - accuracy: 0.5083 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  69/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 115ms/step - accuracy: 0.5083 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  70/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 115ms/step - accuracy: 0.5083 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  71/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 115ms/step - accuracy: 0.5083 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  72/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 115ms/step - accuracy: 0.5083 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  73/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5083 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  74/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5082 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  75/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5082 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  76/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5082 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  77/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5081 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  78/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5081 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  79/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5080 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  80/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 116ms/step - accuracy: 0.5080 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  81/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5080 - loss: 0.6950

<div class="k-default-codeblock">
```

```
</div>
  82/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5079 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  83/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5079 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  84/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5078 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  85/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5078 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  86/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5078 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  87/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5077 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  88/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 117ms/step - accuracy: 0.5077 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  89/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:03 118ms/step - accuracy: 0.5076 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  90/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:02 118ms/step - accuracy: 0.5076 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  91/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:02 117ms/step - accuracy: 0.5075 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  92/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:02 117ms/step - accuracy: 0.5075 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  93/625 ━━[37m━━━━━━━━━━━━━━━━━━  1:02 117ms/step - accuracy: 0.5075 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  94/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:02 117ms/step - accuracy: 0.5074 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  95/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:02 117ms/step - accuracy: 0.5074 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  96/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5074 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  97/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5074 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  98/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5073 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
  99/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5073 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 100/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5072 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 101/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5072 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 102/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5071 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 103/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5071 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 104/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:01 117ms/step - accuracy: 0.5070 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 105/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 117ms/step - accuracy: 0.5070 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 106/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 117ms/step - accuracy: 0.5069 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 107/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 117ms/step - accuracy: 0.5069 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 108/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 117ms/step - accuracy: 0.5068 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 109/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 117ms/step - accuracy: 0.5068 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 110/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 118ms/step - accuracy: 0.5067 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 111/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 118ms/step - accuracy: 0.5067 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 112/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 118ms/step - accuracy: 0.5066 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 113/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 118ms/step - accuracy: 0.5066 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 114/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 118ms/step - accuracy: 0.5065 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 115/625 ━━━[37m━━━━━━━━━━━━━━━━━  1:00 118ms/step - accuracy: 0.5064 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 116/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 118ms/step - accuracy: 0.5064 - loss: 0.6949 

<div class="k-default-codeblock">
```

```
</div>
 117/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 118ms/step - accuracy: 0.5063 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 118/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 118ms/step - accuracy: 0.5063 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 119/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 118ms/step - accuracy: 0.5062 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 120/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 118ms/step - accuracy: 0.5061 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 121/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5061 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 122/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5060 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 123/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5060 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 124/625 ━━━[37m━━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5059 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 125/625 ━━━━[37m━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5058 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 126/625 ━━━━[37m━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5058 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 127/625 ━━━━[37m━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5057 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 128/625 ━━━━[37m━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5056 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 129/625 ━━━━[37m━━━━━━━━━━━━━━━━  59s 119ms/step - accuracy: 0.5056 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 130/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 119ms/step - accuracy: 0.5055 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 131/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 119ms/step - accuracy: 0.5054 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 132/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 119ms/step - accuracy: 0.5054 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 133/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 120ms/step - accuracy: 0.5053 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 134/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 120ms/step - accuracy: 0.5053 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 135/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 120ms/step - accuracy: 0.5052 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 136/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 120ms/step - accuracy: 0.5051 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 137/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 119ms/step - accuracy: 0.5051 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 138/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 120ms/step - accuracy: 0.5050 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 139/625 ━━━━[37m━━━━━━━━━━━━━━━━  58s 120ms/step - accuracy: 0.5049 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 140/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5049 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 141/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5048 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 142/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5048 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 143/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5047 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 144/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5047 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 145/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5046 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 146/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5046 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 147/625 ━━━━[37m━━━━━━━━━━━━━━━━  57s 119ms/step - accuracy: 0.5046 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 148/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 119ms/step - accuracy: 0.5045 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 149/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 119ms/step - accuracy: 0.5045 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 150/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5044 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 151/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5044 - loss: 0.6949

<div class="k-default-codeblock">
```

```
</div>
 152/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 119ms/step - accuracy: 0.5043 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 153/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5043 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 154/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5042 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 155/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5042 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 156/625 ━━━━[37m━━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5042 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 157/625 ━━━━━[37m━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5041 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 158/625 ━━━━━[37m━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5041 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 159/625 ━━━━━[37m━━━━━━━━━━━━━━━  56s 120ms/step - accuracy: 0.5040 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 160/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5040 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 161/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5039 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 162/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5039 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 163/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5039 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 164/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5038 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 165/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5038 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 166/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 121ms/step - accuracy: 0.5038 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 167/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 121ms/step - accuracy: 0.5038 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 168/625 ━━━━━[37m━━━━━━━━━━━━━━━  55s 120ms/step - accuracy: 0.5037 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 169/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 120ms/step - accuracy: 0.5037 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 170/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5037 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 171/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5036 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 172/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5036 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 173/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5036 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 174/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5035 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 175/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5035 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 176/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5035 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 177/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5034 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 178/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 121ms/step - accuracy: 0.5034 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 179/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 122ms/step - accuracy: 0.5034 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 180/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 122ms/step - accuracy: 0.5033 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 181/625 ━━━━━[37m━━━━━━━━━━━━━━━  54s 122ms/step - accuracy: 0.5033 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 182/625 ━━━━━[37m━━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5032 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 183/625 ━━━━━[37m━━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5032 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 184/625 ━━━━━[37m━━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5032 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 185/625 ━━━━━[37m━━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5031 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 186/625 ━━━━━[37m━━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5031 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 187/625 ━━━━━[37m━━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5031 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 188/625 ━━━━━━[37m━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5030 - loss: 0.6948

<div class="k-default-codeblock">
```

```
</div>
 189/625 ━━━━━━[37m━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5030 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 190/625 ━━━━━━[37m━━━━━━━━━━━━━━  53s 122ms/step - accuracy: 0.5030 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 191/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5030 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 192/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5030 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 193/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5029 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 194/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5029 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 195/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5029 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 196/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5029 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 197/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5029 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 198/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5029 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 199/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 122ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 200/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 201/625 ━━━━━━[37m━━━━━━━━━━━━━━  52s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 202/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 203/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 204/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 205/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 206/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 123ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 207/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 208/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 209/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 210/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 211/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 212/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 213/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 214/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 215/625 ━━━━━━[37m━━━━━━━━━━━━━━  51s 124ms/step - accuracy: 0.5028 - loss: 0.6947

<div class="k-default-codeblock">
```

```
</div>
 216/625 ━━━━━━[37m━━━━━━━━━━━━━━  50s 124ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 217/625 ━━━━━━[37m━━━━━━━━━━━━━━  50s 124ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 218/625 ━━━━━━[37m━━━━━━━━━━━━━━  50s 124ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 219/625 ━━━━━━━[37m━━━━━━━━━━━━━  50s 124ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 220/625 ━━━━━━━[37m━━━━━━━━━━━━━  50s 124ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 221/625 ━━━━━━━[37m━━━━━━━━━━━━━  50s 124ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 222/625 ━━━━━━━[37m━━━━━━━━━━━━━  50s 125ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 223/625 ━━━━━━━[37m━━━━━━━━━━━━━  50s 125ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 224/625 ━━━━━━━[37m━━━━━━━━━━━━━  50s 125ms/step - accuracy: 0.5027 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 225/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 226/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 227/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 228/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 229/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 230/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 231/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 232/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 233/625 ━━━━━━━[37m━━━━━━━━━━━━━  49s 125ms/step - accuracy: 0.5028 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 234/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 235/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 236/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 237/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 238/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 239/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6946

<div class="k-default-codeblock">
```

```
</div>
 240/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 125ms/step - accuracy: 0.5029 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 241/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 242/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 243/625 ━━━━━━━[37m━━━━━━━━━━━━━  48s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 244/625 ━━━━━━━[37m━━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 245/625 ━━━━━━━[37m━━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 246/625 ━━━━━━━[37m━━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 247/625 ━━━━━━━[37m━━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5030 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 248/625 ━━━━━━━[37m━━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5031 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 249/625 ━━━━━━━[37m━━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5031 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 250/625 ━━━━━━━━[37m━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5031 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 251/625 ━━━━━━━━[37m━━━━━━━━━━━━  47s 126ms/step - accuracy: 0.5031 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 252/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5031 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 253/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5032 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 254/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5032 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 255/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5032 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 256/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5033 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 257/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5033 - loss: 0.6945

<div class="k-default-codeblock">
```

```
</div>
 258/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 126ms/step - accuracy: 0.5033 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 259/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 127ms/step - accuracy: 0.5034 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 260/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 127ms/step - accuracy: 0.5034 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 261/625 ━━━━━━━━[37m━━━━━━━━━━━━  46s 127ms/step - accuracy: 0.5035 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 262/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5035 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 263/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5036 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 264/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5036 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 265/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5037 - loss: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 266/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5037 - loss: 0.6943

<div class="k-default-codeblock">
```

```
</div>
 267/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5038 - loss: 0.6943

<div class="k-default-codeblock">
```

```
</div>
 268/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5039 - loss: 0.6943

<div class="k-default-codeblock">
```

```
</div>
 269/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5039 - loss: 0.6943

<div class="k-default-codeblock">
```

```
</div>
 270/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5040 - loss: 0.6943

<div class="k-default-codeblock">
```

```
</div>
 271/625 ━━━━━━━━[37m━━━━━━━━━━━━  45s 127ms/step - accuracy: 0.5041 - loss: 0.6942

<div class="k-default-codeblock">
```

```
</div>
 272/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5041 - loss: 0.6942

<div class="k-default-codeblock">
```

```
</div>
 273/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5042 - loss: 0.6942

<div class="k-default-codeblock">
```

```
</div>
 274/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5043 - loss: 0.6942

<div class="k-default-codeblock">
```

```
</div>
 275/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5043 - loss: 0.6942

<div class="k-default-codeblock">
```

```
</div>
 276/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5044 - loss: 0.6941

<div class="k-default-codeblock">
```

```
</div>
 277/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5045 - loss: 0.6941

<div class="k-default-codeblock">
```

```
</div>
 278/625 ━━━━━━━━[37m━━━━━━━━━━━━  44s 127ms/step - accuracy: 0.5046 - loss: 0.6941

<div class="k-default-codeblock">
```

```
</div>
 279/625 ━━━━━━━━[37m━━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5047 - loss: 0.6940

<div class="k-default-codeblock">
```

```
</div>
 280/625 ━━━━━━━━[37m━━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5048 - loss: 0.6940

<div class="k-default-codeblock">
```

```
</div>
 281/625 ━━━━━━━━[37m━━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5048 - loss: 0.6940

<div class="k-default-codeblock">
```

```
</div>
 282/625 ━━━━━━━━━[37m━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5049 - loss: 0.6939

<div class="k-default-codeblock">
```

```
</div>
 283/625 ━━━━━━━━━[37m━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5050 - loss: 0.6939

<div class="k-default-codeblock">
```

```
</div>
 284/625 ━━━━━━━━━[37m━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5051 - loss: 0.6938

<div class="k-default-codeblock">
```

```
</div>
 285/625 ━━━━━━━━━[37m━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5052 - loss: 0.6938

<div class="k-default-codeblock">
```

```
</div>
 286/625 ━━━━━━━━━[37m━━━━━━━━━━━  43s 127ms/step - accuracy: 0.5053 - loss: 0.6937

<div class="k-default-codeblock">
```

```
</div>
 287/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5054 - loss: 0.6937

<div class="k-default-codeblock">
```

```
</div>
 288/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5055 - loss: 0.6937

<div class="k-default-codeblock">
```

```
</div>
 289/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5056 - loss: 0.6936

<div class="k-default-codeblock">
```

```
</div>
 290/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5057 - loss: 0.6936

<div class="k-default-codeblock">
```

```
</div>
 291/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5058 - loss: 0.6935

<div class="k-default-codeblock">
```

```
</div>
 292/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5060 - loss: 0.6934

<div class="k-default-codeblock">
```

```
</div>
 293/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5061 - loss: 0.6934

<div class="k-default-codeblock">
```

```
</div>
 294/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5062 - loss: 0.6933

<div class="k-default-codeblock">
```

```
</div>
 295/625 ━━━━━━━━━[37m━━━━━━━━━━━  42s 127ms/step - accuracy: 0.5063 - loss: 0.6933

<div class="k-default-codeblock">
```

```
</div>
 296/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 127ms/step - accuracy: 0.5064 - loss: 0.6932

<div class="k-default-codeblock">
```

```
</div>
 297/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 127ms/step - accuracy: 0.5065 - loss: 0.6931

<div class="k-default-codeblock">
```

```
</div>
 298/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 127ms/step - accuracy: 0.5067 - loss: 0.6931

<div class="k-default-codeblock">
```

```
</div>
 299/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 127ms/step - accuracy: 0.5068 - loss: 0.6930

<div class="k-default-codeblock">
```

```
</div>
 300/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 127ms/step - accuracy: 0.5069 - loss: 0.6930

<div class="k-default-codeblock">
```

```
</div>
 301/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 127ms/step - accuracy: 0.5070 - loss: 0.6929

<div class="k-default-codeblock">
```

```
</div>
 302/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 128ms/step - accuracy: 0.5072 - loss: 0.6928

<div class="k-default-codeblock">
```

```
</div>
 303/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 128ms/step - accuracy: 0.5073 - loss: 0.6928

<div class="k-default-codeblock">
```

```
</div>
 304/625 ━━━━━━━━━[37m━━━━━━━━━━━  41s 128ms/step - accuracy: 0.5074 - loss: 0.6927

<div class="k-default-codeblock">
```

```
</div>
 305/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 128ms/step - accuracy: 0.5075 - loss: 0.6926

<div class="k-default-codeblock">
```

```
</div>
 306/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 128ms/step - accuracy: 0.5077 - loss: 0.6926

<div class="k-default-codeblock">
```

```
</div>
 307/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 128ms/step - accuracy: 0.5078 - loss: 0.6925

<div class="k-default-codeblock">
```

```
</div>
 308/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 128ms/step - accuracy: 0.5079 - loss: 0.6924

<div class="k-default-codeblock">
```

```
</div>
 309/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 129ms/step - accuracy: 0.5081 - loss: 0.6923

<div class="k-default-codeblock">
```

```
</div>
 310/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 129ms/step - accuracy: 0.5082 - loss: 0.6923

<div class="k-default-codeblock">
```

```
</div>
 311/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 129ms/step - accuracy: 0.5083 - loss: 0.6922

<div class="k-default-codeblock">
```

```
</div>
 312/625 ━━━━━━━━━[37m━━━━━━━━━━━  40s 129ms/step - accuracy: 0.5085 - loss: 0.6921

<div class="k-default-codeblock">
```

```
</div>
 313/625 ━━━━━━━━━━[37m━━━━━━━━━━  40s 129ms/step - accuracy: 0.5086 - loss: 0.6920

<div class="k-default-codeblock">
```

```
</div>
 314/625 ━━━━━━━━━━[37m━━━━━━━━━━  40s 129ms/step - accuracy: 0.5087 - loss: 0.6920

<div class="k-default-codeblock">
```

```
</div>
 315/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5089 - loss: 0.6919

<div class="k-default-codeblock">
```

```
</div>
 316/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5090 - loss: 0.6918

<div class="k-default-codeblock">
```

```
</div>
 317/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5092 - loss: 0.6917

<div class="k-default-codeblock">
```

```
</div>
 318/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5093 - loss: 0.6916

<div class="k-default-codeblock">
```

```
</div>
 319/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5095 - loss: 0.6916

<div class="k-default-codeblock">
```

```
</div>
 320/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5096 - loss: 0.6915

<div class="k-default-codeblock">
```

```
</div>
 321/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5098 - loss: 0.6914

<div class="k-default-codeblock">
```

```
</div>
 322/625 ━━━━━━━━━━[37m━━━━━━━━━━  39s 129ms/step - accuracy: 0.5099 - loss: 0.6913

<div class="k-default-codeblock">
```

```
</div>
 323/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5101 - loss: 0.6912

<div class="k-default-codeblock">
```

```
</div>
 324/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5102 - loss: 0.6911

<div class="k-default-codeblock">
```

```
</div>
 325/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5104 - loss: 0.6910

<div class="k-default-codeblock">
```

```
</div>
 326/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5105 - loss: 0.6909

<div class="k-default-codeblock">
```

```
</div>
 327/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5107 - loss: 0.6908

<div class="k-default-codeblock">
```

```
</div>
 328/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5109 - loss: 0.6907

<div class="k-default-codeblock">
```

```
</div>
 329/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5110 - loss: 0.6907

<div class="k-default-codeblock">
```

```
</div>
 330/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5112 - loss: 0.6906

<div class="k-default-codeblock">
```

```
</div>
 331/625 ━━━━━━━━━━[37m━━━━━━━━━━  38s 129ms/step - accuracy: 0.5113 - loss: 0.6905

<div class="k-default-codeblock">
```

```
</div>
 332/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5115 - loss: 0.6904

<div class="k-default-codeblock">
```

```
</div>
 333/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5117 - loss: 0.6903

<div class="k-default-codeblock">
```

```
</div>
 334/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5118 - loss: 0.6902

<div class="k-default-codeblock">
```

```
</div>
 335/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5120 - loss: 0.6901

<div class="k-default-codeblock">
```

```
</div>
 336/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5122 - loss: 0.6899

<div class="k-default-codeblock">
```

```
</div>
 337/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5123 - loss: 0.6898

<div class="k-default-codeblock">
```

```
</div>
 338/625 ━━━━━━━━━━[37m━━━━━━━━━━  37s 129ms/step - accuracy: 0.5125 - loss: 0.6897

<div class="k-default-codeblock">
```

```
</div>
 339/625 ━━━━━━━━━━[37m━━━━━━━━━━  36s 129ms/step - accuracy: 0.5127 - loss: 0.6896

<div class="k-default-codeblock">
```

```
</div>
 340/625 ━━━━━━━━━━[37m━━━━━━━━━━  36s 129ms/step - accuracy: 0.5128 - loss: 0.6895

<div class="k-default-codeblock">
```

```
</div>
 341/625 ━━━━━━━━━━[37m━━━━━━━━━━  36s 129ms/step - accuracy: 0.5130 - loss: 0.6894

<div class="k-default-codeblock">
```

```
</div>
 342/625 ━━━━━━━━━━[37m━━━━━━━━━━  36s 129ms/step - accuracy: 0.5132 - loss: 0.6893

<div class="k-default-codeblock">
```

```
</div>
 343/625 ━━━━━━━━━━[37m━━━━━━━━━━  36s 129ms/step - accuracy: 0.5133 - loss: 0.6892

<div class="k-default-codeblock">
```

```
</div>
 344/625 ━━━━━━━━━━━[37m━━━━━━━━━  36s 129ms/step - accuracy: 0.5135 - loss: 0.6891

<div class="k-default-codeblock">
```

```
</div>
 345/625 ━━━━━━━━━━━[37m━━━━━━━━━  36s 129ms/step - accuracy: 0.5137 - loss: 0.6889

<div class="k-default-codeblock">
```

```
</div>
 346/625 ━━━━━━━━━━━[37m━━━━━━━━━  36s 129ms/step - accuracy: 0.5139 - loss: 0.6888

<div class="k-default-codeblock">
```

```
</div>
 347/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 129ms/step - accuracy: 0.5140 - loss: 0.6887

<div class="k-default-codeblock">
```

```
</div>
 348/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 129ms/step - accuracy: 0.5142 - loss: 0.6886

<div class="k-default-codeblock">
```

```
</div>
 349/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 129ms/step - accuracy: 0.5144 - loss: 0.6885

<div class="k-default-codeblock">
```

```
</div>
 350/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5146 - loss: 0.6883

<div class="k-default-codeblock">
```

```
</div>
 351/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5148 - loss: 0.6882

<div class="k-default-codeblock">
```

```
</div>
 352/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5149 - loss: 0.6881

<div class="k-default-codeblock">
```

```
</div>
 353/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5151 - loss: 0.6880

<div class="k-default-codeblock">
```

```
</div>
 354/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5153 - loss: 0.6878

<div class="k-default-codeblock">
```

```
</div>
 355/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5155 - loss: 0.6877

<div class="k-default-codeblock">
```

```
</div>
 356/625 ━━━━━━━━━━━[37m━━━━━━━━━  35s 130ms/step - accuracy: 0.5157 - loss: 0.6876

<div class="k-default-codeblock">
```

```
</div>
 357/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 130ms/step - accuracy: 0.5159 - loss: 0.6874

<div class="k-default-codeblock">
```

```
</div>
 358/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 130ms/step - accuracy: 0.5161 - loss: 0.6873

<div class="k-default-codeblock">
```

```
</div>
 359/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5162 - loss: 0.6872

<div class="k-default-codeblock">
```

```
</div>
 360/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5164 - loss: 0.6870

<div class="k-default-codeblock">
```

```
</div>
 361/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5166 - loss: 0.6869

<div class="k-default-codeblock">
```

```
</div>
 362/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5168 - loss: 0.6868

<div class="k-default-codeblock">
```

```
</div>
 363/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5170 - loss: 0.6866

<div class="k-default-codeblock">
```

```
</div>
 364/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5172 - loss: 0.6865

<div class="k-default-codeblock">
```

```
</div>
 365/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 131ms/step - accuracy: 0.5174 - loss: 0.6863

<div class="k-default-codeblock">
```

```
</div>
 366/625 ━━━━━━━━━━━[37m━━━━━━━━━  34s 132ms/step - accuracy: 0.5176 - loss: 0.6862

<div class="k-default-codeblock">
```

```
</div>
 367/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5178 - loss: 0.6861

<div class="k-default-codeblock">
```

```
</div>
 368/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5180 - loss: 0.6859

<div class="k-default-codeblock">
```

```
</div>
 369/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5181 - loss: 0.6858

<div class="k-default-codeblock">
```

```
</div>
 370/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5183 - loss: 0.6856

<div class="k-default-codeblock">
```

```
</div>
 371/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5185 - loss: 0.6855

<div class="k-default-codeblock">
```

```
</div>
 372/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5187 - loss: 0.6854

<div class="k-default-codeblock">
```

```
</div>
 373/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5189 - loss: 0.6852

<div class="k-default-codeblock">
```

```
</div>
 374/625 ━━━━━━━━━━━[37m━━━━━━━━━  33s 132ms/step - accuracy: 0.5191 - loss: 0.6851

<div class="k-default-codeblock">
```

```
</div>
 375/625 ━━━━━━━━━━━━[37m━━━━━━━━  33s 132ms/step - accuracy: 0.5193 - loss: 0.6849

<div class="k-default-codeblock">
```

```
</div>
 376/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 132ms/step - accuracy: 0.5195 - loss: 0.6848

<div class="k-default-codeblock">
```

```
</div>
 377/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 132ms/step - accuracy: 0.5197 - loss: 0.6846

<div class="k-default-codeblock">
```

```
</div>
 378/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 132ms/step - accuracy: 0.5199 - loss: 0.6845

<div class="k-default-codeblock">
```

```
</div>
 379/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 132ms/step - accuracy: 0.5201 - loss: 0.6844

<div class="k-default-codeblock">
```

```
</div>
 380/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 132ms/step - accuracy: 0.5203 - loss: 0.6842

<div class="k-default-codeblock">
```

```
</div>
 381/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 133ms/step - accuracy: 0.5205 - loss: 0.6841

<div class="k-default-codeblock">
```

```
</div>
 382/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 133ms/step - accuracy: 0.5207 - loss: 0.6839

<div class="k-default-codeblock">
```

```
</div>
 383/625 ━━━━━━━━━━━━[37m━━━━━━━━  32s 133ms/step - accuracy: 0.5209 - loss: 0.6837

<div class="k-default-codeblock">
```

```
</div>
 384/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5211 - loss: 0.6836

<div class="k-default-codeblock">
```

```
</div>
 385/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5213 - loss: 0.6834

<div class="k-default-codeblock">
```

```
</div>
 386/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5215 - loss: 0.6833

<div class="k-default-codeblock">
```

```
</div>
 387/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5217 - loss: 0.6831

<div class="k-default-codeblock">
```

```
</div>
 388/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5220 - loss: 0.6830

<div class="k-default-codeblock">
```

```
</div>
 389/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5222 - loss: 0.6828

<div class="k-default-codeblock">
```

```
</div>
 390/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5224 - loss: 0.6827

<div class="k-default-codeblock">
```

```
</div>
 391/625 ━━━━━━━━━━━━[37m━━━━━━━━  31s 133ms/step - accuracy: 0.5226 - loss: 0.6825

<div class="k-default-codeblock">
```

```
</div>
 392/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5228 - loss: 0.6823

<div class="k-default-codeblock">
```

```
</div>
 393/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5230 - loss: 0.6822

<div class="k-default-codeblock">
```

```
</div>
 394/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5232 - loss: 0.6820

<div class="k-default-codeblock">
```

```
</div>
 395/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5234 - loss: 0.6819

<div class="k-default-codeblock">
```

```
</div>
 396/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5236 - loss: 0.6817

<div class="k-default-codeblock">
```

```
</div>
 397/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5238 - loss: 0.6815

<div class="k-default-codeblock">
```

```
</div>
 398/625 ━━━━━━━━━━━━[37m━━━━━━━━  30s 133ms/step - accuracy: 0.5240 - loss: 0.6814

<div class="k-default-codeblock">
```

```
</div>
 399/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 133ms/step - accuracy: 0.5243 - loss: 0.6812

<div class="k-default-codeblock">
```

```
</div>
 400/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 133ms/step - accuracy: 0.5245 - loss: 0.6810

<div class="k-default-codeblock">
```

```
</div>
 401/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 133ms/step - accuracy: 0.5247 - loss: 0.6809

<div class="k-default-codeblock">
```

```
</div>
 402/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 133ms/step - accuracy: 0.5249 - loss: 0.6807

<div class="k-default-codeblock">
```

```
</div>
 403/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 132ms/step - accuracy: 0.5251 - loss: 0.6805

<div class="k-default-codeblock">
```

```
</div>
 404/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 132ms/step - accuracy: 0.5253 - loss: 0.6803

<div class="k-default-codeblock">
```

```
</div>
 405/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 132ms/step - accuracy: 0.5255 - loss: 0.6802

<div class="k-default-codeblock">
```

```
</div>
 406/625 ━━━━━━━━━━━━[37m━━━━━━━━  29s 132ms/step - accuracy: 0.5257 - loss: 0.6800

<div class="k-default-codeblock">
```

```
</div>
 407/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5260 - loss: 0.6798

<div class="k-default-codeblock">
```

```
</div>
 408/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5262 - loss: 0.6797

<div class="k-default-codeblock">
```

```
</div>
 409/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5264 - loss: 0.6795

<div class="k-default-codeblock">
```

```
</div>
 410/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5266 - loss: 0.6793

<div class="k-default-codeblock">
```

```
</div>
 411/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5268 - loss: 0.6791

<div class="k-default-codeblock">
```

```
</div>
 412/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5270 - loss: 0.6790

<div class="k-default-codeblock">
```

```
</div>
 413/625 ━━━━━━━━━━━━━[37m━━━━━━━  28s 132ms/step - accuracy: 0.5272 - loss: 0.6788

<div class="k-default-codeblock">
```

```
</div>
 414/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5275 - loss: 0.6786

<div class="k-default-codeblock">
```

```
</div>
 415/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5277 - loss: 0.6785

<div class="k-default-codeblock">
```

```
</div>
 416/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5279 - loss: 0.6783

<div class="k-default-codeblock">
```

```
</div>
 417/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5281 - loss: 0.6781

<div class="k-default-codeblock">
```

```
</div>
 418/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5283 - loss: 0.6779

<div class="k-default-codeblock">
```

```
</div>
 419/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5285 - loss: 0.6778

<div class="k-default-codeblock">
```

```
</div>
 420/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5287 - loss: 0.6776

<div class="k-default-codeblock">
```

```
</div>
 421/625 ━━━━━━━━━━━━━[37m━━━━━━━  27s 132ms/step - accuracy: 0.5290 - loss: 0.6774

<div class="k-default-codeblock">
```

```
</div>
 422/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5292 - loss: 0.6772

<div class="k-default-codeblock">
```

```
</div>
 423/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5294 - loss: 0.6770

<div class="k-default-codeblock">
```

```
</div>
 424/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5296 - loss: 0.6769

<div class="k-default-codeblock">
```

```
</div>
 425/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5298 - loss: 0.6767

<div class="k-default-codeblock">
```

```
</div>
 426/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5301 - loss: 0.6765

<div class="k-default-codeblock">
```

```
</div>
 427/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5303 - loss: 0.6763

<div class="k-default-codeblock">
```

```
</div>
 428/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5305 - loss: 0.6762

<div class="k-default-codeblock">
```

```
</div>
 429/625 ━━━━━━━━━━━━━[37m━━━━━━━  26s 133ms/step - accuracy: 0.5307 - loss: 0.6760

<div class="k-default-codeblock">
```

```
</div>
 430/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5309 - loss: 0.6758

<div class="k-default-codeblock">
```

```
</div>
 431/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5311 - loss: 0.6756

<div class="k-default-codeblock">
```

```
</div>
 432/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5314 - loss: 0.6754

<div class="k-default-codeblock">
```

```
</div>
 433/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5316 - loss: 0.6753

<div class="k-default-codeblock">
```

```
</div>
 434/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5318 - loss: 0.6751

<div class="k-default-codeblock">
```

```
</div>
 435/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5320 - loss: 0.6749

<div class="k-default-codeblock">
```

```
</div>
 436/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5322 - loss: 0.6747

<div class="k-default-codeblock">
```

```
</div>
 437/625 ━━━━━━━━━━━━━[37m━━━━━━━  25s 133ms/step - accuracy: 0.5325 - loss: 0.6745

<div class="k-default-codeblock">
```

```
</div>
 438/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5327 - loss: 0.6743

<div class="k-default-codeblock">
```

```
</div>
 439/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5329 - loss: 0.6742

<div class="k-default-codeblock">
```

```
</div>
 440/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5331 - loss: 0.6740

<div class="k-default-codeblock">
```

```
</div>
 441/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5333 - loss: 0.6738

<div class="k-default-codeblock">
```

```
</div>
 442/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5336 - loss: 0.6736

<div class="k-default-codeblock">
```

```
</div>
 443/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5338 - loss: 0.6734

<div class="k-default-codeblock">
```

```
</div>
 444/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5340 - loss: 0.6732

<div class="k-default-codeblock">
```

```
</div>
 445/625 ━━━━━━━━━━━━━━[37m━━━━━━  24s 133ms/step - accuracy: 0.5342 - loss: 0.6731

<div class="k-default-codeblock">
```

```
</div>
 446/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 133ms/step - accuracy: 0.5344 - loss: 0.6729

<div class="k-default-codeblock">
```

```
</div>
 447/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 133ms/step - accuracy: 0.5347 - loss: 0.6727

<div class="k-default-codeblock">
```

```
</div>
 448/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 133ms/step - accuracy: 0.5349 - loss: 0.6725

<div class="k-default-codeblock">
```

```
</div>
 449/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 134ms/step - accuracy: 0.5351 - loss: 0.6723

<div class="k-default-codeblock">
```

```
</div>
 450/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 133ms/step - accuracy: 0.5353 - loss: 0.6722

<div class="k-default-codeblock">
```

```
</div>
 451/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 133ms/step - accuracy: 0.5355 - loss: 0.6720

<div class="k-default-codeblock">
```

```
</div>
 452/625 ━━━━━━━━━━━━━━[37m━━━━━━  23s 133ms/step - accuracy: 0.5358 - loss: 0.6718

<div class="k-default-codeblock">
```

```
</div>
 453/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 133ms/step - accuracy: 0.5360 - loss: 0.6716

<div class="k-default-codeblock">
```

```
</div>
 454/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 133ms/step - accuracy: 0.5362 - loss: 0.6714

<div class="k-default-codeblock">
```

```
</div>
 455/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 133ms/step - accuracy: 0.5364 - loss: 0.6713

<div class="k-default-codeblock">
```

```
</div>
 456/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 133ms/step - accuracy: 0.5366 - loss: 0.6711

<div class="k-default-codeblock">
```

```
</div>
 457/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 133ms/step - accuracy: 0.5369 - loss: 0.6709

<div class="k-default-codeblock">
```

```
</div>
 458/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 134ms/step - accuracy: 0.5371 - loss: 0.6707

<div class="k-default-codeblock">
```

```
</div>
 459/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 134ms/step - accuracy: 0.5373 - loss: 0.6705

<div class="k-default-codeblock">
```

```
</div>
 460/625 ━━━━━━━━━━━━━━[37m━━━━━━  22s 134ms/step - accuracy: 0.5375 - loss: 0.6703

<div class="k-default-codeblock">
```

```
</div>
 461/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5377 - loss: 0.6702

<div class="k-default-codeblock">
```

```
</div>
 462/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5380 - loss: 0.6700

<div class="k-default-codeblock">
```

```
</div>
 463/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5382 - loss: 0.6698

<div class="k-default-codeblock">
```

```
</div>
 464/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5384 - loss: 0.6696

<div class="k-default-codeblock">
```

```
</div>
 465/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5386 - loss: 0.6694

<div class="k-default-codeblock">
```

```
</div>
 466/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5388 - loss: 0.6692

<div class="k-default-codeblock">
```

```
</div>
 467/625 ━━━━━━━━━━━━━━[37m━━━━━━  21s 134ms/step - accuracy: 0.5391 - loss: 0.6691

<div class="k-default-codeblock">
```

```
</div>
 468/625 ━━━━━━━━━━━━━━[37m━━━━━━  20s 133ms/step - accuracy: 0.5393 - loss: 0.6689

<div class="k-default-codeblock">
```

```
</div>
 469/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 133ms/step - accuracy: 0.5395 - loss: 0.6687

<div class="k-default-codeblock">
```

```
</div>
 470/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 134ms/step - accuracy: 0.5397 - loss: 0.6685

<div class="k-default-codeblock">
```

```
</div>
 471/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 134ms/step - accuracy: 0.5399 - loss: 0.6683

<div class="k-default-codeblock">
```

```
</div>
 472/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 134ms/step - accuracy: 0.5401 - loss: 0.6681

<div class="k-default-codeblock">
```

```
</div>
 473/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 134ms/step - accuracy: 0.5404 - loss: 0.6679

<div class="k-default-codeblock">
```

```
</div>
 474/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 134ms/step - accuracy: 0.5406 - loss: 0.6678

<div class="k-default-codeblock">
```

```
</div>
 475/625 ━━━━━━━━━━━━━━━[37m━━━━━  20s 134ms/step - accuracy: 0.5408 - loss: 0.6676

<div class="k-default-codeblock">
```

```
</div>
 476/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5410 - loss: 0.6674

<div class="k-default-codeblock">
```

```
</div>
 477/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5412 - loss: 0.6672

<div class="k-default-codeblock">
```

```
</div>
 478/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5415 - loss: 0.6670

<div class="k-default-codeblock">
```

```
</div>
 479/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5417 - loss: 0.6668

<div class="k-default-codeblock">
```

```
</div>
 480/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5419 - loss: 0.6666

<div class="k-default-codeblock">
```

```
</div>
 481/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5421 - loss: 0.6664

<div class="k-default-codeblock">
```

```
</div>
 482/625 ━━━━━━━━━━━━━━━[37m━━━━━  19s 134ms/step - accuracy: 0.5423 - loss: 0.6662

<div class="k-default-codeblock">
```

```
</div>
 483/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5426 - loss: 0.6661

<div class="k-default-codeblock">
```

```
</div>
 484/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5428 - loss: 0.6659

<div class="k-default-codeblock">
```

```
</div>
 485/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5430 - loss: 0.6657

<div class="k-default-codeblock">
```

```
</div>
 486/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5432 - loss: 0.6655

<div class="k-default-codeblock">
```

```
</div>
 487/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5434 - loss: 0.6653

<div class="k-default-codeblock">
```

```
</div>
 488/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5437 - loss: 0.6651

<div class="k-default-codeblock">
```

```
</div>
 489/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5439 - loss: 0.6649

<div class="k-default-codeblock">
```

```
</div>
 490/625 ━━━━━━━━━━━━━━━[37m━━━━━  18s 134ms/step - accuracy: 0.5441 - loss: 0.6647

<div class="k-default-codeblock">
```

```
</div>
 491/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5443 - loss: 0.6645

<div class="k-default-codeblock">
```

```
</div>
 492/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5445 - loss: 0.6643

<div class="k-default-codeblock">
```

```
</div>
 493/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5448 - loss: 0.6642

<div class="k-default-codeblock">
```

```
</div>
 494/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5450 - loss: 0.6640

<div class="k-default-codeblock">
```

```
</div>
 495/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5452 - loss: 0.6638

<div class="k-default-codeblock">
```

```
</div>
 496/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5454 - loss: 0.6636

<div class="k-default-codeblock">
```

```
</div>
 497/625 ━━━━━━━━━━━━━━━[37m━━━━━  17s 134ms/step - accuracy: 0.5456 - loss: 0.6634

<div class="k-default-codeblock">
```

```
</div>
 498/625 ━━━━━━━━━━━━━━━[37m━━━━━  16s 134ms/step - accuracy: 0.5458 - loss: 0.6632

<div class="k-default-codeblock">
```

```
</div>
 499/625 ━━━━━━━━━━━━━━━[37m━━━━━  16s 134ms/step - accuracy: 0.5461 - loss: 0.6630

<div class="k-default-codeblock">
```

```
</div>
 500/625 ━━━━━━━━━━━━━━━━[37m━━━━  16s 134ms/step - accuracy: 0.5463 - loss: 0.6628

<div class="k-default-codeblock">
```

```
</div>
 501/625 ━━━━━━━━━━━━━━━━[37m━━━━  16s 134ms/step - accuracy: 0.5465 - loss: 0.6626

<div class="k-default-codeblock">
```

```
</div>
 502/625 ━━━━━━━━━━━━━━━━[37m━━━━  16s 134ms/step - accuracy: 0.5467 - loss: 0.6624

<div class="k-default-codeblock">
```

```
</div>
 503/625 ━━━━━━━━━━━━━━━━[37m━━━━  16s 134ms/step - accuracy: 0.5469 - loss: 0.6622

<div class="k-default-codeblock">
```

```
</div>
 504/625 ━━━━━━━━━━━━━━━━[37m━━━━  16s 134ms/step - accuracy: 0.5472 - loss: 0.6621

<div class="k-default-codeblock">
```

```
</div>
 505/625 ━━━━━━━━━━━━━━━━[37m━━━━  16s 134ms/step - accuracy: 0.5474 - loss: 0.6619

<div class="k-default-codeblock">
```

```
</div>
 506/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5476 - loss: 0.6617

<div class="k-default-codeblock">
```

```
</div>
 507/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5478 - loss: 0.6615

<div class="k-default-codeblock">
```

```
</div>
 508/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5480 - loss: 0.6613

<div class="k-default-codeblock">
```

```
</div>
 509/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5483 - loss: 0.6611

<div class="k-default-codeblock">
```

```
</div>
 510/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5485 - loss: 0.6609

<div class="k-default-codeblock">
```

```
</div>
 511/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5487 - loss: 0.6607

<div class="k-default-codeblock">
```

```
</div>
 512/625 ━━━━━━━━━━━━━━━━[37m━━━━  15s 134ms/step - accuracy: 0.5489 - loss: 0.6605

<div class="k-default-codeblock">
```

```
</div>
 513/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5491 - loss: 0.6603

<div class="k-default-codeblock">
```

```
</div>
 514/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5493 - loss: 0.6601

<div class="k-default-codeblock">
```

```
</div>
 515/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5496 - loss: 0.6599

<div class="k-default-codeblock">
```

```
</div>
 516/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5498 - loss: 0.6598

<div class="k-default-codeblock">
```

```
</div>
 517/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5500 - loss: 0.6596

<div class="k-default-codeblock">
```

```
</div>
 518/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5502 - loss: 0.6594

<div class="k-default-codeblock">
```

```
</div>
 519/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5504 - loss: 0.6592

<div class="k-default-codeblock">
```

```
</div>
 520/625 ━━━━━━━━━━━━━━━━[37m━━━━  14s 134ms/step - accuracy: 0.5506 - loss: 0.6590

<div class="k-default-codeblock">
```

```
</div>
 521/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5508 - loss: 0.6588

<div class="k-default-codeblock">
```

```
</div>
 522/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5511 - loss: 0.6586

<div class="k-default-codeblock">
```

```
</div>
 523/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5513 - loss: 0.6584

<div class="k-default-codeblock">
```

```
</div>
 524/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5515 - loss: 0.6582

<div class="k-default-codeblock">
```

```
</div>
 525/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5517 - loss: 0.6580

<div class="k-default-codeblock">
```

```
</div>
 526/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5519 - loss: 0.6578

<div class="k-default-codeblock">
```

```
</div>
 527/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5521 - loss: 0.6577

<div class="k-default-codeblock">
```

```
</div>
 528/625 ━━━━━━━━━━━━━━━━[37m━━━━  13s 134ms/step - accuracy: 0.5523 - loss: 0.6575

<div class="k-default-codeblock">
```

```
</div>
 529/625 ━━━━━━━━━━━━━━━━[37m━━━━  12s 134ms/step - accuracy: 0.5526 - loss: 0.6573

<div class="k-default-codeblock">
```

```
</div>
 530/625 ━━━━━━━━━━━━━━━━[37m━━━━  12s 134ms/step - accuracy: 0.5528 - loss: 0.6571

<div class="k-default-codeblock">
```

```
</div>
 531/625 ━━━━━━━━━━━━━━━━[37m━━━━  12s 134ms/step - accuracy: 0.5530 - loss: 0.6569

<div class="k-default-codeblock">
```

```
</div>
 532/625 ━━━━━━━━━━━━━━━━━[37m━━━  12s 134ms/step - accuracy: 0.5532 - loss: 0.6567

<div class="k-default-codeblock">
```

```
</div>
 533/625 ━━━━━━━━━━━━━━━━━[37m━━━  12s 134ms/step - accuracy: 0.5534 - loss: 0.6565

<div class="k-default-codeblock">
```

```
</div>
 534/625 ━━━━━━━━━━━━━━━━━[37m━━━  12s 134ms/step - accuracy: 0.5536 - loss: 0.6563

<div class="k-default-codeblock">
```

```
</div>
 535/625 ━━━━━━━━━━━━━━━━━[37m━━━  12s 134ms/step - accuracy: 0.5538 - loss: 0.6561

<div class="k-default-codeblock">
```

```
</div>
 536/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5540 - loss: 0.6559

<div class="k-default-codeblock">
```

```
</div>
 537/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5543 - loss: 0.6557

<div class="k-default-codeblock">
```

```
</div>
 538/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5545 - loss: 0.6555

<div class="k-default-codeblock">
```

```
</div>
 539/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5547 - loss: 0.6554

<div class="k-default-codeblock">
```

```
</div>
 540/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5549 - loss: 0.6552

<div class="k-default-codeblock">
```

```
</div>
 541/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5551 - loss: 0.6550

<div class="k-default-codeblock">
```

```
</div>
 542/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5553 - loss: 0.6548

<div class="k-default-codeblock">
```

```
</div>
 543/625 ━━━━━━━━━━━━━━━━━[37m━━━  11s 134ms/step - accuracy: 0.5555 - loss: 0.6546

<div class="k-default-codeblock">
```

```
</div>
 544/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 134ms/step - accuracy: 0.5558 - loss: 0.6544

<div class="k-default-codeblock">
```

```
</div>
 545/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 135ms/step - accuracy: 0.5560 - loss: 0.6542

<div class="k-default-codeblock">
```

```
</div>
 546/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 135ms/step - accuracy: 0.5562 - loss: 0.6540

<div class="k-default-codeblock">
```

```
</div>
 547/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 135ms/step - accuracy: 0.5564 - loss: 0.6538

<div class="k-default-codeblock">
```

```
</div>
 548/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 135ms/step - accuracy: 0.5566 - loss: 0.6536

<div class="k-default-codeblock">
```

```
</div>
 549/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 135ms/step - accuracy: 0.5568 - loss: 0.6534

<div class="k-default-codeblock">
```

```
</div>
 550/625 ━━━━━━━━━━━━━━━━━[37m━━━  10s 135ms/step - accuracy: 0.5570 - loss: 0.6532

<div class="k-default-codeblock">
```

```
</div>
 551/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5572 - loss: 0.6531 

<div class="k-default-codeblock">
```

```
</div>
 552/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5574 - loss: 0.6529

<div class="k-default-codeblock">
```

```
</div>
 553/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5577 - loss: 0.6527

<div class="k-default-codeblock">
```

```
</div>
 554/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5579 - loss: 0.6525

<div class="k-default-codeblock">
```

```
</div>
 555/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5581 - loss: 0.6523

<div class="k-default-codeblock">
```

```
</div>
 556/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5583 - loss: 0.6521

<div class="k-default-codeblock">
```

```
</div>
 557/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5585 - loss: 0.6519

<div class="k-default-codeblock">
```

```
</div>
 558/625 ━━━━━━━━━━━━━━━━━[37m━━━  9s 135ms/step - accuracy: 0.5587 - loss: 0.6517

<div class="k-default-codeblock">
```

```
</div>
 559/625 ━━━━━━━━━━━━━━━━━[37m━━━  8s 135ms/step - accuracy: 0.5589 - loss: 0.6515

<div class="k-default-codeblock">
```

```
</div>
 560/625 ━━━━━━━━━━━━━━━━━[37m━━━  8s 135ms/step - accuracy: 0.5591 - loss: 0.6513

<div class="k-default-codeblock">
```

```
</div>
 561/625 ━━━━━━━━━━━━━━━━━[37m━━━  8s 135ms/step - accuracy: 0.5593 - loss: 0.6511

<div class="k-default-codeblock">
```

```
</div>
 562/625 ━━━━━━━━━━━━━━━━━[37m━━━  8s 135ms/step - accuracy: 0.5595 - loss: 0.6510

<div class="k-default-codeblock">
```

```
</div>
 563/625 ━━━━━━━━━━━━━━━━━━[37m━━  8s 135ms/step - accuracy: 0.5598 - loss: 0.6508

<div class="k-default-codeblock">
```

```
</div>
 564/625 ━━━━━━━━━━━━━━━━━━[37m━━  8s 135ms/step - accuracy: 0.5600 - loss: 0.6506

<div class="k-default-codeblock">
```

```
</div>
 565/625 ━━━━━━━━━━━━━━━━━━[37m━━  8s 135ms/step - accuracy: 0.5602 - loss: 0.6504

<div class="k-default-codeblock">
```

```
</div>
 566/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5604 - loss: 0.6502

<div class="k-default-codeblock">
```

```
</div>
 567/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5606 - loss: 0.6500

<div class="k-default-codeblock">
```

```
</div>
 568/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5608 - loss: 0.6498

<div class="k-default-codeblock">
```

```
</div>
 569/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5610 - loss: 0.6496

<div class="k-default-codeblock">
```

```
</div>
 570/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5612 - loss: 0.6494

<div class="k-default-codeblock">
```

```
</div>
 571/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5614 - loss: 0.6493

<div class="k-default-codeblock">
```

```
</div>
 572/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5616 - loss: 0.6491

<div class="k-default-codeblock">
```

```
</div>
 573/625 ━━━━━━━━━━━━━━━━━━[37m━━  7s 135ms/step - accuracy: 0.5618 - loss: 0.6489

<div class="k-default-codeblock">
```

```
</div>
 574/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5620 - loss: 0.6487

<div class="k-default-codeblock">
```

```
</div>
 575/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5623 - loss: 0.6485

<div class="k-default-codeblock">
```

```
</div>
 576/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5625 - loss: 0.6483

<div class="k-default-codeblock">
```

```
</div>
 577/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5627 - loss: 0.6481

<div class="k-default-codeblock">
```

```
</div>
 578/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5629 - loss: 0.6479

<div class="k-default-codeblock">
```

```
</div>
 579/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5631 - loss: 0.6477

<div class="k-default-codeblock">
```

```
</div>
 580/625 ━━━━━━━━━━━━━━━━━━[37m━━  6s 135ms/step - accuracy: 0.5633 - loss: 0.6476

<div class="k-default-codeblock">
```

```
</div>
 581/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5635 - loss: 0.6474

<div class="k-default-codeblock">
```

```
</div>
 582/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5637 - loss: 0.6472

<div class="k-default-codeblock">
```

```
</div>
 583/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5639 - loss: 0.6470

<div class="k-default-codeblock">
```

```
</div>
 584/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5641 - loss: 0.6468

<div class="k-default-codeblock">
```

```
</div>
 585/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5643 - loss: 0.6466

<div class="k-default-codeblock">
```

```
</div>
 586/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5645 - loss: 0.6464

<div class="k-default-codeblock">
```

```
</div>
 587/625 ━━━━━━━━━━━━━━━━━━[37m━━  5s 135ms/step - accuracy: 0.5647 - loss: 0.6462

<div class="k-default-codeblock">
```

```
</div>
 588/625 ━━━━━━━━━━━━━━━━━━[37m━━  4s 135ms/step - accuracy: 0.5649 - loss: 0.6461

<div class="k-default-codeblock">
```

```
</div>
 589/625 ━━━━━━━━━━━━━━━━━━[37m━━  4s 135ms/step - accuracy: 0.5651 - loss: 0.6459

<div class="k-default-codeblock">
```

```
</div>
 590/625 ━━━━━━━━━━━━━━━━━━[37m━━  4s 135ms/step - accuracy: 0.5653 - loss: 0.6457

<div class="k-default-codeblock">
```

```
</div>
 591/625 ━━━━━━━━━━━━━━━━━━[37m━━  4s 135ms/step - accuracy: 0.5655 - loss: 0.6455

<div class="k-default-codeblock">
```

```
</div>
 592/625 ━━━━━━━━━━━━━━━━━━[37m━━  4s 135ms/step - accuracy: 0.5657 - loss: 0.6453

<div class="k-default-codeblock">
```

```
</div>
 593/625 ━━━━━━━━━━━━━━━━━━[37m━━  4s 135ms/step - accuracy: 0.5659 - loss: 0.6451

<div class="k-default-codeblock">
```

```
</div>
 594/625 ━━━━━━━━━━━━━━━━━━━[37m━  4s 135ms/step - accuracy: 0.5661 - loss: 0.6449

<div class="k-default-codeblock">
```

```
</div>
 595/625 ━━━━━━━━━━━━━━━━━━━[37m━  4s 135ms/step - accuracy: 0.5664 - loss: 0.6447

<div class="k-default-codeblock">
```

```
</div>
 596/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 135ms/step - accuracy: 0.5666 - loss: 0.6446

<div class="k-default-codeblock">
```

```
</div>
 597/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 136ms/step - accuracy: 0.5668 - loss: 0.6444

<div class="k-default-codeblock">
```

```
</div>
 598/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 136ms/step - accuracy: 0.5670 - loss: 0.6442

<div class="k-default-codeblock">
```

```
</div>
 599/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 136ms/step - accuracy: 0.5672 - loss: 0.6440

<div class="k-default-codeblock">
```

```
</div>
 600/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 136ms/step - accuracy: 0.5674 - loss: 0.6438

<div class="k-default-codeblock">
```

```
</div>
 601/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 136ms/step - accuracy: 0.5676 - loss: 0.6436

<div class="k-default-codeblock">
```

```
</div>
 602/625 ━━━━━━━━━━━━━━━━━━━[37m━  3s 136ms/step - accuracy: 0.5678 - loss: 0.6434

<div class="k-default-codeblock">
```

```
</div>
 603/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5680 - loss: 0.6433

<div class="k-default-codeblock">
```

```
</div>
 604/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5682 - loss: 0.6431

<div class="k-default-codeblock">
```

```
</div>
 605/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5684 - loss: 0.6429

<div class="k-default-codeblock">
```

```
</div>
 606/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5686 - loss: 0.6427

<div class="k-default-codeblock">
```

```
</div>
 607/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5688 - loss: 0.6425

<div class="k-default-codeblock">
```

```
</div>
 608/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5690 - loss: 0.6423

<div class="k-default-codeblock">
```

```
</div>
 609/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5692 - loss: 0.6422

<div class="k-default-codeblock">
```

```
</div>
 610/625 ━━━━━━━━━━━━━━━━━━━[37m━  2s 136ms/step - accuracy: 0.5694 - loss: 0.6420

<div class="k-default-codeblock">
```

```
</div>
 611/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5696 - loss: 0.6418

<div class="k-default-codeblock">
```

```
</div>
 612/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5698 - loss: 0.6416

<div class="k-default-codeblock">
```

```
</div>
 613/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5700 - loss: 0.6414

<div class="k-default-codeblock">
```

```
</div>
 614/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5702 - loss: 0.6412

<div class="k-default-codeblock">
```

```
</div>
 615/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5704 - loss: 0.6410

<div class="k-default-codeblock">
```

```
</div>
 616/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5706 - loss: 0.6409

<div class="k-default-codeblock">
```

```
</div>
 617/625 ━━━━━━━━━━━━━━━━━━━[37m━  1s 136ms/step - accuracy: 0.5708 - loss: 0.6407

<div class="k-default-codeblock">
```

```
</div>
 618/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5710 - loss: 0.6405

<div class="k-default-codeblock">
```

```
</div>
 619/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5712 - loss: 0.6403

<div class="k-default-codeblock">
```

```
</div>
 620/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5714 - loss: 0.6401

<div class="k-default-codeblock">
```

```
</div>
 621/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5716 - loss: 0.6399

<div class="k-default-codeblock">
```

```
</div>
 622/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5718 - loss: 0.6398

<div class="k-default-codeblock">
```

```
</div>
 623/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5720 - loss: 0.6396

<div class="k-default-codeblock">
```

```
</div>
 624/625 ━━━━━━━━━━━━━━━━━━━[37m━  0s 136ms/step - accuracy: 0.5722 - loss: 0.6394

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 0s 136ms/step - accuracy: 0.5724 - loss: 0.6392

<div class="k-default-codeblock">
```

```
</div>
 625/625 ━━━━━━━━━━━━━━━━━━━━ 91s 146ms/step - accuracy: 0.5726 - loss: 0.6390 - val_accuracy: 0.8676 - val_loss: 0.3140





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7ca6d2db3dd0>

```
</div>
---
## Evaluate the model on the test set


```python
model.evaluate(test_ds)
```

    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  1:31 117ms/step - accuracy: 0.9062 - loss: 0.3036

<div class="k-default-codeblock">
```

```
</div>
   2/782 [37m━━━━━━━━━━━━━━━━━━━━  43s 56ms/step - accuracy: 0.9141 - loss: 0.2958  

<div class="k-default-codeblock">
```

```
</div>
   4/782 [37m━━━━━━━━━━━━━━━━━━━━  36s 47ms/step - accuracy: 0.9121 - loss: 0.2907

<div class="k-default-codeblock">
```

```
</div>
   5/782 [37m━━━━━━━━━━━━━━━━━━━━  37s 48ms/step - accuracy: 0.9084 - loss: 0.2898

<div class="k-default-codeblock">
```

```
</div>
   7/782 [37m━━━━━━━━━━━━━━━━━━━━  34s 44ms/step - accuracy: 0.9004 - loss: 0.2933

<div class="k-default-codeblock">
```

```
</div>
   9/782 [37m━━━━━━━━━━━━━━━━━━━━  34s 44ms/step - accuracy: 0.8899 - loss: 0.3037

<div class="k-default-codeblock">
```

```
</div>
  11/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8810 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
  12/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 43ms/step - accuracy: 0.8777 - loss: 0.3168

<div class="k-default-codeblock">
```

```
</div>
  14/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8723 - loss: 0.3211

<div class="k-default-codeblock">
```

```
</div>
  15/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 43ms/step - accuracy: 0.8701 - loss: 0.3223

<div class="k-default-codeblock">
```

```
</div>
  17/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 43ms/step - accuracy: 0.8674 - loss: 0.3234

<div class="k-default-codeblock">
```

```
</div>
  18/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 44ms/step - accuracy: 0.8666 - loss: 0.3236

<div class="k-default-codeblock">
```

```
</div>
  19/782 [37m━━━━━━━━━━━━━━━━━━━━  34s 45ms/step - accuracy: 0.8663 - loss: 0.3234

<div class="k-default-codeblock">
```

```
</div>
  21/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 44ms/step - accuracy: 0.8660 - loss: 0.3225

<div class="k-default-codeblock">
```

```
</div>
  23/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8661 - loss: 0.3216

<div class="k-default-codeblock">
```

```
</div>
  25/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8658 - loss: 0.3214

<div class="k-default-codeblock">
```

```
</div>
  27/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8651 - loss: 0.3220

<div class="k-default-codeblock">
```

```
</div>
  29/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8649 - loss: 0.3220

<div class="k-default-codeblock">
```

```
</div>
  30/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 44ms/step - accuracy: 0.8648 - loss: 0.3219

<div class="k-default-codeblock">
```

```
</div>
  32/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8646 - loss: 0.3220

<div class="k-default-codeblock">
```

```
</div>
  34/782 [37m━━━━━━━━━━━━━━━━━━━━  31s 42ms/step - accuracy: 0.8645 - loss: 0.3220

<div class="k-default-codeblock">
```

```
</div>
  36/782 [37m━━━━━━━━━━━━━━━━━━━━  31s 42ms/step - accuracy: 0.8644 - loss: 0.3218

<div class="k-default-codeblock">
```

```
</div>
  38/782 [37m━━━━━━━━━━━━━━━━━━━━  31s 42ms/step - accuracy: 0.8643 - loss: 0.3216

<div class="k-default-codeblock">
```

```
</div>
  40/782 ━[37m━━━━━━━━━━━━━━━━━━━  30s 41ms/step - accuracy: 0.8643 - loss: 0.3215

<div class="k-default-codeblock">
```

```
</div>
  41/782 ━[37m━━━━━━━━━━━━━━━━━━━  30s 42ms/step - accuracy: 0.8643 - loss: 0.3214

<div class="k-default-codeblock">
```

```
</div>
  43/782 ━[37m━━━━━━━━━━━━━━━━━━━  30s 41ms/step - accuracy: 0.8644 - loss: 0.3209

<div class="k-default-codeblock">
```

```
</div>
  45/782 ━[37m━━━━━━━━━━━━━━━━━━━  30s 41ms/step - accuracy: 0.8646 - loss: 0.3204

<div class="k-default-codeblock">
```

```
</div>
  47/782 ━[37m━━━━━━━━━━━━━━━━━━━  30s 41ms/step - accuracy: 0.8648 - loss: 0.3199

<div class="k-default-codeblock">
```

```
</div>
  49/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 41ms/step - accuracy: 0.8651 - loss: 0.3194

<div class="k-default-codeblock">
```

```
</div>
  51/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8654 - loss: 0.3189

<div class="k-default-codeblock">
```

```
</div>
  53/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8657 - loss: 0.3185

<div class="k-default-codeblock">
```

```
</div>
  55/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8660 - loss: 0.3181

<div class="k-default-codeblock">
```

```
</div>
  57/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8664 - loss: 0.3176

<div class="k-default-codeblock">
```

```
</div>
  59/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8668 - loss: 0.3172

<div class="k-default-codeblock">
```

```
</div>
  61/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8670 - loss: 0.3170

<div class="k-default-codeblock">
```

```
</div>
  63/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8671 - loss: 0.3168

<div class="k-default-codeblock">
```

```
</div>
  65/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8673 - loss: 0.3166

<div class="k-default-codeblock">
```

```
</div>
  66/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8673 - loss: 0.3165

<div class="k-default-codeblock">
```

```
</div>
  67/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8674 - loss: 0.3164

<div class="k-default-codeblock">
```

```
</div>
  69/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8675 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
  71/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8677 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
  73/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8678 - loss: 0.3158

<div class="k-default-codeblock">
```

```
</div>
  75/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 40ms/step - accuracy: 0.8679 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
  77/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 40ms/step - accuracy: 0.8679 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
  79/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8680 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
  81/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8682 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  83/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8682 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
  85/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8683 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
  87/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8683 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
  89/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8683 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  91/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  93/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  95/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  97/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
  99/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 101/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 103/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 105/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 106/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 108/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 110/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
 112/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
 114/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 116/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 118/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 120/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 122/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 124/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 126/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 128/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8684 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 130/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8685 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 132/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8685 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 134/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8685 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 136/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8685 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 138/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8685 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 140/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8686 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 141/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8686 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
 143/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8686 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
 145/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8686 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 147/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8687 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 149/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8687 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 151/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8687 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
 153/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 38ms/step - accuracy: 0.8687 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
 155/782 ━━━[37m━━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8687 - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 157/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8687 - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 159/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8687 - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 161/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8687 - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 163/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8687 - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 165/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8688 - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 167/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8688 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 169/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8688 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 171/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 38ms/step - accuracy: 0.8688 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 172/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8688 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 173/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8688 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 175/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 177/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 179/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 181/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 183/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 185/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 187/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 188/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 190/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 192/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 194/782 ━━━━[37m━━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 196/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8689 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 198/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8689 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 199/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8689 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 200/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8689 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 202/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8689 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 204/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 206/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 208/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 210/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 212/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 214/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 216/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 218/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 220/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 39ms/step - accuracy: 0.8690 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 222/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 224/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 226/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 227/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 229/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 231/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 232/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 234/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 235/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 237/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 239/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 241/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 243/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 245/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 247/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 249/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 251/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 253/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 255/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 257/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 259/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8690 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 261/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8689 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 263/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8689 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 265/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 39ms/step - accuracy: 0.8689 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 267/782 ━━━━━━[37m━━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 269/782 ━━━━━━[37m━━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 271/782 ━━━━━━[37m━━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 273/782 ━━━━━━[37m━━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 275/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 277/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 279/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 280/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 282/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 284/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 286/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 39ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 288/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 38ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 290/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 292/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 39ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 294/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 39ms/step - accuracy: 0.8689 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 296/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 298/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 299/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 39ms/step - accuracy: 0.8689 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 301/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 303/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 305/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 307/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 309/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 311/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 313/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 38ms/step - accuracy: 0.8689 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 315/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8689 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 317/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8689 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 318/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8689 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 320/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 39ms/step - accuracy: 0.8689 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 322/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8689 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 324/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 39ms/step - accuracy: 0.8689 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 326/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 39ms/step - accuracy: 0.8689 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 328/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8690 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 330/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8690 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 332/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8690 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 334/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8690 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 336/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8690 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 338/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 38ms/step - accuracy: 0.8690 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 340/782 ━━━━━━━━[37m━━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 342/782 ━━━━━━━━[37m━━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 344/782 ━━━━━━━━[37m━━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 346/782 ━━━━━━━━[37m━━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 348/782 ━━━━━━━━[37m━━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 350/782 ━━━━━━━━[37m━━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 352/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 354/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 356/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 358/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 360/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 361/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 38ms/step - accuracy: 0.8690 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 363/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 364/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 366/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 368/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 370/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 372/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 373/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8690 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 375/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8690 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 377/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8690 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 379/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 380/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 381/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 382/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 383/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 385/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 387/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 388/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 390/782 ━━━━━━━━━[37m━━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 392/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 394/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 396/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 398/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 400/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 402/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 403/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 405/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 407/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 409/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 411/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 413/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 415/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 417/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 419/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 420/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 422/782 ━━━━━━━━━━[37m━━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 424/782 ━━━━━━━━━━[37m━━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 426/782 ━━━━━━━━━━[37m━━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 428/782 ━━━━━━━━━━[37m━━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 430/782 ━━━━━━━━━━[37m━━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 432/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 434/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 436/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 438/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 439/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 441/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 443/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 444/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 445/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 446/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 448/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 450/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 452/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8691 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 454/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 456/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 458/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 460/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 462/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 464/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 466/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 468/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 470/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 472/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 473/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 475/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 477/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 479/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 481/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 483/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 485/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8690 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 487/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 489/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 491/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 493/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 494/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 496/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 498/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 500/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 502/782 ━━━━━━━━━━━━[37m━━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 504/782 ━━━━━━━━━━━━[37m━━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 506/782 ━━━━━━━━━━━━[37m━━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 508/782 ━━━━━━━━━━━━[37m━━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 510/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 512/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 513/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 515/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 517/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 519/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 520/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 521/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 522/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 524/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 526/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 527/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 529/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8689 - loss: 0.3122 

<div class="k-default-codeblock">
```

```
</div>
 531/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 533/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 535/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 537/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 539/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8689 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 541/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 543/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 544/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 546/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 548/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 550/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 552/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 554/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 556/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 558/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 560/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 561/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 562/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 564/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 565/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 566/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 568/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 570/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 572/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 573/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 575/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 577/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 579/782 ━━━━━━━━━━━━━━[37m━━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 580/782 ━━━━━━━━━━━━━━[37m━━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 582/782 ━━━━━━━━━━━━━━[37m━━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 584/782 ━━━━━━━━━━━━━━[37m━━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 586/782 ━━━━━━━━━━━━━━[37m━━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 587/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 589/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 591/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 592/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 594/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 596/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 598/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 600/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 602/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 604/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 606/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 608/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 610/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 612/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 614/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 616/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 618/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 619/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 621/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 39ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 622/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 623/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 624/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 626/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 628/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 630/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 632/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 634/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 635/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 637/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 638/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 640/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 641/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 642/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 644/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 646/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 648/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 650/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 652/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 654/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 656/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 658/782 ━━━━━━━━━━━━━━━━[37m━━━━  4s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 660/782 ━━━━━━━━━━━━━━━━[37m━━━━  4s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 662/782 ━━━━━━━━━━━━━━━━[37m━━━━  4s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 664/782 ━━━━━━━━━━━━━━━━[37m━━━━  4s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 666/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 668/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8688 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 670/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 672/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 674/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 676/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 678/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 679/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 680/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 682/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 684/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 686/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 688/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 690/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 692/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 694/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 696/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 698/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 700/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 701/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 703/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 704/782 ━━━━━━━━━━━━━━━━━━[37m━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 706/782 ━━━━━━━━━━━━━━━━━━[37m━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 707/782 ━━━━━━━━━━━━━━━━━━[37m━━  3s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 709/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 711/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 713/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 715/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 716/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 717/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 719/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 721/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 722/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 724/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 726/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 727/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 728/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8687 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 729/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 730/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 731/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 732/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 733/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 734/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 735/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 736/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 738/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 739/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 740/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 742/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 744/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3118

<div class="k-default-codeblock">
```

```
</div>
 746/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 747/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 749/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 751/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 753/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 755/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 757/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 40ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 759/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 760/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 762/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 764/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 766/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 768/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 770/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 772/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 774/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 776/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 778/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 779/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 780/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 781/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 41ms/step - accuracy: 0.8686 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 32s 41ms/step - accuracy: 0.8686 - loss: 0.3119





<div class="k-default-codeblock">
```
[0.31272706389427185, 0.8675600290298462]

```
</div>
---
## Make an end-to-end model

If you want to obtain a model capable of processing raw strings, you can simply
create a new model (using the weights we just trained):


```python

# We create a custom Model to override the evaluate method so
# that it first pre-process text data
class ModelEndtoEnd(keras.Model):

    def evaluate(self, inputs):

        # Turn strings into vocab indices
        test_ds = inputs.map(vectorize_text)
        indices = test_ds.cache().prefetch(buffer_size=10)
        return super().evaluate(indices)

    # Build the model
    def build(self, input_shape):
        self.built = True


def get_end_to_end(model):
    inputs = model.inputs[0]
    outputs = model.outputs
    end_to_end_model = ModelEndtoEnd(inputs, outputs, name="end_to_end_model")
    end_to_end_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


end_to_end_classification_model = get_end_to_end(model)
# Pass raw text dataframe to the model
end_to_end_classification_model.evaluate(raw_test_ds)
```

    
   1/782 [37m━━━━━━━━━━━━━━━━━━━━  1:05 83ms/step - accuracy: 0.8125 - loss: 0.3931

<div class="k-default-codeblock">
```

```
</div>
   2/782 [37m━━━━━━━━━━━━━━━━━━━━  42s 55ms/step - accuracy: 0.8203 - loss: 0.4013 

<div class="k-default-codeblock">
```

```
</div>
   4/782 [37m━━━━━━━━━━━━━━━━━━━━  31s 40ms/step - accuracy: 0.8294 - loss: 0.3984

<div class="k-default-codeblock">
```

```
</div>
   5/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 44ms/step - accuracy: 0.8335 - loss: 0.3878

<div class="k-default-codeblock">
```

```
</div>
   7/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 43ms/step - accuracy: 0.8426 - loss: 0.3660

<div class="k-default-codeblock">
```

```
</div>
   9/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 41ms/step - accuracy: 0.8498 - loss: 0.3484

<div class="k-default-codeblock">
```

```
</div>
  11/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 43ms/step - accuracy: 0.8533 - loss: 0.3379

<div class="k-default-codeblock">
```

```
</div>
  13/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 43ms/step - accuracy: 0.8540 - loss: 0.3337

<div class="k-default-codeblock">
```

```
</div>
  15/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 44ms/step - accuracy: 0.8536 - loss: 0.3328

<div class="k-default-codeblock">
```

```
</div>
  17/782 [37m━━━━━━━━━━━━━━━━━━━━  34s 45ms/step - accuracy: 0.8536 - loss: 0.3325

<div class="k-default-codeblock">
```

```
</div>
  19/782 [37m━━━━━━━━━━━━━━━━━━━━  32s 43ms/step - accuracy: 0.8533 - loss: 0.3325

<div class="k-default-codeblock">
```

```
</div>
  20/782 [37m━━━━━━━━━━━━━━━━━━━━  33s 44ms/step - accuracy: 0.8536 - loss: 0.3321

<div class="k-default-codeblock">
```

```
</div>
  23/782 [37m━━━━━━━━━━━━━━━━━━━━  31s 41ms/step - accuracy: 0.8550 - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
  25/782 [37m━━━━━━━━━━━━━━━━━━━━  31s 42ms/step - accuracy: 0.8559 - loss: 0.3291

<div class="k-default-codeblock">
```

```
</div>
  27/782 [37m━━━━━━━━━━━━━━━━━━━━  30s 41ms/step - accuracy: 0.8571 - loss: 0.3274

<div class="k-default-codeblock">
```

```
</div>
  29/782 [37m━━━━━━━━━━━━━━━━━━━━  30s 40ms/step - accuracy: 0.8582 - loss: 0.3260

<div class="k-default-codeblock">
```

```
</div>
  31/782 [37m━━━━━━━━━━━━━━━━━━━━  30s 40ms/step - accuracy: 0.8590 - loss: 0.3249

<div class="k-default-codeblock">
```

```
</div>
  33/782 [37m━━━━━━━━━━━━━━━━━━━━  30s 40ms/step - accuracy: 0.8598 - loss: 0.3238

<div class="k-default-codeblock">
```

```
</div>
  35/782 [37m━━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8606 - loss: 0.3227

<div class="k-default-codeblock">
```

```
</div>
  37/782 [37m━━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8611 - loss: 0.3219

<div class="k-default-codeblock">
```

```
</div>
  39/782 [37m━━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8617 - loss: 0.3211

<div class="k-default-codeblock">
```

```
</div>
  41/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8622 - loss: 0.3203

<div class="k-default-codeblock">
```

```
</div>
  43/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8627 - loss: 0.3195

<div class="k-default-codeblock">
```

```
</div>
  45/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8630 - loss: 0.3190

<div class="k-default-codeblock">
```

```
</div>
  47/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8632 - loss: 0.3187

<div class="k-default-codeblock">
```

```
</div>
  49/782 ━[37m━━━━━━━━━━━━━━━━━━━  29s 40ms/step - accuracy: 0.8636 - loss: 0.3183

<div class="k-default-codeblock">
```

```
</div>
  51/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 39ms/step - accuracy: 0.8639 - loss: 0.3178

<div class="k-default-codeblock">
```

```
</div>
  53/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 39ms/step - accuracy: 0.8643 - loss: 0.3174

<div class="k-default-codeblock">
```

```
</div>
  55/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 39ms/step - accuracy: 0.8646 - loss: 0.3169

<div class="k-default-codeblock">
```

```
</div>
  57/782 ━[37m━━━━━━━━━━━━━━━━━━━  28s 39ms/step - accuracy: 0.8649 - loss: 0.3165

<div class="k-default-codeblock">
```

```
</div>
  59/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8651 - loss: 0.3163

<div class="k-default-codeblock">
```

```
</div>
  61/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 38ms/step - accuracy: 0.8653 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
  63/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 38ms/step - accuracy: 0.8655 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
  65/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8656 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
  67/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8658 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
  69/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8659 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
  71/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8661 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
  73/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8662 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
  75/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 38ms/step - accuracy: 0.8663 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
  76/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8664 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
  78/782 ━[37m━━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8664 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
  80/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8665 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
  82/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8666 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
  84/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8666 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
  86/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8666 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
  88/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8666 - loss: 0.3158

<div class="k-default-codeblock">
```

```
</div>
  90/782 ━━[37m━━━━━━━━━━━━━━━━━━  27s 39ms/step - accuracy: 0.8667 - loss: 0.3158

<div class="k-default-codeblock">
```

```
</div>
  92/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8667 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
  94/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8667 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
  96/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8667 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
  98/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8667 - loss: 0.3160

<div class="k-default-codeblock">
```

```
</div>
 100/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3160

<div class="k-default-codeblock">
```

```
</div>
 102/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 40ms/step - accuracy: 0.8668 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 104/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 106/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 108/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 110/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 112/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 113/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 115/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 116/782 ━━[37m━━━━━━━━━━━━━━━━━━  26s 39ms/step - accuracy: 0.8668 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 118/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8669 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 120/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8669 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 122/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8669 - loss: 0.3162

<div class="k-default-codeblock">
```

```
</div>
 124/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8670 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 126/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8670 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 128/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8670 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 129/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8670 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 131/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8671 - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 133/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8671 - loss: 0.3160

<div class="k-default-codeblock">
```

```
</div>
 135/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8671 - loss: 0.3160

<div class="k-default-codeblock">
```

```
</div>
 136/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8672 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
 138/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8672 - loss: 0.3159

<div class="k-default-codeblock">
```

```
</div>
 140/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8672 - loss: 0.3158

<div class="k-default-codeblock">
```

```
</div>
 142/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8673 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 143/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 39ms/step - accuracy: 0.8673 - loss: 0.3157

<div class="k-default-codeblock">
```

```
</div>
 144/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 40ms/step - accuracy: 0.8673 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 146/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 40ms/step - accuracy: 0.8674 - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 147/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 40ms/step - accuracy: 0.8674 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
 149/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 40ms/step - accuracy: 0.8674 - loss: 0.3155

<div class="k-default-codeblock">
```

```
</div>
 151/782 ━━━[37m━━━━━━━━━━━━━━━━━  25s 40ms/step - accuracy: 0.8675 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 153/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8675 - loss: 0.3154

<div class="k-default-codeblock">
```

```
</div>
 155/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8675 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 156/782 ━━━[37m━━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8675 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 158/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8676 - loss: 0.3153

<div class="k-default-codeblock">
```

```
</div>
 160/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8676 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
 162/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8676 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
 164/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8676 - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
 166/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8676 - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 168/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8677 - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 169/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8677 - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 171/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8677 - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 173/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8677 - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 175/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8678 - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 177/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8678 - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 179/782 ━━━━[37m━━━━━━━━━━━━━━━━  24s 40ms/step - accuracy: 0.8678 - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 181/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8678 - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 183/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 185/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 186/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 188/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 190/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 191/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 193/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 195/782 ━━━━[37m━━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 196/782 ━━━━━[37m━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 198/782 ━━━━━[37m━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 200/782 ━━━━━[37m━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 202/782 ━━━━━[37m━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 204/782 ━━━━━[37m━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 206/782 ━━━━━[37m━━━━━━━━━━━━━━━  23s 40ms/step - accuracy: 0.8679 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 208/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 210/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 211/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 212/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 213/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 214/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 216/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 218/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 220/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 222/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 223/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 225/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 227/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3145

<div class="k-default-codeblock">
```

```
</div>
 229/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8680 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 231/782 ━━━━━[37m━━━━━━━━━━━━━━━  22s 40ms/step - accuracy: 0.8681 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 233/782 ━━━━━[37m━━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 235/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 237/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 239/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
 241/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 243/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 244/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 246/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 248/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 250/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 252/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 254/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 256/782 ━━━━━━[37m━━━━━━━━━━━━━━  21s 40ms/step - accuracy: 0.8681 - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 258/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8681 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 260/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8681 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 262/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8681 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 263/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 265/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 267/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 269/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 271/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 273/782 ━━━━━━[37m━━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 274/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3140

<div class="k-default-codeblock">
```

```
</div>
 275/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 277/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 278/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 280/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 282/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 283/782 ━━━━━━━[37m━━━━━━━━━━━━━  20s 40ms/step - accuracy: 0.8682 - loss: 0.3139

<div class="k-default-codeblock">
```

```
</div>
 285/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 287/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 288/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 290/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 291/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 292/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 294/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3138

<div class="k-default-codeblock">
```

```
</div>
 296/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 298/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 300/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 302/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 303/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 305/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 307/782 ━━━━━━━[37m━━━━━━━━━━━━━  19s 40ms/step - accuracy: 0.8682 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 309/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8682 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 311/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8682 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 312/782 ━━━━━━━[37m━━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8682 - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 313/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8682 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 314/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 316/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 318/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 320/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 322/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 324/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 326/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 328/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 330/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 332/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 334/782 ━━━━━━━━[37m━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.8683 - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 336/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8683 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 338/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8683 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 340/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 342/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 344/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 346/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 347/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 349/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 350/782 ━━━━━━━━[37m━━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 352/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3131

<div class="k-default-codeblock">
```

```
</div>
 354/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 356/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 357/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 359/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 40ms/step - accuracy: 0.8684 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 360/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 41ms/step - accuracy: 0.8684 - loss: 0.3130

<div class="k-default-codeblock">
```

```
</div>
 362/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 41ms/step - accuracy: 0.8684 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 363/782 ━━━━━━━━━[37m━━━━━━━━━━━  17s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 365/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 366/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 367/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 368/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 369/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 370/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 371/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3129

<div class="k-default-codeblock">
```

```
</div>
 373/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 375/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 377/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 379/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 381/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 383/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 385/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 386/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 388/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 390/782 ━━━━━━━━━[37m━━━━━━━━━━━  16s 41ms/step - accuracy: 0.8685 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 392/782 ━━━━━━━━━━[37m━━━━━━━━━━  16s 41ms/step - accuracy: 0.8686 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 394/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 396/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 398/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 400/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 402/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 404/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 406/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 408/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 410/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 412/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 413/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 415/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 416/782 ━━━━━━━━━━[37m━━━━━━━━━━  15s 41ms/step - accuracy: 0.8686 - loss: 0.3126

<div class="k-default-codeblock">
```

```
</div>
 418/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 420/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 422/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 424/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 426/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 428/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 430/782 ━━━━━━━━━━[37m━━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 432/782 ━━━━━━━━━━━[37m━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 433/782 ━━━━━━━━━━━[37m━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 435/782 ━━━━━━━━━━━[37m━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 437/782 ━━━━━━━━━━━[37m━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 439/782 ━━━━━━━━━━━[37m━━━━━━━━━  14s 41ms/step - accuracy: 0.8686 - loss: 0.3125

<div class="k-default-codeblock">
```

```
</div>
 441/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 443/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 445/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 447/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 449/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 451/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 453/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 455/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 457/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 458/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 459/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 461/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 463/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 464/782 ━━━━━━━━━━━[37m━━━━━━━━━  13s 41ms/step - accuracy: 0.8686 - loss: 0.3124

<div class="k-default-codeblock">
```

```
</div>
 466/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 467/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 469/782 ━━━━━━━━━━━[37m━━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 470/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 472/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 474/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 475/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 476/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 478/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 480/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 482/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 484/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 486/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 488/782 ━━━━━━━━━━━━[37m━━━━━━━━  12s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 490/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 492/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 494/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 496/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 498/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 500/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8686 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 501/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 502/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 503/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 505/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 507/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 508/782 ━━━━━━━━━━━━[37m━━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 509/782 ━━━━━━━━━━━━━[37m━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 511/782 ━━━━━━━━━━━━━[37m━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 512/782 ━━━━━━━━━━━━━[37m━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 514/782 ━━━━━━━━━━━━━[37m━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 515/782 ━━━━━━━━━━━━━[37m━━━━━━━  11s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 517/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 518/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 520/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 522/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 524/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 526/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 527/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 529/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 531/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 533/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 535/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 537/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 539/782 ━━━━━━━━━━━━━[37m━━━━━━━  10s 41ms/step - accuracy: 0.8685 - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 541/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122 

<div class="k-default-codeblock">
```

```
</div>
 543/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 545/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 547/782 ━━━━━━━━━━━━━[37m━━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 549/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 550/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 552/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 554/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 556/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 558/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 559/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 561/782 ━━━━━━━━━━━━━━[37m━━━━━━  9s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 563/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 565/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 567/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 569/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 571/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 573/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 575/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3122

<div class="k-default-codeblock">
```

```
</div>
 577/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 578/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 580/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 581/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 582/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 584/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 585/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 586/782 ━━━━━━━━━━━━━━[37m━━━━━━  8s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 588/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 590/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 592/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 594/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 596/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 598/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 600/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 602/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 604/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 605/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 607/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 609/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 611/782 ━━━━━━━━━━━━━━━[37m━━━━━  7s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 613/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 614/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 615/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 616/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 617/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 618/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 619/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 620/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 622/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 623/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 625/782 ━━━━━━━━━━━━━━━[37m━━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 627/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 41ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 628/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 629/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 631/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 632/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 633/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 634/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 635/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 636/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 638/782 ━━━━━━━━━━━━━━━━[37m━━━━  6s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 639/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 641/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 643/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 645/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 647/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 648/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 650/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 652/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 654/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 656/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 658/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 660/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 662/782 ━━━━━━━━━━━━━━━━[37m━━━━  5s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 663/782 ━━━━━━━━━━━━━━━━[37m━━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 665/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 666/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 668/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 670/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 672/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 674/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 676/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8685 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 678/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 680/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 682/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 684/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 686/782 ━━━━━━━━━━━━━━━━━[37m━━━  4s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 688/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 689/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 690/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 691/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 692/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 694/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 695/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 697/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 699/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 700/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 701/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 703/782 ━━━━━━━━━━━━━━━━━[37m━━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 705/782 ━━━━━━━━━━━━━━━━━━[37m━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 707/782 ━━━━━━━━━━━━━━━━━━[37m━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 709/782 ━━━━━━━━━━━━━━━━━━[37m━━  3s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 711/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 713/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 715/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 717/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 718/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 719/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 720/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 721/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 722/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 723/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 724/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 725/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 727/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 729/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 731/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 733/782 ━━━━━━━━━━━━━━━━━━[37m━━  2s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 735/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 736/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 738/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 740/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 741/782 ━━━━━━━━━━━━━━━━━━[37m━━  1s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 743/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8684 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 745/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 747/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 748/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3119

<div class="k-default-codeblock">
```

```
</div>
 750/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 752/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 754/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 756/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 757/782 ━━━━━━━━━━━━━━━━━━━[37m━  1s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 759/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 761/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 763/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 765/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 767/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 769/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 770/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 771/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 773/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 774/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 776/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 777/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 779/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 781/782 ━━━━━━━━━━━━━━━━━━━[37m━  0s 42ms/step - accuracy: 0.8683 - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 782/782 ━━━━━━━━━━━━━━━━━━━━ 33s 42ms/step - accuracy: 0.8683 - loss: 0.3120





<div class="k-default-codeblock">
```
[0.31272727251052856, 0.8675600290298462]

```
</div>