

[![Build Status](https://raw.githubusercontent.com/ilkayDevran/Searchable_WikiCorpus_with_Spark_TF-IDF/master/assets/spark_logo.png)](https://spark.apache.org/)

## TF-IDF 

[Term frequency inverse document frequency (TF-IDF)](http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvVGbigJNpZGY) is a feature vectorization method. It is generally used in text mining to externalize the matter of a term to a document in the corpus.

Indicate a term by *__t__*, a document by *__d__*, and the corpus by *__D__*. Term frequency *__TF(t, d)__* is the number of times that term *__t__* appears in document *__d__*, while document frequency *__DF(t, D)__* is the number of documents that contains term *__t__*. 

>If we only use term frequency to measure the importance, it is very easy to over-emphasize terms that appear very often but carry little information about the document, e.g., “a”, “the”, and “of”. If a term appears very often across the corpus, it means it doesn’t carry special information about a particular document.

Inverse document frequency is a numerical measure of how much information a term provides:

>![](https://latex.codecogs.com/gif.latex?IDF%28t%2C%20D%29%3Dlog%5Cfrac%7B%7CD%7C&plus;1%7D%7BDF%28t%20%2CD%29&plus;1%7D )

where *__|D|__* is the total number of documents in the corpus. Since logarithm is used, if a term appears in all documents, its IDF value becomes 0. Note that a smoothing term is applied to avoid dividing by zero for terms outside the corpus. The TF-IDF measure is simply the product of TF and IDF:
>   ![](https://latex.codecogs.com/gif.latex?TFIDF%28t%2Cd%2CD%29%3DTF%28t%2Cd%29.IDF%28t%2CD%29)

This implementation of term frequency utilizes the [hashing trick](http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvRmVhdHVyZV9oYXNoaW5n). A raw feature is mapped into an index (term) by applying a hash function. Then term frequencies are calculated based on the mapped indices. This approach avoids the need to compute a global term-to-index map, which can be expensive for a large corpus, but it suffers from potential hash collisions, where different raw features may become the same term after hashing. To reduce the chance of collision, we can increase the target feature dimension, i.e., the number of buckets of the hash table. The default feature dimension is: 
>![](https://latex.codecogs.com/gif.latex?2%5E%7B20%7D%3D1%2C048%2C576)
***
### Sample Code
Import necessary libraries and configuration of Spark job. 
```python
# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)
```
Load the data
```python
# Load documents (one per line).
rawData = sc.textFile("wikipediaCorpus.tsv") 
```
Get the fields by splitting from *\t* . With this way, rawData is became reachable to the desired column for processing
```python
fields = rawData.map(lambda x: x.split("\t"))
```
*wikipediaCorpus.tsv* format looks like: 
>*'index_number **__\t__** term **__\t__** date  **__\t__** article_content'*

Thus, to get word lists *(documents)* in an article content need to get 3. index of fields
```python 
documents = fields.map(lambda x: x[3].split(" ")) 

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])
```
To show a sample of documents
```python
documents.first()
"""
Output:
['Anarchism',
 '(sometimes',
 'referred',
 'to',
 'as',
 ...]
"""
```
Now hash the words in each document to their term frequencies:
```python
# 100K hash buckets just to save some memory
# We can increase and decrease it according to the case 
hashingTF = HashingTF(100000)
# Transforming words into numbers after hashing    
tf = hashingTF.transform(documents) 
```
At this point we have an RDD of sparse vectors representing each document, where each value maps to the term frequency of each unique hash value. Let's compute the TF*IDF of each term in each document:
```python
tf.cache()
# Ignore any word which appears more than once
idf = IDF(minDocFreq=2).fit(tf) 
tfidf = idf.transform(tf)
```
Now we have an RDD of sparse vectors, where each value is the TFxIDF of each unique hash value for each document.

The article for "Abraham Lincoln" is in our data set, so search for "Gettysburg" (Lincoln gave a famous speech there).

First, let's figure out what hash value "Gettysburg" maps to by finding the index a sparse vector from HashingTF gives us back:
```python
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])
```
Now we will extract the TF*IDF score for Gettsyburg's hash value into a new RDD for each document:
```python
gettysburgRelevance = tfidf.map(lambda x:x[gettysburgHashValue])
```
We'll zip in the document names so we can see which is which:
```python
#zip: rdd to write a new rdd
zippedResults = gettysburgRelevance.zip(documentNames) 
```
And, print the document with the maximum TF*IDF value:
```python
print("Best document for Gettysburg is:")
print(zippedResults.max())

"""
Output:
Best document for Gettysburg is:
(11.592697024775392, 'ASCII')
"""
```


