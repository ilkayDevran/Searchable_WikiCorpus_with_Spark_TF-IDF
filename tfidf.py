# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line).
rawData = sc.textFile("wikipediaCorpus.tsv")  


fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])


print (documents.first())


# Now hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000) 
tf = hashingTF.transform(documents) 


# Let's compute the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf) #ignore any word which appears more than once
tfidf = idf.transform(tf)


# First, let's figure out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["şarkıcı"])
gettysburgHashValue = int(gettysburgTF.indices[0])


# Now we will extract the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])


# We'll zip in the document names so we can see which is which:
zippedResults = gettysburgRelevance.zip(documentNames) #zip: rdd den yeni bir rdd yazmayı sağlar. 


# And, print the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())
print()