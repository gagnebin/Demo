#import sys
#from math import sqrt
#import numpy as np

from pyspark.sql import SparkSession # for dataframe and datasets operation
from pyspark.sql import Row
#from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql.functions import col, concat_ws, collect_list, udf, concat, UserDefinedFunction, sqrt


def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if denominator:
        score = numerator / float(denominator)

    return (score, numPairs)


# Use Spark bulit-in cluster manager to use all cores on computer
spark = SparkSession.builder.appName("MovieSimilarities").getOrCreate()


print("\nLoading movie names...")
# Map movie info to prepare data frame
item = spark.sparkContext.textFile("ml-100k/u.ITEM")
# Not ideal - but trying to provide readability
categories = item.map(lambda x: Row(movieID =int(x.split('|')[0]), movieTitle =x.split('|')[1], catUnknown =int(x.split('|')[5]), catAction =int(x.split('|')[6]), catAdventure =int(x.split('|')[7]), 
                                                catAnimation =int(x.split('|')[8]), catChildren =int(x.split('|')[9]), catComedy  =int(x.split('|')[10]),
                                                catCrime  =int(x.split('|')[11]), catDocumentary =int(x.split('|')[12]), catDrama=int(x.split('|')[13]),
                                                catFantasy=int(x.split('|')[14]), catFilmNoir =int(x.split('|')[15]), catHorror=int(x.split('|')[16]), 
                                                catMusical=int(x.split('|')[17]), catMystery=int(x.split('|')[18]), catRomance=int(x.split('|')[19]),
                                                catSciFi=int(x.split('|')[20]), catThriller=int(x.split('|')[21]),catWar =int(x.split('|')[22]),
                                                catWestern =int(x.split('|')[23])))

# Convert that to a DataFrame
categoriesDataset = spark.createDataFrame(categories)

categoriesDataset.show()

# Create a unique code for each combination of categories
# Will use a trick by considering the ones and zeros as a binary number and converting it into a integer
# First, creating the binary number by concatenating the columns
NewcategoriesDataset = categoriesDataset.select(col("movieID"), col("movieTitle"), concat(col("catUnknown"), col("catAction"), col("catAdventure"), col("catAnimation"), col("catChildren"), 
                             col("catComedy"), col("catCrime"), col("catDocumentary"), col("catDrama"), col("catFantasy"),
                             col("catFilmNoir"), col("catHorror"), col("catMusical"), col("catMystery"), col("catRomance"),
                             col("catSciFi"), col("catThriller"), col("catWar"), col("catWestern")).alias("binCat"))
# Check
#NewcategoriesDataset.show()

# Applying the conversion from binary to integer
myfct = UserDefinedFunction(lambda x: int(x,2), StringType())

IntcategoriesDataset = NewcategoriesDataset.withColumn("intCat", myfct(NewcategoriesDataset.binCat)).cache()

# Check
#IntcategoriesDataset.show()


print("\nLoading rating data...")
data = spark.sparkContext.textFile("ml-100k/u.data")

# Map ratings to prepare data frame
ratings = data.map(lambda x: Row(userID =int(x.split()[0]), movieID =int(x.split()[1]), rating =float(x.split()[2])))

# Convert that to a DataFrame
ratingsDataset = spark.createDataFrame(ratings)

# Retaining only the movie with rating higher than 2.
goodRatings = ratingsDataset.filter(ratingsDataset['rating'] > 2.)

#goodRatings.show()


# Joining ratings and categories will be used to keep only pairs of movies of the same category watched by the same user
joinedCatRatings = goodRatings.alias("df1").join(IntcategoriesDataset.alias("df2"), col("df1.movieID") == col("df2.movieID"))\
.select(col("df1.userID"), col("df1.movieID"), col("df1.rating"), col("df2.intCat"))

#joinedCatRatings.show()

# Emit every movie rated together by the same user from the same movie category (Creates pairs)
# Self-join to find every combination.
# Most of the computation will be done here (intensive). Factorial relationships.
joinedRatings = joinedCatRatings.alias("df1").join(joinedCatRatings.alias("df2"), (col("df1.userID") == col("df2.userID")) & (col("df1.intCat") == col("df2.intCat")))\
.select(col("df1.userID"), col("df1.movieID").alias("movieID1"), col("df1.rating").alias("rating1"), col("df2.movieID").alias("movieID2"), col("df2.rating").alias("rating2")).cache()

#joinedRatings.show()

# Filter out duplicate pairs
moviePairRatings = joinedRatings.dropDuplicates().cache()

moviePairRatings.show()

# Need now to collect all ratings for each movie pair and compute similarity
# For that, we should apply a similarity function on grouped data

# It seems UDF on groups is a little bit tricky in python
# but let's try the solution provided here:
    # # https://stackoverflow.com/questions/40006395/applying-udfs-on-groupeddata-in-pyspark-with-functioning-python-example

# Create an UDF to compute the similarity score
def myFunc(data_list):

    sum_xx = sum_yy = sum_xy = 0.
    
    for val in data_list: # val will be each pairs of ratings for a single movies pair
        fields = val.split(',')

        ratingX = float(fields[0])
        ratingY = float(fields[1])
        
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
    
    numerator = float(sum_xy)
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    
    score = 0
    if denominator:
        score = numerator / float(denominator)

    return score

# Register the UDF
myUdf = udf(myFunc, StringType())

# The groupement is done with collect_list
moviePairRatings.withColumn('data', concat_ws(',',col('rating1'),col('rating2')))\
.groupBy('movieID1', 'movieID2').agg(collect_list('data').alias('data'))\
.withColumn('score', myUdf('data'))\
.show(20)

##############################################################
######################### ERROR ##############################
# AttributeError: 'NoneType' object has no attribute '_jvm'  #
# It seems that it cannot handle the call of sqrt on object  #
# provided by collect_list.                                  #
##############################################################