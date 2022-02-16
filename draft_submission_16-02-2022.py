#Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')
from datetime import date

# Step 1: let's take a look at the data
# The dataset consists of:
# - building characteristics (e.g. floor area, facility type etc), 
# - weather data for the location of the building (e.g. annual average temperature, annual total precipitation etc)
# - the energy usage for the building and the given year, measured as Site Energy Usage Intensity (Site EUI). 

#Each row in the data corresponds to the a single building observed in a given year.

#read in data
buildings = pd.read_csv(r'../input/widsdatathon2022/train.csv')
print(buildings.shape)
buildings.head()

## Preparing the data 
# Before we can feed the data into our model we need to clean the data to make sure everything is in the format we would like it.
#  This will also help us familiarise ourselves with the data. First let's start by checking for missing data.

#get information on data types
print(buildings.info())

# get information on missing data for each variable

#create a Pandas DataFrame object which contains the sum of all null values in each column
na_buildings = (pd.DataFrame(buildings.isna().sum())).reset_index()

#set the column names in this dataframe to variable and missing count
na_buildings.columns = ['variable', 'missing_count']

#add a new column which represents the missing count as a proportion of the total number of observations
na_buildings['missing_proportion'] = na_buildings['missing_count']/buildings.shape[0]

#print a subset of this dataframe only including variables whose missing count is not equal to 0
print(na_buildings[na_buildings['missing_count']!= 0])

## EDA so far (15/02/2022)

#The buildings dataset comprises 64 variables and 75757 observations.  
#6 of these variables have missing values, of which 4 are missing >54% of values. 

#We cannot simply drop rows missing values , this is far too great and would result in losing a substantial amount of valuable data. 
# Drop variables a considerable proportion of observations are missing? Impute?

#We can also see that a huge amount of data (36 variables) pertain to the min, max and avg. temp (fahrenheit) of the building by month.
#  We should plot this alongside  `site_eui` to look for any associations between energy usage and temperature.

#check for duplicates
buildings.duplicated().any()