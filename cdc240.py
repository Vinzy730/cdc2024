import numpy as np
import pandas as pd

ds = pd.read_csv('data_SS.csv')
df = pd.DataFrame(columns = ["mean", "number of reviews", "number of no reviews"])
i=1

while i < 25:
    num_of_review = 0
    num_not_review = 0
    mean = 0

    num_of_review = ds[ds[f"Category {i}"] > 0].count(axis="index").loc['User']
    #before .count -> filters out rows with 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    num_not_review = ds[ds[f"Category {i}"] == 0].count(axis="index").loc['User']
    #before .count -> filters to only get rows with 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    mean = ds[f"Category {i}"].sum() / num_of_review
    # .sum -> adds up the row (don't have to filter out 0 cause adding 0 doesn't change anything)

    df.loc[len(df)] = {"mean" : mean, "number of reviews" : num_of_review, "number of no reviews" : num_not_review}
    #add to the new dataset the mean, number of users who review that type of attraction, number of users who don't review that type of attraction

    i +=1
    

print('\nResult Dataset:\n', df)

df.to_csv('data_1.csv', header = False)
#Undo DocString to save First Analysis as csv