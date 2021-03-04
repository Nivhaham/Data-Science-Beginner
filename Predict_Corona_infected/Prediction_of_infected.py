import numpy as np
import matplotlib.pyplot as plt
import csv

import pandas as pd

# 5.
# Range of rows for each location divided to 34 weeks array each from date 15.03.2020 - 07.11.2020
# Need to predict the the 28th week till the 34th week of France.
# Belgium 2:239
# France 240:477 - data set 240:428, predict 429:477
# Germany 480:717
# Italy 720:957
# Portugal 960:1197
# Spain 1200:1437
# Switzerland 1439:1676
# UK 1678:1916
#### Todo : Many days in spain and switz need to be exluded because of alot 0.
df = pd.read_csv("Data.csv")
# print(df)
i, j = 0, 7
france_pop = 65273512
# each list is 34 long with weekly avarage divided by populaiton
belarr, frarr, gerarr, itaarr, porarr, spaarr, swiarr, ukarr = [], [], [], [], [], [], [], []
francepreditcion = np.zeros(7)

# These loop append weekly avarge of new cases relative to population
while (j <= 240):
    # print(df['new_cases'].iloc[i:j].mean)
    belarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[2, 3]))
    j += 7
    i += 7
# print(belarr)
i, j = 238, 245
while (j < 430):
    # print(df['new_cases'].iloc[i:j].mean)
    frarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[240, 3]))
    j += 7
    i += 7
# print(frarr)
i, j = 478, 485
while (j < 720):
    # print(df['new_cases'].iloc[i:j].mean)
    gerarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[480, 3]))
    j += 7
    i += 7
# print(Gerarr)
i, j = 718, 725
while (j < 960):
    # print(df['new_cases'].iloc[i:j].mean)
    itaarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[720, 3]))
    j += 7
    i += 7
# print(Itaarr)
i, j = 958, 965
while (j < 1200):
    # print(df['new_cases'].iloc[i:j].mean)
    porarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[960, 3]))
    j += 7
    i += 7
# print(len(Porarr))
i, j = 1198, 1205
while (j < 1440):
    # print(df['new_cases'].iloc[i:j].mean)
    # if(df['new_cases'].iloc[i:j]!=0):
    spaarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[1200, 3]))
    j += 7
    i += 7
# print(Spaarr)
i, j = 1437, 1444
while (j < 1678):
    # print(df['new_cases'].iloc[i:j].mean)
    # if(df['new_cases'].iloc[i:j]!=0):
    swiarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[1437, 3]))
    j += 7
    i += 7
# print(Swiarr)
i, j = 1677, 1684
while (j < 1917):
    # print(df['new_cases'].iloc[i:j].mean)
    ukarr.append((df['new_cases'].iloc[i:j].mean()) / float(df.iloc[1677, 3]))
    j += 7
    i += 7
# print(Ukarr)

# print(FrPerdictionarr)

# TODO:
# belgium , germany, italy, portugal, spain, switzerland, uk
# these countries are represented by 34X7 matrix
other_countries = np.zeros((34, 7))
other_countries[:, 0] = np.array(belarr)
other_countries[:, 1] = np.array(gerarr)
other_countries[:, 2] = np.array(itaarr)
other_countries[:, 3] = np.array(porarr)
other_countries[:, 4] = np.array(spaarr)
other_countries[:, 5] = np.array(swiarr)
other_countries[:, 6] = np.array(ukarr)

france = np.array(frarr)
france = np.append(france, francepreditcion)
france.resize(27, 1)
# print(other_countries[27:34])
other_countries_predict = np.copy(other_countries[27:34])
other_countries.resize(27, 7)
# print(other_countries)
# print(france)
# print(len(other_countries))
# We will do Some Math, beta =(At * A)-1 * At x
# We will discover the beta using our training set
# print(np.linalg.inv(other_countries.transpose().dot(other_countries)))
# My Best equation for least squre :)
beta = np.linalg.inv(other_countries.transpose().dot(other_countries)).dot(other_countries.transpose()).dot(france)
# print(beta)

# now that we have our beta, we will try to predict covid weekly cases of weeks 28-34.
print("The weekly avarage of new cases predicted by my not so good Algo is:")
prediction = (other_countries_predict.dot(beta)).transpose() * france_pop
print(prediction)#(other_countries_predict.dot(beta)).transpose() * france_pop)
print("Real Data is")
i, j = 429, 436
while (j < 482):
    # print(df['new_cases'].iloc[i:j].mean)
    frarr.append((df['new_cases'].iloc[i:j].mean()))
    j += 7
    i += 7
print(frarr[27:34])

print("last week prediction is pretty bad maybe because some 0 new cases in switz and spain")

#plt.plot(prediction)
#plt.plot(frarr[27:34])
x = (np.arange(7))+1
#print(x)
plt.xlabel("Week")
plt.ylabel("New Cases")
plt.title("France new cases prediction last 7 weeks ")
plt.scatter(x, prediction, )
plt.scatter(x, frarr[27:34])
plt.legend(["prediction", "real data"])
plt.show()