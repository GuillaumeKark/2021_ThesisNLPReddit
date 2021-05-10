# -*- coding: utf-8 -*-
"""
April 2021
Comment: This file reads the txt files and convert them into a formatted CSV that can be used to identify stocks.
"""

import pandas as pd

PATH = ""

#convert stocks to .csv
f = open(PATH + "stock_names.txt", "r")
text = f.read()
f.close()

text = text.split("\n")

output = []

for i in text:
    print(i)
    output.append(i.split(" - ", 1))
    
output = pd.DataFrame.from_records(output, columns = ["ticker", "name"])

output.to_csv(PATH + "stock_names.csv", index = False)


#convert cryptos to .csv
f = open(PATH + "crypto_names.txt", "r")
text = f.read()
f.close()

text = text.split("\n")
temp = []

#star shifts the text from one line, so we remove lines with * to split names
for i in text:
    if "*" not in i:
        temp.append(i)
        
text = []
tick = []
for i in range(2, len(temp)):
    if (i - 2) % 11 == 0: #Each stock takes 11 lines so we can run a loop to extract lines 2 and 3 for each.
        text.append(temp[i])
        
    elif (i - 3) % 11 == 0:
        tick.append(temp[i])

output = pd.DataFrame({"ticker": tick, "name": text})

output.to_csv(PATH + "crypto_names.csv", index = False)
     