import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#=====================Negatif=========================# 
data = pd.read_csv('data_akhir.csv', sep=';', usecols=('content_clear','polarity'))
data = data[data['polarity'] == 'negatif']
data = data.join(data['content_clear'].str.split(' ', expand=True).add_prefix(''))
del data['polarity']
del data['content_clear']
items=(data['0'].unique())
items
itemset = set(items)
encoded_vals = []
for index, row in data.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]

neg_apri = pd.DataFrame(encoded_vals)
freq_items = apriori(neg_apri, min_support=0.01, use_colnames=True)
freq_items.head(10)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.25)
rules.head()
rules1 = pd.DataFrame(rules).sort_values(by='support', ascending=True)
rules1.head(10)
rules1.to_csv("asosiasinegatif.csv", sep=";", index=False)

#=====================Positif=========================# 
data = pd.read_csv('data_akhir.csv', sep=';', usecols=('content_clear','polarity'))
data = data[data['polarity'] == 'positif']
data = data.join(data['content_clear'].str.split(' ', expand=True).add_prefix(''))
del data['polarity']
del data['content_clear']
data.head()

items=(data['0'].unique())
items
itemset = set(items)
encoded_vals = []
for index, row in data.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]

pos_apri = pd.DataFrame(encoded_vals)
freq_items = apriori(pos_apri, min_support=0.01, use_colnames=True)
freq_items.head(20)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.25)
rules.tail(10)
rules2 = pd.DataFrame(rules).sort_values(by='support', ascending=True)
rules2.head(10)
rules2.to_csv("asosiasipositif.csv", sep=";", index=False)
