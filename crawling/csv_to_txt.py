import pandas as pd

csv = pd.read_csv('dataset.csv')
csv = csv['txt']

csv.to_csv('dataset.txt', index=False, header=None, sep="\t")