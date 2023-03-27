import pandas as pd
print(open('test.txt').read())
cols=['n','fp','time','time/w']
rows= ['1','2','3','4']
data = pd.read_csv('test.txt', header = None)
print(data)

