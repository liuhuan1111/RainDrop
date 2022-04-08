import altair as alt
import pandas as pd

#for line in open('./MSELoss_list.txt')
with open(r'G:/Desktop/MSELoss_list.txt','r',encoding='utf-8') as file:
    content_list = file.readlines()
ii=[i+1 for i in range(len(content_list))]
# print(ii)
data = pd.DataFrame({'x':ii,
                     'y': content_list})

alt.Chart(data).mark_line(point=False).encode(
    x='x:Q',  # specify ordinal data
    y='y:Q',  # specify quantitative data
).show()