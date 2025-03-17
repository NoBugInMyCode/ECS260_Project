import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('AI_repos.csv')
df2 = pd.read_csv('nonAI_repos.csv')


df['creation_date'] = pd.to_datetime(df['creation_date'])
df = df.sort_values(by='creation_date')
df['year'] = df['creation_date'].dt.year

df2['creation_date'] = pd.to_datetime(df2['creation_date'])
df2 = df2.sort_values(by='creation_date')
df2['year'] = df2['creation_date'].dt.year

avg_ai = []
repo_num_ai = []
avg_nonai = []
repo_num_nonai = []
year = []
year2 = []

group_ai = [df[df['year'] == y] for y in df['year'].unique()]
for data in group_ai:
    avg_ai_score1 = data['stars'].mean()
    avg_ai.append(avg_ai_score1)
    temp = str(data['year'].iloc[0])
    if temp not in year:
        year.append(temp)
    repo_num_ai.append(len(data))



group_nonai = [df2[df2['year'] == y] for y in df2['year'].unique()]
for data in group_nonai:
    avg_nonai_score1 = data['stars'].mean()
    avg_nonai.append(avg_nonai_score1)

    temp = str(data['year'].iloc[0])
    if temp not in year2:
        year2.append(temp)
    repo_num_nonai.append(len(data))


ai_repo_propotion =[x/(x+y) for x,y in zip(repo_num_ai,repo_num_nonai)]
print(ai_repo_propotion)
# plot
x = np.arange(len(year)) 
#x = np.arange(len(year)) 
x2 = np.arange(len(year2)) 
fig, ax1 = plt.subplots()
bar_width = 0.3
ax1.bar(x - bar_width/2, repo_num_ai, width=bar_width, label='AI repo numbers', color='blue', alpha=0.6)
ax1.bar(x2 + bar_width/2, repo_num_nonai, width=bar_width, label='Non AI repo numbers', color='green', alpha=0.6)

ax2 = ax1.twinx()
# ax2.plot(year, avg_ai, marker='o', linestyle='-', color='red', label='AI average score')
# ax2.plot(year2, avg_nonai, marker='s', linestyle='--', color='purple', label='Non AI average score')
ax2.plot(year2, ai_repo_propotion, marker='s', linestyle='--', color='purple', label='AI repos proportion')

ax1.set_xlabel('Year')
ax1.set_ylabel('Repo Number')
ax2.set_ylabel('AI Repo Proportion')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# plt.title('AI and non AI repo numbers and popularity changing with year')
plt.title('AI repos proportion changing with year')
plt.show()



