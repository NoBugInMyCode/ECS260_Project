import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('AI_repos.csv')
df2 = pd.read_csv('nonAI_repos.csv')

#print(df.iloc[0])

ai_top_5_percent = df[df['stars'] >= df['stars'].quantile(0.95)]
ai_bottom_5_percent = df[df['stars'] <= df['stars'].quantile(0.05)]
ai_mid_50_55 = df[df['stars'].between(df['stars'].quantile(0.5),df['stars'].quantile(0.55))]

ai_avg_line_of_codes = df['lines_of_codes'].mean()
ai_avg_readme_size = df['readme_size'].mean()
ai_avg_pull_requests = df['pull_requests'].mean()
ai_avg_releases_freq = df['releases_freq'].mean()
ai_avg_stars = df['stars'].mean()

ai_top = ai_top_5_percent[ai_top_5_percent['lines_of_codes'] <= ai_avg_line_of_codes]
ai_top = ai_top[ai_top['readme_size'] >= ai_avg_readme_size]
ai_top = ai_top[ai_top['releases_freq'] >= ai_avg_releases_freq]

ai_bottom = ai_bottom_5_percent[ai_bottom_5_percent['lines_of_codes'] >= ai_avg_line_of_codes]
ai_bottom = ai_bottom[ai_bottom['readme_size'] <= ai_avg_readme_size]
ai_bottom = ai_bottom[ai_bottom['releases_freq'] <= ai_avg_releases_freq]


# print(ai_top.size / ai_top_5_percent.size)
# print(ai_bottom.size / ai_bottom_5_percent.size)
# print(ai_top.size)
# print(ai_top.loc[df['stars'].idxmax()])
print(ai_top.iloc[2])
print(ai_bottom.iloc[2])
# print(ai_avg_stars)
# print(ai_avg_line_of_codes)
# print(ai_avg_readme_size)
# print(ai_avg_pull_requests)
# print(ai_avg_releases_freq)

non_ai_top_5_percent = df2[df2['stars'] >= df2['stars'].quantile(0.95)]
non_ai_bottom_5_percent = df2[df2['stars'] <= df2['stars'].quantile(0.05)]
non_ai_mid_50_55 = df2[df2['stars'].between(df2['stars'].quantile(0.5),df2['stars'].quantile(0.55))]

non_ai_avg_line_of_codes = df2['lines_of_codes'].mean()
non_ai_avg_readme_size = df2['readme_size'].mean()
non_ai_avg_pull_requests = df2['pull_requests'].mean()
non_ai_avg_releases_freq = df2['releases_freq'].mean()
non_ai_avg_stars = df2['stars'].mean()

non_ai_top = non_ai_top_5_percent[non_ai_top_5_percent['lines_of_codes'] <= non_ai_avg_line_of_codes]
non_ai_top = non_ai_top[non_ai_top['readme_size'] >= non_ai_avg_readme_size]
non_ai_top = non_ai_top[non_ai_top['releases_freq'] >= non_ai_avg_releases_freq]

non_ai_bottom = non_ai_bottom_5_percent[non_ai_bottom_5_percent['lines_of_codes'] >= non_ai_avg_line_of_codes]
non_ai_bottom = non_ai_bottom[non_ai_bottom['readme_size'] <= non_ai_avg_readme_size]
non_ai_bottom = non_ai_bottom[non_ai_bottom['releases_freq'] <= non_ai_avg_releases_freq]

# print(non_ai_top.size / non_ai_top_5_percent.size)
# print(non_ai_bottom.size / non_ai_bottom_5_percent.size)
# print(non_ai_top.size)
# print(non_ai_top.loc[df2['stars'].idxmax()])
print(non_ai_top.iloc[1])
print(non_ai_bottom.iloc[0])
# print(non_ai_avg_stars)
# print(non_ai_avg_line_of_codes)
# print(non_ai_avg_readme_size)
# print(non_ai_avg_pull_requests)
# print(non_ai_avg_releases_freq)


print(ai_mid_50_55.iloc[0])
print(non_ai_mid_50_55.iloc[0])
