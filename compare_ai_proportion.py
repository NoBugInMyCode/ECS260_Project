import pandas as pd

df = pd.read_csv('AI_repos.csv')
df2 = pd.read_csv('nonAI_repos.csv')

date2016 = pd.to_datetime("2016-01-01T00:00:00Z")

ai_before2016 = df[pd.to_datetime(df["creation_date"]) < date2016]
ai_after2016 =  df[pd.to_datetime(df["creation_date"]) > date2016]

ai_popular_before2016_stars = ai_before2016[ai_before2016['stars'] > ai_before2016['stars'].mean()]
ai_popular_after2016_stars = ai_after2016[ai_after2016['stars'] > ai_after2016['stars'].mean()]
ai_popular_before2016_score1 = ai_before2016[ai_before2016['popularity_score_1'] > ai_before2016['popularity_score_1'].mean()]
ai_popular_after2016_score1 = ai_after2016[ai_after2016['popularity_score_1'] > ai_after2016['popularity_score_1'].mean()]
ai_popular_before2016_score2 = ai_before2016[ai_before2016['popularity_score_2'] > ai_before2016['popularity_score_2'].mean()]
ai_popular_after2016_score2 = ai_after2016[ai_after2016['popularity_score_2'] > ai_after2016['popularity_score_2'].mean()]
# ai_popular_before2016_score3 = ai_before2016[ai_before2016['popularity_score_3'] > ai_before2016['popularity_score_3'].mean()]
# ai_popular_after2016_score3 = ai_after2016[ai_after2016['popularity_score_3'] > ai_after2016['popularity_score_3'].mean()]

nonai_before2016 = df2[pd.to_datetime(df2["creation_date"]) < date2016]
nonai_after2016 =  df2[pd.to_datetime(df2["creation_date"]) > date2016]

nonai_popular_before2016_stars = nonai_before2016[nonai_before2016['stars'] > nonai_before2016['stars'].mean()]
nonai_popular_after2016_stars = nonai_after2016[nonai_after2016['stars'] > nonai_after2016['stars'].mean()]
nonai_popular_before2016_score1 = nonai_before2016[nonai_before2016['popularity_score_1'] > nonai_before2016['popularity_score_1'].mean()]
nonai_popular_after2016_score1 = nonai_after2016[nonai_after2016['popularity_score_1'] > nonai_after2016['popularity_score_1'].mean()]
nonai_popular_before2016_score2 = nonai_before2016[nonai_before2016['popularity_score_2'] > nonai_before2016['popularity_score_2'].mean()]
nonai_popular_after2016_score2 = nonai_after2016[nonai_after2016['popularity_score_2'] > nonai_after2016['popularity_score_2'].mean()]
# nonai_popular_before2016_score3 = nonai_before2016[nonai_before2016['popularity_score_3'] > nonai_before2016['popularity_score_3'].mean()]
# nonai_popular_after2016_score3 = nonai_after2016[nonai_after2016['popularity_score_3'] > nonai_after2016['popularity_score_3'].mean()]

ai_popular_proportion_before2016_stars = len(ai_popular_before2016_stars)/(len(ai_popular_before2016_stars)+len(nonai_popular_before2016_stars))
ai_popular_proportion_after2016_stars = len(ai_popular_after2016_stars)/(len(ai_popular_after2016_stars)+len(nonai_popular_after2016_stars))
ai_popular_proportion_before2016_score1 = len(ai_popular_before2016_score1)/(len(ai_popular_before2016_score1)+len(nonai_popular_before2016_score1))
ai_popular_proportion_after2016_score1 = len(ai_popular_after2016_score1)/(len(ai_popular_after2016_score1)+len(nonai_popular_after2016_score1))
ai_popular_proportion_before2016_score2 = len(ai_popular_before2016_score2)/(len(ai_popular_before2016_score2)+len(nonai_popular_before2016_score2))
ai_popular_proportion_after2016__score2 = len(ai_popular_after2016_score2)/(len(ai_popular_after2016_score2)+len(nonai_popular_after2016_score2))
# ai_popular_proportion_before2016_score3 = len(ai_popular_before2016_score3)/(len(ai_popular_before2016_score3)+len(nonai_popular_before2016_score3))
# ai_popular_proportion_after2016_score3 = len(ai_popular_after2016_score3)/(len(ai_popular_after2016_score3)+len(nonai_popular_after2016_score3))



print("stars:")
print('ai popular proportion before 2016: '+ str(ai_popular_proportion_before2016_stars))
print('ai popular proportion after 2016: '+ str(ai_popular_proportion_after2016_stars))

print("popularity score1:")
print('ai popular proportion before 2016: '+ str(ai_popular_proportion_before2016_score1))
print('ai popular proportion after 2016: '+ str(ai_popular_proportion_after2016_score1))

print("popularity score2:")
print('ai popular proportion before 2016: '+ str(ai_popular_proportion_before2016_score2))
print('ai popular proportion after 2016: '+ str(ai_popular_proportion_after2016__score2))

# print("popularity score3:")
# print('ai popular proportion before 2016: '+ str(ai_popular_proportion_before2016_score3))
# print('ai popular proportion after 2016: '+ str(ai_popular_proportion_after2016_score3))

