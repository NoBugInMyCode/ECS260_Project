import pandas as pd

df = pd.read_csv('ai.csv')
df2 = pd.read_csv('non_ai.csv')

date2017 = pd.to_datetime("2017-06-12T17:57:34Z")

ai_before2017 = df[pd.to_datetime(df["creation_date"]) < date2017]
ai_after2017 =  df[pd.to_datetime(df["creation_date"]) > date2017]

ai_popular_before2017_stars = ai_before2017[ai_before2017['stars'] > ai_before2017['stars'].mean()]
ai_popular_after2017_stars = ai_after2017[ai_after2017['stars'] > ai_after2017['stars'].mean()]
ai_popular_before2017_score1 = ai_before2017[ai_before2017['popularity_score_1'] > ai_before2017['popularity_score_1'].mean()]
ai_popular_after2017_score1 = ai_after2017[ai_after2017['popularity_score_1'] > ai_after2017['popularity_score_1'].mean()]
ai_popular_before2017_score2 = ai_before2017[ai_before2017['popularity_score_2'] > ai_before2017['popularity_score_2'].mean()]
ai_popular_after2017_score2 = ai_after2017[ai_after2017['popularity_score_2'] > ai_after2017['popularity_score_2'].mean()]
ai_popular_before2017_score3 = ai_before2017[ai_before2017['popularity_score_3'] > ai_before2017['popularity_score_3'].mean()]
ai_popular_after2017_score3 = ai_after2017[ai_after2017['popularity_score_3'] > ai_after2017['popularity_score_3'].mean()]

nonai_before2017 = df2[pd.to_datetime(df2["creation_date"]) < date2017]
nonai_after2017 =  df2[pd.to_datetime(df2["creation_date"]) > date2017]

nonai_popular_before2017_stars = nonai_before2017[nonai_before2017['stars'] > nonai_before2017['stars'].mean()]
nonai_popular_after2017_stars = nonai_after2017[nonai_after2017['stars'] > nonai_after2017['stars'].mean()]
nonai_popular_before2017_score1 = nonai_before2017[nonai_before2017['popularity_score_1'] > nonai_before2017['popularity_score_1'].mean()]
nonai_popular_after2017_score1 = nonai_after2017[nonai_after2017['popularity_score_1'] > nonai_after2017['popularity_score_1'].mean()]
nonai_popular_before2017_score2 = nonai_before2017[nonai_before2017['popularity_score_2'] > nonai_before2017['popularity_score_2'].mean()]
nonai_popular_after2017_score2 = nonai_after2017[nonai_after2017['popularity_score_2'] > nonai_after2017['popularity_score_2'].mean()]
nonai_popular_before2017_score3 = nonai_before2017[nonai_before2017['popularity_score_3'] > nonai_before2017['popularity_score_3'].mean()]
nonai_popular_after2017_score3 = nonai_after2017[nonai_after2017['popularity_score_3'] > nonai_after2017['popularity_score_3'].mean()]

ai_popular_proportion_before2017_stars = len(ai_popular_before2017_stars)/(len(ai_popular_before2017_stars)+len(nonai_popular_before2017_stars))
ai_popular_proportion_after2017_stars = len(ai_popular_after2017_stars)/(len(ai_popular_after2017_stars)+len(nonai_popular_after2017_stars))
ai_popular_proportion_before2017_score1 = len(ai_popular_before2017_score1)/(len(ai_popular_before2017_score1)+len(nonai_popular_before2017_score1))
ai_popular_proportion_after2017_score1 = len(ai_popular_after2017_score1)/(len(ai_popular_after2017_score1)+len(nonai_popular_after2017_score1))
ai_popular_proportion_before2017_score2 = len(ai_popular_before2017_score2)/(len(ai_popular_before2017_score2)+len(nonai_popular_before2017_score2))
ai_popular_proportion_after2017__score2 = len(ai_popular_after2017_score2)/(len(ai_popular_after2017_score2)+len(nonai_popular_after2017_score2))
ai_popular_proportion_before2017_score3 = len(ai_popular_before2017_score3)/(len(ai_popular_before2017_score3)+len(nonai_popular_before2017_score3))
ai_popular_proportion_after2017_score3 = len(ai_popular_after2017_score3)/(len(ai_popular_after2017_score3)+len(nonai_popular_after2017_score3))



print("stars:")
print('ai popular proportion before 2017: '+ str(ai_popular_proportion_before2017_stars))
print('ai popular proportion after 2017: '+ str(ai_popular_proportion_after2017_stars))

print("popularity score1:")
print('ai popular proportion before 2017: '+ str(ai_popular_proportion_before2017_score1))
print('ai popular proportion after 2017: '+ str(ai_popular_proportion_after2017_score1))

print("popularity score2:")
print('ai popular proportion before 2017: '+ str(ai_popular_proportion_before2017_score2))
print('ai popular proportion after 2017: '+ str(ai_popular_proportion_after2017__score2))

print("popularity score3:")
print('ai popular proportion before 2017: '+ str(ai_popular_proportion_before2017_score3))
print('ai popular proportion after 2017: '+ str(ai_popular_proportion_after2017_score3))

