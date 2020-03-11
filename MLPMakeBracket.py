import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import MinMaxScaler
import tqdm

# ## Making predictions with model

# ### Extract data desired

data_dir = './mens-tournament/MDataFiles_Stage1/'
df_sample_sub = pd.read_csv('./mens-tournament/' + 'MSampleSubmissionStage1_2020.csv')
df_adv = pd.read_csv('MarchMadnessAdvStats_allSeasons.csv')
df_seeds = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')

# load rankings
df_rank = pd.read_csv('Sorted_Massey_Ordinals.csv')
# now, df_rank has columns season|week|System|teamID|rank
# we need to make each system name a col in df_tour and group on season and team id 
# in order to add a column with the raking for each team 
# there is probably a fancy way, but it makes sense reading wise to just loop through and squish on
ord_list = df_rank.SystemName.unique()

def assign_ords(teamID, season):
    """Assign ordinal rankings to teams according to teamID and season"""
    team_ords = []
    for o in ord_list:
        # make temp dataframe for each ordinal
        df_temp = df_rank[df_rank.SystemName==o]
        # keep only the columns of interest
        df_temp = df_temp.drop(columns=['RankingDayNum', 'SystemName'], axis=1)
        # make list of ordinal rankings
        ord_temp = df_temp.OrdinalRank[np.bitwise_and(df_temp.TeamID==teamID, df_temp.Season==season)].values
        if not ord_temp: ord_temp = np.nan
        team_ords.append(ord_temp)

    return team_ords

n_test_games = df_sample_sub.shape[0]

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def seed_to_int(seed):
    '''Get just the digits from the seeding. Return as int'''
    s_int = int(seed[1:3])
    return s_int

print('Loading data for submission test')

# Make the seeding an integer
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label
print(df_seeds.head())

# -----------------Assign seed, advanced stats, and ordinal-----------------------------
seed_dict = []
adv_dict  = []
ord_dict  = []
# init team dict with years from 1985-2019 (range is not inclusive)
teams_by_year_dict = dict.fromkeys(np.arange(1985,2020))
# extract teams and years from tourney
for ii, row in tqdm.tqdm(df_sample_sub.iterrows(),total=n_test_games):
    year, t1, t2 = get_year_t1_t2(row.ID)
    teams_by_year_dict[year].append(t1)
    teams_by_year_dict[year].append(t2)
# find unique teams per year


# assign data for unique teams
for year, t1, t2 in teams:
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    t1_adv  = df_adv[(df_adv.TeamID == t1) & (df_adv.Season == year)].values[0]
    t2_adv  = df_adv[(df_adv.TeamID == t2) & (df_adv.Season == year)].values[0]
    t1_ords = assign_ords(t1,year)
    t2_ords = assign_ords(t2,year)
    T1_seed.append(t1_seed)
    T1_adv.append(t1_adv)
    T1_ord.append(t1_ords)
    T2_seed.append(t2_seed)
    T2_adv.append(t2_adv)
    T2_ord.append(t2_ords)

# ??? what am i doing here
T1_adv = [row[2:] for row in T1_adv]
T2_adv = [row[2:] for row in T2_adv]
T1_seed = np.reshape(T1_seed, [n_test_games,-1]).tolist()
T2_seed = np.reshape(T2_seed, [n_test_games, -1]).tolist()

X_pred = np.concatenate((T1_seed, T1_adv, T1_ord, T2_seed, T2_adv, T2_ord), axis=1)

df_pred = pd.DataFrame(X_pred)
df_pred.to_csv('Submission_Features.csv')
