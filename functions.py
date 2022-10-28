import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import linear_model
import csv
from scipy.optimize import minimize_scalar

pd.options.mode.chained_assignment = None

def remove_rank(x):
    return x[x[:5].find(')')+2 if x[:5].find(')')>0 else 0:]

def weight(weeks_ago, game_count, max_games, decay=1/3):
    game_weight = game_count / max_games
    time_weight = 1/(weeks_ago**decay)
    return (game_weight*time_weight)**(1/2)

def get_schedule(soup, week, hfa = 3, decay = 1/3):
    
    items = []
    for item in soup.find_all('td'):
        items.append(item.get_text())

    weeks = items[0::10]
    dates = items[1::10]
    times = items[2::10]
    days = items[3::10]
    winners = items[4::10]
    winner_pts = items[5::10]
    location = items[6::10]
    loser = items[7::10]
    loser_pts = items[8::10]
    notes = items[9::10]

    zipped = list(zip(weeks, winners, winner_pts, location, loser, loser_pts))
    schedule_raw = pd.DataFrame(zipped, columns=['week', 'winner', 'winner_pts', 'location', 'loser','loser_pts',])

    schedule_raw['week'] = schedule_raw['week'].astype(str).astype(int)
    schedule_raw = schedule_raw[schedule_raw['week']<=week]
    schedule_raw['winner_pts'] = pd.to_numeric((schedule_raw['winner_pts'].astype(str)), errors='coerce')
    schedule_raw['loser_pts'] = pd.to_numeric((schedule_raw['loser_pts'].astype(str)), errors='coerce')
    schedule_raw = schedule_raw.dropna() 
    schedule_raw = schedule_raw[((schedule_raw['winner_pts'] + schedule_raw['loser_pts']) > 0)]

    conditions = [schedule_raw['location'] == '@', schedule_raw['location'] == '', schedule_raw['location'] == 'N']
    location_text = ['away', 'home', 'neutral']
    
    location_adjustment = [hfa, -hfa, 0]
    schedule_raw['location'] = np.select(conditions, location_text)
    schedule_raw['location_adjustment'] = np.select(conditions, location_adjustment)

    schedule_raw['winner'] = schedule_raw['winner'].apply(remove_rank)
    schedule_raw['loser'] = schedule_raw['loser'].apply(remove_rank)

    schedule_raw['margin'] = schedule_raw['winner_pts'] - schedule_raw['loser_pts']
    schedule_raw['true_margin'] = schedule_raw['margin'] + schedule_raw['location_adjustment']

    schedule_raw = schedule_raw.drop_duplicates()

    team_count = pd.concat([schedule_raw['winner'], schedule_raw['loser']], axis = 0, ignore_index=True).value_counts().to_frame()
    team_count['team'] = team_count.index
    team_count.columns = ['games', 'team']
    team_count.reset_index(inplace = True, drop = True)

    schedule_raw = pd.merge(schedule_raw, team_count, left_on = 'winner', right_on = 'team', how = 'left')
    schedule_raw = pd.merge(schedule_raw, team_count, left_on = 'loser', right_on = 'team', how = 'left', suffixes = ['_winner', '_loser'])
    schedule_raw['total_games'] = schedule_raw['games_winner'] + schedule_raw['games_loser']
    schedule_raw['weeks_ago'] = (max(schedule_raw['week'])+1) - schedule_raw['week']

    max_games = max(schedule_raw['total_games'])
    schedule_raw['weight'] = schedule_raw.apply(lambda x : weight(x['weeks_ago'], x['total_games'], max_games, decay), axis=1)
    weight_shift = 100/schedule_raw['weight'].sum()
    schedule_raw['weight'] = (schedule_raw['weight']*weight_shift)
    
    schedule = schedule_raw.drop(['team_loser', 'team_winner', 'games_loser', 'games_winner', 'location_adjustment', 'location', 'winner_pts', 'loser_pts','margin', 'weeks_ago', 'total_games'], axis=1)
    return schedule

def get_initial(schedule):

    extras = schedule[['true_margin', 'weight']]
    transform = schedule.drop(['true_margin', 'weight'], axis = 1)
    winners = pd.get_dummies(transform, prefix='',prefix_sep = '', columns=['winner'])
    losers = pd.get_dummies(transform, prefix='',prefix_sep = '', columns=['loser'])*-1
    combined = pd.concat([winners,losers], axis = 1)
    

    x = combined.groupby(by = combined.columns, axis=1).sum().drop(['loser','winner'], axis = 1)
    a = pd.DataFrame([1] * x.shape[0])
    x = pd.concat([a, x.reset_index(drop=True)], axis = 1).to_numpy()
    y = extras['true_margin'].to_numpy()
    w = np.sqrt(extras['weight'].to_numpy())

    regr = linear_model.LinearRegression(
       fit_intercept = False, copy_X = True, n_jobs = -1
    ).fit(x,y, sample_weight = w)

    result=regr.coef_
    result = result[1:] - result[1:].mean()
    teams = combined.groupby(by = combined.columns, axis=1).sum().drop(['loser','winner'], axis = 1).columns.values
    d = {'teams': teams, 'coefs': result}
    r1_ratings = pd.DataFrame(data = d)

    r1_ratings.sort_values(by=['coefs'], inplace=True, ascending=False)

    schedule.set_index('winner', inplace=True, drop = False)
    r1_ratings.set_index('teams', inplace=True, drop = False)
    with_winner = schedule.join(r1_ratings, how='left').set_index('loser', drop = False)

    with_ratings = with_winner.join(r1_ratings, how = 'left', lsuffix='_winner', rsuffix='_loser').drop(['teams_winner', 'teams_loser'], axis = 1)
    with_ratings.reset_index(inplace = True, drop = True)
    return with_ratings

def get_rating(subject, initial):
    with_ratings = initial[['winner', 'loser', 'true_margin', 'weight','coefs_winner', 'coefs_loser']]
    
    winner_subject = with_ratings.loc[(with_ratings['winner']==subject)] 
    loser_subject = with_ratings.loc[(with_ratings['loser']==subject)] 
    loser_subject['true_margin'] = -loser_subject['true_margin']

    winner_subject.columns = ['team1', 'team2', 'true_margin', 'weight','rating_team1', 'rating_team2']
    loser_subject.columns = ['team2', 'team1', 'true_margin', 'weight','rating_team2', 'rating_team1']

    subject_set = pd.concat([winner_subject, loser_subject], ignore_index=True)
    subject_set['y'] = subject_set['true_margin']+subject_set['rating_team2']
    subject_set['x'] = 1
    x = subject_set['x'].to_numpy()
    y = subject_set['y'].to_numpy()
    w = subject_set['weight'].to_numpy()

    rating = np.dot(y,w)/np.dot(w,x)
    return rating

def get_teams(schedule):
    return pd.concat([schedule['winner'], schedule['loser']], axis = 0, ignore_index=True).unique().tolist()

def get_ratings(schedule):
    initial = get_initial(schedule)
    teams = get_teams(schedule)
    output_list = list(map(lambda x: get_rating(x, initial), teams))
    ratings = pd.DataFrame(list(zip(teams, output_list)), columns=['teams', 'ratings'])
    return ratings.sort_values("ratings", axis = 0, ascending = False)

def get_error(schedule, ratings):
    schedule.drop(['week'], inplace = True, axis = 1)
    ratings.sort_values(by=['ratings'], inplace=True, ascending=False)

    schedule.set_index('winner', inplace=True, drop = False)
    ratings.set_index('teams', inplace=True, drop = False)
    with_winner = schedule.join(ratings, how='left').set_index('loser', drop = False)

    with_ratings = with_winner.join(ratings, how = 'left', lsuffix='_winner', rsuffix='_loser').drop(['teams_winner', 'teams_loser'], axis = 1)
    with_ratings.reset_index(inplace = True, drop = True)
    with_ratings['error'] = (with_ratings['true_margin'] - (with_ratings['ratings_winner'] - with_ratings['ratings_loser']))**2

    with_ratings.drop(['true_margin','ratings_winner', 'ratings_loser'], inplace = True, axis = 1)

    with_ratings2 = with_ratings.copy()

    with_ratings.columns = ['team1', 'team2', 'weight', 'error']
    with_ratings2.columns = ['team2', 'team1', 'weight', 'error']

    error_set = (pd.concat([with_ratings, with_ratings2], ignore_index=True)).drop(['team2'], axis = 1)
    ##need to factor in weight
    error_sum = pd.DataFrame(error_set.groupby(by = 'team1', axis=0).apply(lambda x: (x.weight*x.error).sum()))
    error_count = error_set.drop(['weight'], axis = 1).groupby(by = 'team1', axis=0).count()


    error_total = error_sum.join(error_count, lsuffix = "r", rsuffix = "l")
    error_total.reset_index(inplace = True)
    error_total.columns = ['team', 'error', 'games']

    error_total['rmse'] = (error_total['error']/error_total['games'])**0.5
    error_total['psudo_sd'] = ((error_total['rmse']*error_total['games'])+9*12)/(error_total['games']+12)
    error = error_total.drop(['error','games','rmse'], axis = 1)
    return error

def n_max(given_list, k, default = 0):
    length = len(given_list)+1
    if k >= length:
        return default
    given_list.sort()
    return given_list[-k]

def n_min(given_list, k, default = 0):
    length = len(given_list)+1
    if k >= length:
        return default
    given_list.sort()
    return given_list[-(length - k)]

class Team:
    def __init__(self, name, victory, defeat):
        self.name = name
        self.victory = victory
        self.defeat = defeat
        
    def get_wins(self):
        return len(self.victory)
    
    def get_victories(self):
        return self.victory
    
    def get_losses(self):
        return len(self.defeat)
    
    def get_defeats(self):
        return self.defeat

    def __repr__(self):
        return f"Team: {self.name}, Victory: {self.victory}, Defeat: {self.defeat}"
    
    
def get_worster(year, week = 20):
    url1 = 'https://www.sports-reference.com/cfb/years/'
    url2 = '-schedule.html'
    
    url = url1+str(year)+url2
    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    schedule = get_schedule(soup, week = week)
    team_dict = {}
    for i in get_teams(schedule):
        victory = schedule.loc[(schedule['winner']==i)]['loser'].tolist()
        defeat = schedule.loc[(schedule['loser']==i)]['winner'].tolist()
        team_dict[i] = Team(i, victory, defeat)
    team_list = get_teams(schedule)

    team_df = pd.DataFrame(team_list, columns = ['team'])
    team_df = team_df.assign(wins = list(map(lambda x: team_dict[x].get_wins(), get_teams(schedule))))
    team_df = team_df.assign(losses = list(map(lambda x: team_dict[x].get_losses(), get_teams(schedule))))
    team_df = team_df.assign(wins_from_1best = list(map(lambda y: max(list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),default=0), get_teams(schedule))))
    team_df = team_df.assign(wins_from_1worst = list(map(lambda y: min(list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),default=0), get_teams(schedule))))

    team_df = (pd.DataFrame(team_list, columns = ['team'])
     .assign(wins = 
             list(map(lambda x: team_dict[x].get_wins(), get_teams(schedule))))
     .assign(losses = 
             list(map(lambda x: team_dict[x].get_losses(), get_teams(schedule))))
     .assign(wins_from_1best = 
             list(map(lambda y: max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_1worst = 
             list(map(lambda y: min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_2best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),2,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_2worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),2,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_3best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),3,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_3worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),3,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_4best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),4,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_4worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),4,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_5best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),5,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_5worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),5,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_6best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),6,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_6worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),6,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_7best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),7,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_7worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),7,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_8best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),8,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_8worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),8,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_9best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),9,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_9worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),9,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_10best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),10,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_10worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),10,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_11best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),11,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_11worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),11,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_12best = 
             list(map(lambda y: n_max(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_victories())),12,default=0)
                      , get_teams(schedule)
                     )))
     .assign(wins_from_12worst = 
             list(map(lambda y: n_min(
                 list(map(lambda x: team_dict[x].get_wins(), team_dict[y].get_defeats())),12,default=0)
                      , get_teams(schedule)
                     )))
    )


    team_df = (
        team_df.sort_values(by = (['wins','losses',
                                'wins_from_1best', 'wins_from_1worst',
                                'wins_from_2best', 'wins_from_2worst',
                                'wins_from_3best', 'wins_from_3worst',
                                'wins_from_4best', 'wins_from_4worst',
                                'wins_from_5best', 'wins_from_5worst',
                                'wins_from_6best', 'wins_from_6worst',
                                'wins_from_7best', 'wins_from_7worst',
                                'wins_from_8best', 'wins_from_8worst',
                                'wins_from_9best', 'wins_from_9worst',
                                'wins_from_10best', 'wins_from_10worst',
                                'wins_from_11best', 'wins_from_11worst',
                                'wins_from_12best', 'wins_from_12worst']),
                         axis=0, ascending=([False, True, 
                                             False, False, 
                                             False, False, 
                                             False, False, 
                                             False, False,
                                             False, False, 
                                             False, False, 
                                             False, False, 
                                             False, False,
                                             False, False, 
                                             False, False, 
                                             False, False, 
                                             False, False]) )
    )
    return team_df

def combined(ratings, error):
    error.set_index('team', drop = False, inplace = True) 
    rating_error = ratings.join(error, how = 'left', lsuffix='_l', rsuffix='_r').drop(['teams','team'], axis = 1).reset_index()
    rating_error.columns = ['team','rating','psudo_sd']
    return rating_error

def error_hfa(x, url, week):
    hfa = x 
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    schedule = get_schedule(soup, hfa = hfa, decay = 0, week = week)
    ratings = get_ratings(schedule)
    return get_error(schedule, ratings)['psudo_sd'].sum()

def error_decay(x, hfa, url, week):
    decay = x 
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    schedule = get_schedule(soup, hfa = hfa, decay = x, week = week)
    ratings = get_ratings(schedule)
    return get_error(schedule, ratings)['psudo_sd'].sum()

def get_underwood(year, week = 20, hyperparameters = {}):
    
    url1 = 'https://www.sports-reference.com/cfb/years/'
    url2 = '-schedule.html'
    
    url = url1+str(year)+url2
    
    if "home_adv" in hyperparameters:
        hfa = hyperparameters["home_adv"]
    else: 
        result = minimize_scalar(error_hfa, args=(url, week), method='brent')
        hfa = result['x'] 
        
    if "decay_rate" in hyperparameters:
        decay = hyperparameters["decay_rate"]
    else: 
        result = minimize_scalar(error_decay, bounds=(0,1), args=(hfa, url, week), method='bounded')
        decay = result['x']
        
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    schedule = get_schedule(soup, hfa = hfa, decay = decay, week = week)
    ratings = get_ratings(schedule)
    error = get_error(schedule, ratings)

    return combined(ratings, error)


def get_graph_data(year, week = 20, hyperparameters = {}):

    url1 = 'https://www.sports-reference.com/cfb/years/'
    url2 = '-schedule.html'

    url = url1+str(year)+url2

    if "home_adv" in hyperparameters:
        hfa = hyperparameters["home_adv"]
    else: 
        result = minimize_scalar(error_hfa, args=(url, week), method='brent')
        hfa = result['x'] 

    if "decay_rate" in hyperparameters:
        decay = hyperparameters["decay_rate"]
    else: 
        result = minimize_scalar(error_decay, bounds=(0,1), args=(hfa, url, week), method='bounded')
        decay = result['x']

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    schedule = get_schedule(soup, hfa = hfa, decay = decay, week = week)
    initial = get_initial(schedule)

    winners = initial[['winner', 'coefs_winner']]
    winners.columns = ['teams', 'initial_rating']
    losers = initial[['loser', 'coefs_loser']]
    losers.columns = ['teams', 'initial_rating']

    initial_ratings = (
        winners.append(losers, ignore_index=True, verify_integrity=False)
     .drop_duplicates()
     .sort_values(by = 'initial_rating', ascending = False)
     .set_index('teams', drop = False))
    ratings = get_ratings(schedule)

    schedule.set_index('winner', inplace=True, drop = False)
    with_winner = schedule.join(initial_ratings, how='left').set_index('loser', drop = False)

    with_ratings = with_winner.join(initial_ratings, how = 'left', lsuffix='_winner', rsuffix='_loser').drop(['teams_winner', 'teams_loser'], axis = 1)
    with_ratings.reset_index(inplace = True, drop = True)

    with_ratings2 = with_ratings.copy()
    with_ratings2['true_margin'] = -with_ratings['true_margin']

    with_ratings.columns = ['week', 'team1', 'team2', 'true_margin', 'weight', 'ratings_team1', 'ratings_team2']
    with_ratings2.columns = ['week', 'team2', 'team1', 'true_margin', 'weight', 'ratings_team2', 'ratings_team1']

    error_set = (pd.concat([with_ratings, with_ratings2], ignore_index=True))

    graph_data = pd.DataFrame(columns =['week','team1', 'team2', 'ratings_team1', 'error'])

    for j in get_teams(schedule):
        subject = j
        game_set = error_set[error_set['team1']==subject]
        proposed = round((ratings[ratings['teams']==subject]['ratings'].values[0]),0)
        rating_range = np.arange(proposed - 8, proposed + 8)

        for i in rating_range:

            game_set['ratings_team1'] = i
            game_set['error'] = (game_set['weight']*(game_set['true_margin'] - 
                                                     (game_set['ratings_team1']-game_set['ratings_team2']))**2)
            temp = game_set.drop(['true_margin', 'weight', 'ratings_team2'], axis = 1)
            graph_data = graph_data.append(temp, ignore_index = True)
            graph_data['error'] = round(graph_data['error'],2)

    graph_data.columns=['week','team', 'opponent', 'hypothetical rating', 'error']

    return graph_data
