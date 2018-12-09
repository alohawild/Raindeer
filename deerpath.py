#!/usr/bin/env python
"""
    Copyright 2018 by Michael Wild (alohawild) and Corwin Tiers

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

==============================================================================


"""
__author__ = 'michaelwild'
__copyright__ = "Copyright (C) 2018 Michael Wild and Corwin Tiers"
__license__ = "Apache License, Version 2.0"
__version__ = "0.1.2"
__credits__ = ["Michael Wild", "Corwin Tiers"]
__maintainer__ = "Michael Wild"
__email__ = "alohawild@mac.com"
__status__ = "Initial"


import numpy as np
import pandas as pd
from scipy import spatial
from time import process_time
from sklearn.preprocessing import MaxAbsScaler
import sys


# ######################## shared ##############################


def isprime(x):
    """
    This is the old way of doing this...But easy
    """
    if x < 2:
        return False  # Number 1 or less is
    if x == 2:  # Number is 2
        return True
    if x % 2 == 0:  # Number is even
        return False
    if x > 2:  # The rest
        for i in range(3, int(np.sqrt(x))+1, 2):
            if x % i == 0:
                return False
        return True


def calced(start_point, end_point):
    """
    Calculate and return as a single number the distance between two points
    I used this instead of writing it out to avoid a typo that would be impossible to find
    :param start_point: [[x, y]] array of points
    :param end_point: [[x, y]] array of points
    :return: real number of distance
    """
    euclidean_distance = spatial.distance.cdist(start_point, end_point, "euclidean")
    euclidean_distance = euclidean_distance[0]  # Force to list of one number
    euclidean_distance = euclidean_distance[0]  # Force to number
    return euclidean_distance


def run_time(start):
    """
    Just takes in previous time and returns elapsed time
    :param start: start time
    :return: elapsed time
    """
    return process_time() - start


def snake(number, length=10):

    if length <= 0:
        raise Exception('Bad length number')

    row = int(number / length)
    if even(row):
        snake_value = number
    else:
        snake_value = (length - (number % length))+ row*length -1

    return snake_value


def even(number):
    return (number % 2) == 0


def dist_city(start_city, end_city, dict):
    """
    Measures distance between two cities
    :param start_city: number of city (int)
    :param end_city: number of other city
    :param dict: The dictionary with the cites in it.
    :return: returns the euclidian distance
    """

    if (start_city < len(dict)) & (end_city < len(dict)):
        start_point = [dict[start_city]['Loc']]
        end_point = [dict[end_city]['Loc']]
        ed = calced(start_point, end_point)
    else:
        raise Exception('Bad city number')

    return ed


def get_cites(dict, search):

    list_cities = []

    for i in range(0, len(dict)):
        if len(list_cities) >= search:
            break
        if dict[i]['Used'] <= 0:
            list_cities.append(dict[i]['CityId'])

    return list_cities

def update_cites(dict, city, verbose=False):

    successful = False
    for i in range(0, len(dict)):
        if verbose:
            print("Place:", dict[i], "City:", city)
        if int(dict[i]['CityId']) == int(city):  # Yes, that was a problem...hummm
            row = dict[i]
            dict[i] = {'DistF': row['DistF'], 'CityId': row['CityId'], 'Used': 1}
            if verbose:
                print("Updated:", dict[i])
            successful = True
            break

    if verbose:
        print("sucessful:", successful)

    if successful == False:
        fail = 'Bad City update: %i' % int(city)
        raise Exception(fail)

    return

def repeat_cites(dict, city, verbose=False):

    successful = False
    for i in range(0, len(dict)):
        if verbose:
            print("Path:", dict[i], "City:", city)
        if int(dict[i]['Path']) == int(city):  # Yes, that was a problem...hummm
            successful = True
            break

    if verbose:
        print("sucessful:", successful)

    return successful

# ######################## Classes ##############################

class LoadCities:

    default_file = 'cities.csv'
    max_X = 0.0
    max_Y = 0.0
    focus = 1
    North_Pole = 0

    def __init__(self, focus=10):

        self.focus = focus

        return

    def load_file(self, filename=default_file):

        df_cities = self.alignnow(pd.read_csv(filename))

        return df_cities

    def alignnow(self, df, verbose=False):

        self.max_X = df['X'].max()
        self.max_Y = df['Y'].max()

        # Is city a prime?
        df['Prime'] = False
        # Adjust
        df['Prime'] = df[['CityId', 'Prime']].apply(lambda x:
                                                    True if isprime(x['CityId']) else False, axis=1)
        df['Loc'] = df.apply(lambda row: [row.X, row.Y], axis=1)

        df['X_s'] = df['X']
        df['Y_s'] = df['Y']

        # Cool kids scale to a standard
        sct = MaxAbsScaler()
        scale_columns = ['X_s', 'Y_s']
        df_s = sct.fit_transform(df[scale_columns])
        df_s = pd.DataFrame(df_s, columns=scale_columns, index=df.index.get_values())
        # add the scaled columns back into the dataframe
        df[scale_columns] = df_s
        # Now create a focus based on it.

        df['X_s'] = df.apply(lambda row: int(row.X_s * 10), axis=1)
        df['Y_s'] = df.apply(lambda row: int(row.Y_s * 10), axis=1)
        df['Focus'] = df.apply(lambda row: (int(row.X_s) + int(row.Y_s*10)), axis=1)
        df['Focus'] = df[['Focus']].apply(lambda x: 99 if x['Focus'] > 99 else x['Focus'], axis=1) # One edge case
        df = df.drop(['X_s', 'Y_s'], axis=1)

        df.sort_values(by=['Focus', "CityId"], axis=0,
                              ascending=True, inplace=True, kind='quicksort', na_position='last')
        dict_focus = {}
        for i in range(0, 100):
            search_string = 'Focus == %i' % i
            df_focus = df.query(search_string)
            # Decided not to copy list of cities into this: dfList = df_focus['CityId'].tolist()
            if len(df_focus) > 0:
                cent_X = df_focus['X'].sum() / len(df_focus)
                cent_Y = df_focus['Y'].sum() / len(df_focus)
            else:
                cent_X = -1
                cent_Y = -1
            dict_focus[i] = {'Focus': i, 'Count': len(df_focus), 'Centroid': [cent_X, cent_Y]}
        df = df.drop(['X', 'Y'], axis=1)

        if verbose:
            print(df)
        df.sort_values(by=['CityId'], axis=0,
                              ascending=True, inplace=True, kind='quicksort', na_position='last')
        return df, dict_focus

    def get_cities(self):
        return self.cities


    def betterpath(self, dict_cities, dict_focus, df, search_value=500, verbose=True, test=False):

        dict_path = {}

        North_pole = dict_cities[self.North_Pole]

        dict_path[0] = {"Path": North_pole['CityId']}
        dict_path[1] = {"Path": North_pole['CityId']}

        already_done = []
        i = dict_cities[self.North_Pole]['Focus']
        step = 1
        place = 1

        while True:

            if i>99:
                i = 0
            if i in already_done:
                break
            if step>10:
                step = 0

            if verbose:
                print("Focus area =",i)

            # get all cities in Focus
            # add used flag and sort by distance from centroid
            search_string = 'Focus == %i' % snake(i)  # Use the snake
            df_focus = df.query(search_string)
            cent = [dict_focus[i]['Centroid']]
            df_focus['DistF'] = df_focus.apply(lambda row: calced([row.Loc], cent), axis=1)
            df_focus['Used'] = 0
            # Create two dictionaries for prime and not prime
            # Dictionaries are fast and pre-sorted
            df_prime = df_focus.query('Prime == True')
            df_not = df_focus.query('Prime != True & CityId != 0') # Do not include Northpole again
            df_prime = df_prime.filter(['CityId', 'DistF', 'Used'], axis=1)
            df_prime.sort_values(by=['DistF', 'CityId'], axis=0,
                           ascending=True, inplace=True, kind='quicksort', na_position='last')
            dict_prime = df_prime.to_dict('records')
            df_not = df_not.filter(['CityId', 'DistF', 'Used'], axis=1)
            df_not.sort_values(by=['DistF', 'CityId'], axis=0,
                           ascending=True, inplace=True, kind='quicksort', na_position='last')
            dict_not = df_not.to_dict('records')

            while True:

                # handle step
                prime = False
                select_cities = []
                if step == 10:
                    prime = True

                # Get cites to add to list, just the one in greedy process
                if prime:
                    select_cities = get_cites(dict_prime, search_value)
                    if len(select_cities) <= 0:
                        select_cities = get_cites(dict_not, search_value)
                        prime = False
                else:
                    select_cities = get_cites(dict_not, search_value)
                    if len(select_cities) <= 0:
                        select_cities = get_cites(dict_prime, search_value)
                        prime = True

                if test:
                    print("     Cities:", select_cities)

                if len(select_cities) <= 0:  # If we still have nothing then break.
                    break

                best_dist = sys.maxsize
                previous_city = dict_path[place - 1]['Path']
                next_city = dict_path[place]['Path']
                best_city = sys.maxsize  # We already know we have more than one

                # Selection logic goes here
                for city in select_cities:
                    # greedy without much more regard
                    new_dist = dist_city(int(previous_city), int(city), dict_cities)
                    new_dist = new_dist + dist_city(int(city), int(next_city), dict_cities)
                    if new_dist < best_dist:
                        best_city = city
                        best_dist = new_dist

                if test:
                    print("     Best:", best_city)

                if test:
                    if repeat_cites(dict_path, best_city):
                        print("     Ugh!", best_city)
                        fail = 'Repeat City %i' % int(best_city)
                        raise Exception(fail)

                dict_path[place+1] = dict_path[place]
                dict_path[place] = {"Path": int(best_city)}

                if test:
                    print("     Added:", dict_path[place])

                if prime:
                    update_cites(dict_prime, best_city)
                else:
                    update_cites(dict_not, best_city)

                # Next step and next place
                step = step + 1
                if step > 10:
                    step = 1
                place = place + 1

            already_done.append(i)
            i = i + 1


        return dict_path


# ######################## Start up ##############################


print("Raindeer Graph")
print(__version__, " ", __copyright__, " ", __license__)

name_out = ""
#name_out = input("What file to output (default=deerpath.csv) ")
if name_out == "":
    name_out = "deerpath.csv"

begin_time = process_time()

print("  Get City Data and align....")
start_time = process_time()
# get data and add prime column in a fine DF for easy of use
load_cities = LoadCities()
df_cities, dict_focus = load_cities.load_file()
dict_cities = df_cities.to_dict('records')
print("  Execution time:", run_time(start_time))

print("  Create Path....")
start_time = process_time()
# Make path
dict_path = load_cities.betterpath(dict_cities, dict_focus, df_cities)

print("  Execution time:", run_time(start_time))

print("...Convert and Write...")
start_time = process_time()
df_final = pd.DataFrame.from_dict(dict_path,orient='index')
df_final.to_csv(name_out, index=False)
print("...Execution time:", run_time(start_time))

print(" ")
print("Run time:", run_time(begin_time))
print("...Finished...End of Line")