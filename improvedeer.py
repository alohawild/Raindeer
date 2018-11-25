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
This program read the city file for Kaggle contest, a previous path answer, and
applies Monte Carlo and greedy improvement over a number of epochs. It then writes
the results out to a requested file name and log table.

Known flaw: Does not take in "prime carrot" for greedy and travel time seems 1/2
percent off.



"""
__author__ = 'michaelwild'
__copyright__ = "Copyright (C) 2018 Michael Wild and Corwin Tiers"
__license__ = "Apache License, Version 2.0"
__version__ = "0.0.1"
__credits__ = ["Michael Wild", "Corwin Tiers"]
__maintainer__ = "Michael Wild"
__email__ = "alohawild@mac.com"
__status__ = "Initial"


import numpy as np
import pandas as pd
from scipy import spatial
from time import process_time

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


def dist_city(start_city, end_city, dict):
    """
    Measures distance between two cities
    :param start_city: number of city (int)
    :param end_city: number of other city
    :param dict: The dictionary with the cites in it.
    :return: returns the euclidian distance
    """

    if (start_city < len(dict)) & (end_city < len(dict)):
        start_row = dict[start_city]
        end_row = dict[end_city]
        start_point = [[start_row['X'], start_row['Y']]]
        end_point = [[end_row['X'], end_row['Y']]]
        ed = calced(start_point, end_point)
    else:
        raise Exception('Bad city number')

    return ed

# ######################## Classes ##############################

class AlignData:
    """
    This is a helper class. It is used to align the data.

    """

#    def __init__(self):

    def alignnow(self, df, verbose=False):
        """
        Adds prime flag to initial data set
        :param df: data frame from initial load
        :param verbose: just for tracing
        :return: aligned data frame with primes identified
        """
        # Is city a prime?
        df['Prime'] = False
        # Adjust
        df['Prime'] = df[['CityId', 'Prime']].apply(lambda x:
                                                    True if isprime(x['CityId']) else False, axis=1)

        if verbose:
            print(df)

        return df


    def travel_time(self, dict_path, dict_cities, verbose=False):
        """
        My almost working travel time estimation. Seems to be off a bit from Kaggle.
        :param dict_path: dictionary version of path
        :param dict_cities: dictionary of cities aligned from above
        :param verbose: Print trace
        :return: total euclidian dist with carrot logic to add 10% (off by 1/2 % from Kaggle)
        """

        length = len(dict_path) -1
        previous_city = 0  # Always North Pole, zero
        count = 1  # Skip pole and first entry as it is not a step
        previous = False  # Used to track that previous step caused no carrot add-on
        ed_total = 0.0  # Total distance
        step = 1  # Step we are on

        while True:
            if count > length:
                break
            row = dict_path[count]
            city = row['Path']

            euclidean_distance = dist_city(previous_city, city, dict_cities)

            if previous:
                ed = euclidean_distance * 1.1
                previous = False
            else:
                ed = euclidean_distance
            ed_total = ed_total + ed
            if step == 10:
                step = 0
                if ~(dict_cities[count]['Prime']):
                    previous = True

            previous_city = city
            count = count + 1
            step = step + 1

        if verbose:
            print("Path =", ed_total)

        return ed_total


    def improve_random(self, dict_path, dict_cities, epochs, verbose=True):
        """
        The Monte Carlo greedy improvement. Runs an epoch of reading every entry in path and randomly trying to update
        it with a better path. Checks if path is better. If it is then replace.
        Works until about 75million paths and then value decreases.
        :param dict_path: The dictionary of path to improve
        :param dict_cities: Aligned dictionary of cities from above
        :param epochs: How many times to pass-thru the whole list
        :param verbose: Trace
        :return: improved path in dictionary form, list of logs from epoch
        """
        epoch_count = 1
        logs = []

        if verbose:
            print("Starting Epochs:", epochs)

        while epoch_count <= epochs:
            if verbose:
                print("Epoch #", epoch_count)
            count = 1  # Skip pole and first entry as it is not a step
            length = len(dict_path) - 3
            better = 0.0
            start_time = process_time()

            while True:
                if count > length:
                    break
                picked = np.random.randint(length) + 1  # 1..(end -1)
                # Get current impact to distance from current placement
                cur = dict_path[count]['Path']
                chg = dict_path[picked]['Path']

                p_cur = dict_path[count - 1]['Path']
                n_cur = dict_path[count + 1]['Path']
                p_chg = dict_path[picked - 1]['Path']
                n_chg = dict_path[picked + 1]['Path']

                # Get current impact to distance from current placement
                curr_dist = dist_city(p_cur, cur, dict_cities) + \
                            dist_city(cur, n_cur, dict_cities) + \
                            dist_city(p_chg, chg, dict_cities) + \
                            dist_city(chg, n_chg, dict_cities)
                # Get new impact to distance from change
                new_dist = dist_city(p_cur, chg, dict_cities) + \
                           dist_city(chg, n_cur, dict_cities) + \
                           dist_city(p_chg, cur, dict_cities) + \
                           dist_city(cur, n_chg, dict_cities)
                check_dist = curr_dist - new_dist

                if check_dist > 0.0:
                    better = better + check_dist
                    dict_path[count] = {'Path': chg}
                    dict_path[picked] = {'Path': cur}
                count = count + 1
            exc_time = run_time(start_time)
            if verbose:
                print("Better =", better)
                print("Epoch", epoch_count," Improvement:", better," Execution time:", exc_time)
            logs.append([epoch_count, better, exc_time])

            epoch_count = epoch_count + 1

        return dict_path, logs

# ######################## Start up ##############################


print("Improve Raindeer")
print(__version__, " ", __copyright__, " ", __license__)
begin_time = process_time()

name = input("What file to process (default=path.csv) ")
if name == "":
    name = "path.csv"
df_path = pd.read_csv(name)
dict_path = df_path.to_dict('records')

name_out = input("What file to output (default=deerimp.csv) ")
if name_out == "":
    name_out = "deerimp.csv"

name_log = input("What file to output log (default=deerimplog.txt) ")
if name_log == "":
    name_log = "deerimplog.txt"

name = input("How many runs (default=5) ")
if name == "":
    name = "5"
runs = int(name)

print("  Get City Data and align....")
start_time = process_time()
# get data and add prime column in a fine DF for easy of use
align = AlignData()
df_cities = align.alignnow(pd.read_csv('cities.csv'))
dict_cities = df_cities.to_dict('records')
print("  Execution time:", run_time(start_time))

print("  Travel Time...")
start_time = process_time()
initial_travel_cost = align.travel_time(dict_path, dict_cities)
print("  Execution time:", run_time(start_time))
print("  Travel cost:", initial_travel_cost)

print("...Random Improve...")
start_time = process_time()
dict_new, run_log = align.improve_random(dict_path, dict_cities, runs)
travel_cost = align.travel_time(dict_new, dict_cities)
print("...Execution time:", run_time(start_time))
print("...Travel cost:",travel_cost)

print("...Convert and Write...")
start_time = process_time()
df_final = pd.DataFrame.from_dict(dict_new)
df_final.to_csv(name_out, index=False)
with open(name_log, 'w') as file_handler:
    for item in run_log:
        file_handler.write("{}\n".format(item))
print("...Execution time:", run_time(start_time))

print(" ")
print("Run time:", run_time(begin_time))
print("...Finished...End of Line")