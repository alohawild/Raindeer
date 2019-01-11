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
__version__ = "1.2.2"
__credits__ = ["Michael Wild", "Corwin Tiers"]
__maintainer__ = "Michael Wild"
__email__ = "alohawild@mac.com"
__status__ = "Initial"


import numpy as np
import pandas as pd
from scipy import spatial
from time import process_time

import itertools
import sys
import csv


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
        start_point = [dict[start_city]['Loc']]
        end_point = [dict[end_city]['Loc']]
        ed = calced(start_point, end_point)
    else:
        err = 'Bad City Numbers: %i %i' % (start_city, end_city)
        raise Exception(err)

    return ed

def dist_list_city(lt, dict):
    """

    """
    ed = 0
    if len(lt) > 1:
        previous_city = lt[0]
        for i in range(1, len(lt)):
            start_point = [dict[previous_city]['Loc']]
            end_point = [dict[lt[i]]['Loc']]
            ed = ed + calced(start_point, end_point)
            previous_city = lt[i]

    return ed

def fix_path(df, dict_cities):

    # move back to dictionary to allow for fast ordered processing
    dict_path = df.to_dict('records')
    # create new dictionary with distance
    dict_new = {}
    dict_new[0] = {'Step': 0, 'Path': int(dict_path[0]['Path']), 'Dist': 0.0}
    previous_city = int(dict_path[0]['Path'])
    for i in range(1,len(dict_path)):
        ed_dist = dist_city(previous_city, int(dict_path[i]['Path']), dict_cities)
        dict_new[i] = {'Step': i, 'Path': int(dict_path[i]['Path']), 'Dist': ed_dist}
        previous_city = int(dict_path[i]['Path'])
    # move to data frame with new columns with just one key!
    columns = ['Step', 'Path', 'Dist']
    index = ['Step']
    df_new = pd.DataFrame(index=index, columns=columns)
    df_new = pd.DataFrame.from_dict(dict_new, orient='index')

    return dict_new, df_new  # Always useful to have both


def apply_path(dict_path, dict_cities, delete_list, add_list, add_check, verbose=False):

    place = 1
    dict = {}
    dict[0] = dict_path[0]  # copy North Pole
    previous_city = 0

    for i in range(1, len(dict_path) - 1):
        city = dict_path[i]['Path']
        if city in delete_list:
            # skip adding a city that is to be skipped
            if verbose:
                print("Delete:", city)
        else:
            # add back in
            ed_dist = dist_city(previous_city, city, dict_cities)
            dict[place] = {'Step': place, 'Path': city, 'Dist': ed_dist}
            place = place + 1
            previous_city = city
        if city in add_check:
            for row in add_list:
                if city == row[0]:
                    for new_city in row[1]:
                        ed_dist = dist_city(previous_city, new_city, dict_cities)
                        dict[place] = {'Step': place, 'Path': new_city, 'Dist': ed_dist}
                        place = place + 1
                        previous_city = new_city
                        if verbose:
                            print("Insert:", new_city, " after", city)

    ed_dist = dist_city(previous_city, dict_path[(len(dict_path) - 1)]['Path'], dict_cities)
    dict[place] = {'Step': place, 'Path': dict_path[(len(dict_path) - 1)]['Path'],
                   'Dist': ed_dist}

    # move to data frame with new columns with just one key!
    columns = ['Step', 'Path', 'Dist']
    index = ['Step']
    df_new = pd.DataFrame(index=index, columns=columns)
    df_new = pd.DataFrame.from_dict(dict, orient='index')

    return dict, df_new

# ######################## Classes ##############################

class LoadCities:

    default_file = 'cities.csv'

    North_Pole = 0


    def __init__(self):

        return

    def load_file(self, filename=default_file):

        df_cities = self.alignnow(pd.read_csv(filename))

        return df_cities

    def alignnow(self, df, verbose=False):

        # Is city a prime?
        df['Prime'] = False
        # Adjust
        df['Prime'] = df[['CityId', 'Prime']].apply(lambda x:
                                                    True if isprime(x['CityId']) else False, axis=1)
        df['Loc'] = df.apply(lambda row: [row.X, row.Y], axis=1)

        #df = df.drop(['X', 'Y'], axis=1)

        if verbose:
            print(df)

        df.sort_values(by=['CityId'], axis=0,
                              ascending=True, inplace=True, kind='quicksort', na_position='last')
        return df

    def travel_time(self, dict_path, dict_cities, verbose=False):
        """
        My almost working travel time estimation. Seems to be off a bit from Kaggle.
        :param dict_path: dictionary version of path
        :param dict_cities: dictionary of cities aligned from above
        :param verbose: Print trace
        :return: total euclidian dist with carrot logic to add 10% (off by 1/2 % from Kaggle)
        """

        length = len(dict_path) - 1
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

            #if step == 10 & ~(isprime(previous_city)):
            if previous:
                ed = euclidean_distance * 1.1
                previous = False

            else:
                ed = euclidean_distance
            ed_total = ed_total + ed
            if step == 10:
                step = 0
                if ~(isprime(city)):
                    previous = True

            previous_city = city
            count = count + 1
            step = step + 1

        if verbose:
            print("Path =", ed_total)

        return ed_total

    def absolute_dist(self, dict_path, dict_cities, verbose=False):
        """

        """

        dt_dist = {}
        dt_count = 0
        dt_dist[dt_count] = {'Path': 0, 'Length:' : 0}
        previous_city = 0  # North Pole
        ed = 0.0

        for i in range(1,len(dict_path)):
            dt_count = dt_count + 1
            row = dict_path[i]
            city = row['Path']
            euclidean_distance = dist_city(previous_city, city, dict_cities)
            dt_dist[dt_count] = {'Path': city, 'Length': euclidean_distance}
            ed = ed + euclidean_distance
            previous_city = city

        return ed, dt_dist

    def worst_path(self, df, dt, number_bad=50, verbose=False):
        """

        """

        x = list(df.nlargest(number_bad, 'Dist')['Step'])
        if verbose:
            print("Longest path cities:",x)
        for step in x:
            if verbose:
                print(dt[step])

        return x

    def better_scan(self, dict_path, dict_cities, number_scan=5, limit_scan=1.0, verbose=False, test_run=False):


        i = 1
        looping = True
        improvement_total = 0.0

        while looping:

            scan_list = []

            if i >= (len(dict_path)-1):  # stop if we are over
                break
            if (i + number_scan) >= (len(dict_path)-1):  # if not quite over then force last five
                i = len(dict_path) - number_scan - 1
                loop = False

            if verbose:
                print("Checking:", i, " Improvement so far:", improvement_total)

            for k in range(i, i + number_scan):  # Just copy it
                scan_list.append(dict_path[k]['Path'])
            # must have before and after to check base dist of group
            before_city = dict_path[i - 1]['Path']
            # possible that the last one is the last one
            if (i + number_scan) > (len(dict_path)-1):
                after_city = scan_list[-1]
            else:
                after_city = dict_path[i + number_scan]['Path']
            #  Calculate the dist of the proceeding and current
            before_dist = dist_city(before_city, scan_list[0], dict_cities)
            after_dist = dist_city(after_city, scan_list[-1], dict_cities)
            best_dist = dist_list_city(scan_list, dict_cities) + before_dist + after_dist
            org_dist = best_dist
            best_list = scan_list
            replace = False
            if test_run:
                print("Start:", before_city, after_city, org_dist)

            # loop thru all the combinations and see if we can tighten the group
            for choice in itertools.permutations(scan_list):
                before_dist = dist_city(before_city, choice[0], dict_cities)
                after_dist = dist_city(after_city, choice[-1], dict_cities)
                new_dist = dist_list_city(choice, dict_cities) + before_dist + after_dist
                if test_run:
                    print("Dist:", before_dist, after_dist, dist_list_city(choice, dict_cities))
                    print("Best so far:", best_dist, best_list, "New:", new_dist, choice)

                if best_dist > new_dist:
                    replace = True
                    best_list = choice
                    best_dist = new_dist
            if test_run:
                print("Results:", scan_list, replace, best_list, (org_dist - best_dist))
            #  It is best to force change with a good improvement thus the limit_scan
            if replace & ((org_dist-best_dist) > limit_scan):
                if verbose:
                    improvement = org_dist-best_dist
                    print("Improvement!", improvement)
                    improvement_total = improvement_total + improvement
                j = 0
                for k in range(i, i + number_scan):
                    dict_path[k] = {'Step': k, 'Path': best_list[j], 'Dist': 0.0}
                    if test_run:
                        print("Updated:", dict_path[k])
                    j = j + 1
            i = i + number_scan
        if verbose:
            print("Total improvement (no carrots):", improvement_total)

        return dict_path

    def improve_scan(self, dict_path, dict_cities, worst_list, verbose=False):

        for step in worst_list:

            if verbose:
                print("Fix:", dict_path[step])
                better = False
                best_dist = sys.maxsize

            for i in range(1,len(dict_path)):

                if abs(step - i) < 3:
                    continue  # Skip the same city and same next city
                if i in worst_list:
                    continue  # Skip ones we already have on the list
                if dict_path[i]['Path'] == self.North_Pole:
                    picked = i - 1  # we can't move our end point.
                else:
                    picked = i
                if dict_path[step]['Path'] == self.North_Pole:
                    step = step - 1  # we can't move our end point.


                cur = dict_path[step]['Path']
                chg = dict_path[picked]['Path']

                p_cur = dict_path[step - 1]['Path']
                n_cur = dict_path[step + 1]['Path']
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
                    better = True
                    if check_dist < best_dist:
                        best_picked = picked
                        best_chg = chg
                        best_cur = cur
                        if verbose:
                            print("Better:", check_dist, " for", step)
                    dict_path[step] = {'Step': step, 'Path': chg, 'Dist': 0.0}
                    dict_path[picked] = {'Step': picked, 'Path': cur, 'Dist': 0.0}
                    if verbose:
                        print(dict_path[step],dict_path[picked])
                    break
            if better:
                dict_path[step] = {'Step': step, 'Path': best_chg, 'Dist': 0.0}
                dict_path[best_picked] = {'Step': best_picked, 'Path': best_cur, 'Dist': 0.0}
                if verbose:
                    print("Switch:",dict_path[step], dict_path[picked])
            else:
                if verbose:
                    print("Failed to improve")
                shake = []
                if step < 6:
                    for i in range(1,6):
                        shake.append(i)

                elif (step+5) > len(dict_path):
                    last_place = len(dict_path) - 6
                    for i in range(last_place, len(dict_path)):
                        shake.append(i)
                else:
                    for i in range(step, step+5):
                        shake.append(i)
                print(shake,step)

        return dict_path


    def split_scan(self, dict_path, dict_cities, worst_list, df, already_done=[], adjust=4, verbose=False):

        delete_list = []
        add_list = []
        add_check = []

        for step in worst_list:

            if dict_path[step] in delete_list:  # possible that we processed it
                continue

            c_city = dict_path[step]['Path']
            p_city = dict_path[step-1]['Path']

            dist_long = dist_city(c_city, p_city, dict_cities)

            X1 = dict_cities[p_city]['X']
            Y1 = dict_cities[p_city]['Y']
            X2 = dict_cities[c_city]['X']
            Y2 = dict_cities[c_city]['Y']

            XN = (X1+X2) / 2.0
            YN = (Y1+Y2) / 2.0

            dist_hunt = dist_long / adjust

            if verbose:
                msg = "Dist: %f Previous: %i [%f , %f] Current: %i [%f , %f] Mid: [%f , %f]" % \
                      (dist_long, p_city, X1, Y1, c_city, X2, Y2, XN, YN)
                print(msg)
            XL = XN - dist_hunt
            if XL < 0.0:
                XL = 0.0
            YL = YN - dist_hunt
            if YL < 0.0:
                YL = 0.0
            XH = XN + dist_hunt
            YH = YN + dist_hunt
            if verbose:
                msg = "Hunt: X between %f and %f] Y Between %f , %f" % (XL, XH, YL, YH)
                print(msg)
            search_string = '((X >= %f) & (X <= %f)) & ((Y >= %f) & (Y <= %f))' % (XL, XH, YL, YH)
            df_focus = df.query(search_string)
            #print (df_focus)
            near_cities = list(df_focus['CityId'])

            replace = False
            best_city = -1
            best_dist = sys.maxsize
            for city in near_cities:
                if city == c_city:
                    continue
                if city == p_city:
                    continue
                if city in add_check:
                    continue
                if city in delete_list:
                    continue
                if city in already_done:
                    continue
                new_dist = calced([[dict_cities[city]['X'], dict_cities[city]['Y']]], [[XN,YN]])
                if new_dist < best_dist:
                    best_city = city
                    best_dist = new_dist
                    replace = True
            if verbose:
                print("Best City:", best_city, best_dist, [dict_cities[best_city]['X'], dict_cities[best_city]['Y']])
            if replace:
                delete_list.append(best_city)
                add_list.append([p_city, [best_city]])
                add_check.append(p_city)

        if verbose:
            print("Delete:", delete_list)
            print("Add after:", add_list)

        place = 1
        dict = {}
        dict[0] = dict_path[0] # copy Northpole
        previous_city = self.North_Pole
        for i in range(1,len(dict_path)-1):
            city = dict_path[i]['Path']
            if city in delete_list:
            # skip adding a city that is to be skipped
                if verbose:
                    print("Delete:", city)
            else:
            # add back in
                ed_dist = dist_city(previous_city, city, dict_cities)
                dict[place] = {'Step': place, 'Path': city, 'Dist': ed_dist}
                place = place + 1
                previous_city = city
            if city in add_check:
                for row in add_list:
                    if city == row[0]:
                        for new_city in row[1]:
                            ed_dist = dist_city(previous_city, new_city, dict_cities)
                            dict[place] = {'Step': place, 'Path': new_city, 'Dist': ed_dist}
                            place = place + 1
                            previous_city = new_city
                            if verbose:
                                print("Insert:", new_city, " after", city)

        ed_dist = dist_city(previous_city, dict_path[(len(dict_path) - 1)]['Path'], dict_cities)
        dict[place] = {'Step': place, 'Path': dict_path[(len(dict_path) - 1)]['Path'], 'Dist': ed_dist} # copy Northpole # copy last place

        # move to data frame with new columns with just one key!
        columns = ['Step', 'Path', 'Dist']
        index = ['Step']
        df_new = pd.DataFrame(index=index, columns=columns)
        df_new = pd.DataFrame.from_dict(dict, orient='index')

        return dict, df_new, delete_list

    def carrot_scan(self, dict_path, dict_cities, df, df_path, dist_hunt=100, verbose=True):


        place = 0
        rev_dict = {}
        # look-up dictionary excluding duplicate North Pole
        for i in range(1, (len(dict_path) - 1)):
            rev_dict[place] = {'Path': dict_path[i]['Path'], 'Step': dict_path[i]['Step']}
            place = place + 1

        used_list = []

        for i in range(10, len(dict_path), 10):
            c_city = dict_path[i]['Path']
            prime = dict_cities[c_city]['Prime']

            X1 = dict_cities[c_city]['X']
            Y1 = dict_cities[c_city]['Y']

            if verbose:
                print("Step:", i, "City:", c_city, "prime:", prime, "[", X1, ",", Y1, "]")

            if c_city == self.North_Pole:
                err = 'North Pole!: %i' % c_city
                raise Exception(err)

            if prime:
                continue
            XL = X1 - dist_hunt
            if XL < 0.0:
                XL = 0.0
            YL = Y1 - dist_hunt
            if YL < 0.0:
                YL = 0.0
            XH = X1 + dist_hunt
            YH = Y1 + dist_hunt
            if verbose:
                msg = "Hunt: X between %f and %f] Y Between %f , %f" % (XL, XH, YL, YH)
                print(msg)
            search_string = '((X >= %f) & (X <= %f)) & ((Y >= %f) & (Y <= %f)) & (Prime == 1)' % (XL, XH, YL, YH)
            df_focus = df.query(search_string)
            near_primes = list(df_focus['CityId'])

            # Remove cities that are already in correct place and already assigned
            for city in near_primes:
                step = rev_dict[city]['Step']
                if (step % 10) == 0:
                    near_primes.remove(city)
                elif city in used_list:
                    near_primes.remove(city)


            if verbose:
                print("Primes available:", len(near_primes))

            replace = False
            best_prime = -1
            best_dist = sys.maxsize
            for city in near_primes:

                new_dist = calced([[dict_cities[city]['X'], dict_cities[city]['Y']]], [[X1, Y1]])
                if new_dist > (1.1 * dict_path[i]['Dist']):
                    continue

                if new_dist < best_dist:
                    best_prime = city
                    best_dist = new_dist
                    replace = True
            if verbose:
                print("Best Prime:", best_prime, best_dist, [dict_cities[best_prime]['X'], dict_cities[best_prime]['Y']])
            if replace:
                prime_step = rev_dict[best_prime]['Step']
                dict_path[i] = {'Step': i, 'Path': best_prime, 'Dist': 0.0}
                dict_path[prime_step] = {'Step': prime_step, 'Path': c_city, 'Dist': 0.0}
                used_list.append(best_prime)
                if verbose:
                    print("Updated:", dict_path[i], dict_path[prime_step] )


        #dict, df_new = apply_path(dict_path, dict_cities, delete_list, add_list, add_check)

        return dict_path
# ######################## Start up ##############################


print("Raindeer Herd")
print("Version ", __version__, " ", __copyright__, " ", __license__)
print("Running on ", sys.version)

name_in = ""
#name_in = input("What file to process (default=deerherd.csv) ")
if name_in == "":
    name_in = "deerherd.csv"
df_path = pd.read_csv(name_in)

name_out = ""
#name_out = input("What file to output (default=deerherd.csv) ")
if name_out == "":
    name_out = "deerherd.csv"

begin_time = process_time()
reload = False  # Change this to reload or put an option for it.


print("  Get City Data and align....")
start_time = process_time()
# get data and add prime column in a fine DF for easy of use
load_cities = LoadCities()
df_cities = load_cities.load_file()
dict_cities = df_cities.to_dict('records')
print("  Execution time:", run_time(start_time))

print("  Load and measure....")
start_time = process_time()
df_path = pd.read_csv(name_in)
dict_path, df_path = fix_path(df_path, dict_cities)
print("  Execution time:", run_time(start_time))

print("  Travel....")
start_time = process_time()
initial_travel_time = load_cities.travel_time(dict_path, dict_cities)
print(initial_travel_time)
print("  Execution time:", run_time(start_time))

print("  Patch and fill....")
start_time = process_time()
#fix_these = load_cities.worst_path(df_path, dict_path)
#dict_new = load_cities.improve_scan(dict_path, dict_cities, fix_these)
#dict_new = load_cities.better_scan(dict_path, dict_cities)
dict_new = dict_path
skip_list = []
for i in range(1,11):
    print("Epoch:", i)
    fix_these = load_cities.worst_path(df_path, dict_new, 100)
    dict_new, df_path, used_list = load_cities.split_scan(dict_path, dict_cities, fix_these, df_cities, skip_list)
    dict_path, df_path = fix_path(df_path, dict_cities)
    print("Number of changes:", len(used_list))
    skip_list = skip_list + used_list  # Only move a city once! This stops cycles.
    print(skip_list)
#dict_new = load_cities.better_scan(dict_new, dict_cities, 4)  # Possible we reverse something that should be put back
#dict_new = {}
#dict_new = load_cities.carrot_scan(dict_path, dict_cities, df_cities, df_path)
print("  Execution time:", run_time(start_time))

print("  Travel....")
start_time = process_time()
travel_time = load_cities.travel_time(dict_new, dict_cities)
print(travel_time)
print("  Execution time:", run_time(start_time))
print("  Previous time:", initial_travel_time)
sys.exit()
if int(initial_travel_time) < int(travel_time):
    print("Run Failed")
    sys.exit(0)

print("   Convert and Write...")
df = pd.DataFrame.from_dict(dict_new, orient='index')
df = df.drop(['Step', 'Dist'], axis=1)
df.to_csv(name_out, index=False)
start_time = process_time()
print("   Execution time:", run_time(start_time))

print(" ")
print("Run time:", run_time(begin_time))
print("Finished...End of Line")