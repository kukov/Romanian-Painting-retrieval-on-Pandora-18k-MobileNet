import os
import pandas as pd
from label_image_func import test_image
from math import sqrt
PATH = '/home/kukov/Pandora18k'


def get_list_of_dirs(path=PATH):
    dirs = os.listdir(path)
    return dirs

def create_dataframe():
    my_dict = {"name": list(), "dir":list()}
    img_dirs = get_list_of_dirs()
    for dir in img_dirs:
        new_path = os.path.join(PATH, dir)
        for r, d, f in os.walk(new_path):
            for file in f:
                my_dict['name'].append(file)
                my_dict['dir'].append(dir)

    img_df = pd.DataFrame(my_dict)
    return img_df


def select_img_dataframe_pandora(nr_of_selected_pictures=1000, expected_dataframe=False):
    if not expected_dataframe:
        all_img_dataframe = create_dataframe()
    else:
        all_img_dataframe = expected_dataframe.copy()
    final_dict = dict()
    selected_images = all_img_dataframe.sample(n=nr_of_selected_pictures)
    unique_pictures = selected_images.nunique()['name']
    if unique_pictures != nr_of_selected_pictures:
        raise Exception("Not all names are unique")

    for index, row in selected_images.iterrows():
        #print(os.path.join(PATH, row['dir'], row['name']))
        classifier_dict_result = test_image(os.path.join(PATH, row['dir'], row['name']))
        final_dict[row['name']] = {'dir': os.path.join(PATH, row['dir']), 'classifier_values':classifier_dict_result}

    #print(len(final_dict))
    return final_dict

def select_img_dataframe_rom(expected_dataframe, nr_of_selected_pictures=10 ):
    all_img_dataframe = expected_dataframe.copy()
    # print(all_img_dataframe.head())
    final_dict = dict()
    selected_images = all_img_dataframe.sample(n=nr_of_selected_pictures)
    unique_pictures = selected_images.nunique()['image_id']
    if unique_pictures != nr_of_selected_pictures:
        raise Exception("Not all names are unique")

    for index, row in selected_images.iterrows():
        # print(row['image_id'])
        #print(os.path.join(PATH, row['dir'], row['name']))
        classifier_dict_result = test_image(f"/home/kukov/PycharmProjects/MLAV_Proiect/img/{row['image_id']}.jpg")
        final_dict[row['image_id']] = {'dir': '/home/kukov/PycharmProjects/MLAV_Proiect/img/',
                                   'classifier_values':classifier_dict_result}

    #print(len(final_dict))
    #print(final_dict)
    return final_dict


def calculate_euclidian_distance(item, dict_item):
    distance_dict = dict()

    for key in dict_item.keys():
        classificator_value = dict_item[key]['classifier_values']
        dist = calc_dist(item,classificator_value)
        distance_dict[key] = dist

    distance_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}
    top_5_results = {k: distance_dict[k] for k in list(distance_dict)[:-5]}
    return top_5_results


def calc_dist(dict1, dict2):
    dist = 0
    for key in dict1.keys():
        value_dict1 = dict1[key]
        value_dict2 = dict2[key]
        dist += (value_dict1 - value_dict2)**2
    dist = sqrt(dist)
    return dist

#select_img_dataframe_pandora()
