import pandas as pd
from label_image_func import test_image
from sklearn.metrics import precision_score, recall_score
from helpers import select_img_dataframe_pandora, select_img_dataframe_rom, calculate_euclidian_distance
from skimage import data, io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os

PATH = '/home/kukov/Pandora18k'

## dict --> key pandora 18k
##          value romanian paintings
classificator_classes = {
    'abstractart': 'Abstract Art',
    'cubism': 'Cubism',
    'expressionism': 'Expressionism',
    'fauvism': 'Fauvism',
    'impressionism': 'Impressionism',
    'naiveart': 'Naive Art',
    'popart': 'Pop Art',
    'post impressionism': 'Post-Impressionism',
    'realism': 'Realism',
    'romanticism': 'Romanticism',
    'surrealism': 'Surrealism',
    'symbolism': 'Symbolism',
    'baroque': 'baroque',
    'byzantin iconography': 'byzantin iconography',
    'early renaissance': 'early renaissance',
    'high renaissance': 'high renaissance',
    'northern renaissance': 'northern renaissance',
    'rococo': 'rococo',


}

excel_file = 'romanian_wikiart_2.xlsx'
paintings = pd.read_excel(excel_file)
nr_sample_pictures = 400




paitings_col = []
for col in paintings.columns:
    paitings_col.append(col)
paintings_of_interest= pd.DataFrame(columns=paitings_col)

#print(paintings_of_interest.shape)
#print(paintings['style'].unique())
#print(paintings[paintings['style'] == 'Impressionism'])

rows_of_interest= 0
for key, val in classificator_classes.items():
    paints = paintings[paintings['style'] == val]
    paintings_of_interest = pd.concat([paintings_of_interest, paints])
    #print(f'Number of painting of style {key} is {paints.shape[0]}')
    rows_of_interest += paints.shape[0]

#print(paintings_of_interest.head())
#print(paintings_of_interest.shape)

random_subset = paintings_of_interest.sample(n=nr_sample_pictures)
# print(random_subset.head())
# print(random_subset['style'])

pandora_classified_imgs = select_img_dataframe_pandora()
rom_classified_imgs = select_img_dataframe_rom(paintings_of_interest)


# print(pandora_classified_imgs)
# print(rom_classified_imgs)


for key in rom_classified_imgs.keys():
    rom_classified_result = rom_classified_imgs[key]['classifier_values']
    top_5_images = calculate_euclidian_distance(rom_classified_result, pandora_classified_imgs)
    #print(top_5_images)
    dim = (300,300)
    orig_img = cv2.imread(f"/home/kukov/PycharmProjects/MLAV_Proiect/img/{key}.jpg")
    orig_img = cv2.resize(orig_img, dim, interpolation=cv2.INTER_AREA)
    #width, height, color_plane = orig_img.shape
    #dim = (height, width)
    #print(dim)
    top_5_images_list = list()
    for img_key in top_5_images.keys():
        #print(img_key)
        img_obj = cv2.imread(os.path.join(pandora_classified_imgs[img_key]['dir'],img_key))
        resized_img_obj = cv2.resize(img_obj, dim, interpolation=cv2.INTER_AREA)
        #print(resized_img_obj.shape)
        top_5_images_list.append(resized_img_obj)
    image_comparison = np.concatenate((orig_img, top_5_images_list[0], top_5_images_list[1], top_5_images_list[2],
                                       top_5_images_list[3], top_5_images_list[4]), axis=1)
    cv2.imshow('image_comparison', image_comparison)
    cv2.waitKey(0)






# ground_truth_style = []
# classificator_result = []
# clear_classificator_result = []
# for index, row in random_subset.iterrows():
#     ground_truth_style.append(row['style'])
#     result = test_image(f"/home/kukov/PycharmProjects/MLAV_Proiect/img/{row['image_id']}.jpg")
#     #print(result)
#     classificator_result.append(result)
#
# for var in classificator_result:
#     #clear_label = classificator_classes.get(var)
#     clear_label = classificator_classes.get(var, 'Not in Classificator Classes')
#     clear_classificator_result.append(clear_label)
#
# not_in_class = 0
# for var in clear_classificator_result:
#     if var == 'Not in Classificator Classes':
#         not_in_class += 1
# print(f"Not in class {not_in_class}")
#
# # for i in range(10):
# #     print(classificator_result[i], clear_classificator_result[i], ground_truth_style[i])
#
#
# # print(f"Ground truth len is {len(ground_truth_style)}")
# # print(f"Clear ")
# recall = recall_score(ground_truth_style, clear_classificator_result, average='micro')
# precision = precision_score(ground_truth_style, clear_classificator_result, average='micro')
# print(f"Recall in Romanian Paintings is {recall}")
# print(f"Precision in Romanian Paintings is {precision}")
#
# #print(ground_truth_style)
#
#
# pandora_ground_truth_style = []
# pandora_classificator_result = []
# pandora_clear_classificator_result = []
# for index, row in pandora_test_paintings.iterrows():
#     pandora_ground_truth_style.append(row['style'])
#     result = test_image(f"/home/kukov/PycharmProjects/MLAV_Proiect/Testare/{row['image_id']}")
#     #print(result)
#     pandora_classificator_result.append(result)
#
# for var in pandora_classificator_result:
#     #clear_label = classificator_classes.get(var)
#     clear_label = classificator_classes.get(var, 'Not in Classificator Classes')
#     pandora_clear_classificator_result.append(clear_label)
#
#
# pandora_recall = recall_score(pandora_ground_truth_style, pandora_clear_classificator_result, average='micro')
# pandora_precision = precision_score(pandora_ground_truth_style, pandora_clear_classificator_result, average='micro')
# print(f"Recall in Pandora Paintings is {pandora_recall}") #R = TP / (TP + FN)
# print(f"Precision in Pandora Paintings is {pandora_precision}") #P = TP / (TP + FP)
#
#
#

