import pandas as pd

## dict --> key pandora 18k
##          value romanian paintings
classificatior_classes= {
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
    'symbolism': 'Symbolism'
}

excel_file = 'romanian_wikiart_2.xlsx'
paintings = pd.read_excel(excel_file)
paitings_col = []
for col in paintings.columns:
    paitings_col.append(col)
paintings_of_interest= pd.DataFrame(columns=paitings_col)

print(paintings_of_interest.shape)
#print(paintings['style'].unique())
#print(paintings[paintings['style'] == 'Impressionism'])

rows_of_interest= 0
for key, val in classificatior_classes.items():
    paints = paintings[paintings['style'] == val]
    paintings_of_interest = pd.concat([paintings_of_interest, paints])
    print(f'Number of painting of style {key} is {paints.shape[0]}')
    rows_of_interest += paints.shape[0]

print(paintings_of_interest.shape)

from shutil import copyfile
for index, row in paintings_of_interest.iterrows():
    try:
        copyfile(f"/home/kukov/PycharmProjects/MLAV_Proiect/images/{row['image_id']}.jpg", f"/home/kukov/PycharmProjects/MLAV_Proiect/img/{row['image_id']}.jpg")
    except Exception as exception:
        print(exception)





