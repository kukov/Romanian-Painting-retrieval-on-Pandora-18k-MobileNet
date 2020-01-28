# Romanian Paiting retrieval on Pandora 18k using MobileNet
Project for MLAV at Master TAID UPB Romania
Steps:
1. Train MobileNet arhitecture using retrain.py on Pandora 18k database
2. Run the classifier on a selected number of images(1000 in this case) from Pandora 18k
3. Run the classifier on a selected number of iamges(400 u=in this case) from Pandora RO
4. Calculate distance(euclidian in this case) for each image from the step 3 to each image from the step 2
5. Select the best 5 distances
6. Show original image and the selected 5 images with the lowest distance
