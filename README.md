# Traffic-Sign-Detection
This project was developed during my efforts to self-learn Python. This repository is based on Chapter 6 of [Modern Computer Vision with Pytorch]. Traffic Sign Detetion is a commmon problem in image classification. The dataset used can be found in the following [link]. A csv file(train.csv) was created using the CSV Creator.py containing the following columns: Filepath,Sign Name,5 Digit Int Label,1 Digit Int Label. Examples of the images in the dataset can be seen below:

![Screenshot_6](https://github.com/aristosp/Traffic-Sign-Detection/assets/62808962/7cfcd65d-2415-486e-ac89-7ddf417f91e8)

The network is a simple CNN, developed after trial and error. There may be signs of overfitting, but the results of the test set are satisfactory, as the accuracy reaches approximately 96% and the loss ~0.12. Example predictions can be seen below:



[Modern Computer Vision with Pytorch]: https://www.oreilly.com/library/view/modern-computer-vision/9781839213472/
[link]:https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
