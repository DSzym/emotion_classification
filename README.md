# emotion_classification
Classification of three emotions: sad, happy and suprised presented on one's photo

## Requirements
- Python 3.9
- libaries:  cv2, numpy, os, glob
- downloaded and placed in folder with program: haarcascade_frontalface_alt2.xml, haarcascade_frontalface_default.xml, haarcascade_frontalface_alt.xml

## Description
Using SIFT algorithm and haarcascade classifier, this program classifies 3 possible emotions from pictures. But before it's able to properly classify emotion from picture, the classification params need to be adjusted by analysing given train data. I used about 550 images collected from the internet as train and test data (75% of them are randomly taken as train and rest is test data) and it resulted in 50-70% classification accuracy. There also needed to be selected about 10 per each emotion representation data - those are images that compared to other from same emotion give high scores and compared to other not from the same emotion give low scores.  
