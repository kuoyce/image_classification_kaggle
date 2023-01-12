import os
from tensorflow import keras
import cv2
from tqdm import tqdm
import numpy as np

print('keras is in!')

folderpath = r'.\database\archive'

train_filepath = os.path.join(folderpath, 'seg_train', 'seg_train')
test_filepath = os.path.join(folderpath, 'seg_test', 'seg_test')

def load_dataset(train_filepath,test_filepath):
    rtn = []
    for filepath in [train_filepath,test_filepath]:
        print(os.listdir(filepath))
        TRAIN_IMGSIZE = (150,150)

        wrong_size = 0
        images,labels = [], []
        for label in os.listdir(filepath):
            img_labelpath = os.path.join(filepath, label)
            for img_id in tqdm(os.listdir(img_labelpath)):
                img_loc = os.path.join(img_labelpath, img_id)
                image = cv2.imread(img_loc)
                if image.shape[:2] != TRAIN_IMGSIZE:
                    image = cv2.resize(image, TRAIN_IMGSIZE)
                    wrong_size += 1
                images.append(image)
                labels.append(label)
        
        
        print(wrong_size, 'images were incorrectly sized')
        images = np.array(images)
        labels = np.array(labels)
        rtn.append((images,labels))

    return rtn


((trainX,trainY),(testX,testY)) = load_dataset(train_filepath,test_filepath)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


