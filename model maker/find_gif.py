import os
dataset_path = 'D:/Anaconda/Coin detection and count/model maker/dataset/'
path = os.path.join(dataset_path, 'train')

files = os.listdir(path)

for file in files:
    if file.endswith('.JPG'):
        print(file)