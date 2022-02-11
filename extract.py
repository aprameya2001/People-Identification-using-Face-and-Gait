import os

os.system('mkdir extracted_dataset')

files = os.listdir('dataset')

for file in files:
    os.system('unzip dataset/' + file)
    os.system('mv ' + file.split('.')[0] + ' extracted_dataset/' + file.split('.')[0])