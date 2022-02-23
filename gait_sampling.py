import os
import random

cutoff = 80

for i in range(40):

    folder = 'extracted_dataset' + '/' + 'person' + str(i+1) + '/' + 'gait'

    styles = os.listdir(folder)
    styles.sort()

    for style in styles:

        folder2 = folder + '/' + style
        
        angles = os.listdir(folder2)
        angles.sort()
        
        for angle in angles:

            folder3 = folder2 + '/' + angle
            
            files = os.listdir(folder3)
            
            if len(files) == 0:
                continue
            
            counter = 0
            while len(os.listdir(folder3)) < cutoff:
                counter += 1
                for file in files:
                    os.system('cp ' + folder3 + '/' + file + ' ' + folder3 + '/' + file + '.' + str(counter))
            
            files = os.listdir(folder3)
            random.shuffle(files)
            files = files[cutoff:]
            
            for file in files:
                path = folder3 + '/' + file
                os.system('rm -rf ' + path)
            
            files = os.listdir(folder3)
            files.sort()
            
            for k in range(cutoff):

                newFile = str(k+1)
                while len(newFile)<2:
                    newFile = "0" + newFile
                    
                if (folder3 + '/' + files[k]) != (folder3 + '/' + newFile + '.png'):
                    os.system('mv ' + folder3 + '/' + files[k] + ' ' + folder3 + '/' + newFile + '.png')
            
            print('person', str(i+1), '\tstyle', style, '\tangle', angle, '\t...done')