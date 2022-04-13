# People-Identification-using-Face-and-Gait

People identification has become a major challenge in the present world due to the growing importance of security, privacy and surveillance. People identification is currently being done by analysing features such as facial features, gait features etc. But, each of these techniques has certain flaws and may give poor results when certain conditions are met. The effect of these flaws can be reduced by combining different techniques of people identification. In this paper, a novel method is proposed which uses feature level fusion of facial and gait features to perform people identification at a distance. Deep learning techniques are used for extracting the relevant features, fusing those extracted features and then creating a model which can correctly predict the person by using the facial and gait features of that person as input. To train the deep learning-based model, the ORL dataset is used for facial data, and the CASIA-B dataset is used for gait data. The model is created in Python programming language, using TensorFlow, which is a popular software library extensively used for Deep Learning and Machine learning tasks. The proposed base fusion model achieved a accuracy of 90.83\% and after performing data augmentation and training the model on the augmented data, the model achieved an accuracy of 97.50\%.

## Structure
- dataset --> Dataset used for the project. ORL and CASIA-B datasets are combined to create this dataset.
- IT416_Dataset_Preprocessing.ipynb --> Notebook to preprocess the dataset and upload the preprocessed dataset to drive
- IT416_Dataset_Visualization.ipynb --> Notebook for dataset visualisation
- IT416_Fusion_Model_without_batch_normalisation.ipynb --> Notebook for implementing fusion model without Batch Normalisation
- IT416_Fusion_Model_with_only_one_ConvLSTM2D_layer.ipynb --> Notebook for implementing fusion model with only one ConvLSTM2D layer
- IT416_Fusion_Model.ipynb --> Notebook for proposed fusion model
- IT416_Fusion_Model_Augmentation.ipynb --> Notebook for Fusion model with data augmentation

## Instructions
- Upload the dataset in your drive.
- Run the notebook 'IT416_Dataset_Preprocessing.ipynb' 
- Run the remaining notebooks in any order that you want. Make sure to change the path of the dataset according to your need.

## Purpose  
The project 'Feature Level Fusion of Gait and Facial Features for People Identification at a Distance (People-Identification-using-Face-and-Gait)' has been created as a mini project for the course IT416- Computer Vision. 

## Contributors  
- Pratham Nayak (191IT241)  
- Aprameya Dash (191IT209)
