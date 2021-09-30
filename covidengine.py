#here we import needed libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import six
import sys
sys.modules['sklearn.externals.six'] = six 
from six import StringIO
from IPython.display import Image  
import pydotplus
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.tree import _tree
import numpy as np
import cv2
import knowledgeBase


#in this part we ask about user's symptoms
fever= int(input('Do you have fever? 1 as yes and 0 as no '))
h_ache= int(input('Do you have head ache? 1 as yes and 0 as no '))
pain= int(input('Do you have pain? 1 as yes and 0 as no '))
weak= int(input('Do you feel weak? 1 as yes and 0 as no '))
cough=int(input('Do you cough? 1 as yes and 0 as no '))
r_nose= int(input('Do you have a runny nose? 1 as yes and 0 as no '))
s_throat= int(input('Do you have sore throat? 1 as yes and 0 as no '))
sneez= int(input('Do you sneez? 1 as yes and 0 as no '))
digestive= int(input('Do you have digestive problems? 1 as yes and 0 as no '))
anosmia= int(input('Have you experienced loss in taste or smell? 1 as yes and 0 as no '))
gender= int(input('Are you male or female? 1 as male and 0 as female '))
age= int(input('How old are you? '))
if age<20:
 age=0
else:
 age=1
#here we call engine function which finds the class related to the patient's symptoms 
c=knowledgeBase.engine(fever,h_ache,pain,weak,cough,r_nose,gender,s_throat,sneez,digestive,anosmia,age,digestive)
def illness (c):
 if c==1:
  return 'Cold'
 elif c==2:
  return 'Influenza'
 elif c==3:
  return 'Allergy'
 elif c==4:
  return 'Covid19'
 else:
  return 'none of the covid nor cold nor influenza nor allergy'
if c==5:
 print("You are healthy!")
else:
 print("Unfortunately you have been diagnosed with " + illness(c))
#the following part displays the decision tree
img = mpimg.imread('tree.png')
plt.imshow(img)
plt.show()