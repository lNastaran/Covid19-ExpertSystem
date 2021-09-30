#importing libraries
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
from id3 import Id3Estimator
estimator = Id3Estimator()

col_names = ['fever', 'headAche', 'pain','weakness', 'runnyNose', 'sneezing', 'soreThroat', 'cough', 'asthma','gender','age','anosmia','digestive','label']
#loading dataset
dataset = pd.read_csv("dataset.csv", header=None, names=col_names)
dataset.head()
#the following array is for feature columns 
feature_cols = ['fever','headAche','pain','weakness','runnyNose','sneezing','soreThroat','cough','asthma','gender','age','anosmia','digestive']
#here we split dataset in features(symptoms) and target variable(illness name)
X = dataset[feature_cols] # Features
y = dataset.label # Target variable
#in this part we split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train.fillna(X_test.mean())
#creating decision tree classifer object using ID3 algorithm
clf = DecisionTreeClassifier(criterion="entropy")
#training decision tree classifer
clf = clf.fit(X_train,y_train)
#predicting the label for test dataset
y_pred = clf.predict(X_test)
#the following lines draws the tree in tree.png file
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['1','2','3','4','5'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())
tree.plot_tree(clf);
#these are the illnesses which our system can diagnose
label_names=['COLDS','influenza','Allergy','Covid19']
filename = "knowledgeBase.py"
#the tree is converted to a python code and written as knowledge base in the following function
def tree_to_code(tree, feature_names):
 tree_ = tree.tree_
 feature_name = [
  feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
  for i in tree_.feature
 ]
 file = open(filename, "w")
 #the function used in engine file
 file.write("def engine({}):\n".format(", ".join(feature_names)))
 def recurse(node, depth):
  indent = " " * depth
  if tree_.feature[node] != _tree.TREE_UNDEFINED:
   name = feature_name[node]
   threshold = tree_.threshold[node]
   file.write("{}if {} <= {}:\n".format(indent, name, np.round(threshold,2)))
   recurse(tree_.children_left[node], depth + 1)
   file.write("{}else:\n".format(indent, name, np.round(threshold,2)))
   recurse(tree_.children_right[node], depth + 1)
  else:
    #finding the suitable class for each leaf based on class's probability
    x=np.argmax(tree_.value[node])+1
    file.write("{}return {}\n".format(indent, x))
 recurse(0, 1)
 file.close()
#by culling the above function with our tree and the names of features knowledge base is written
tree_to_code(clf,feature_cols)