import os
#first of all we make our descision tree and build a knowledge base in python form
os.system('py tree.py')
#in the following file we ask user abot his symptoms
#and based on knowledge base we diagnose his illness
os.system('py covidengine.py')