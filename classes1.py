import pickle

classes = ['dementia', 'mild_cognitive_impairment', 'delirium']

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)