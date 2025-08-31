import tensorflow as tf
import nltk
import pyttsx3
import importlib.metadata

print(f"TensorFlow version: {tf.__version__}")
print(f"NLTK version: {nltk.__version__}")
print(f"pyttsx3 version: {importlib.metadata.version('pyttsx3')}")