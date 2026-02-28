import re
import os
import numpy as np
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



## NLTK setup

nltk.download("stopwords")
nltk.download("wordnet")


model_path = ""
