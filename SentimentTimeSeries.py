import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pylab import rcParams

import pandas as pd
import numpy as np
import nltk
from textblob.translate import Translator
from langdetect import detect
from nltk.tag.stanford import StanfordPOSTagger


from textblob import TextBlob as TextBlobEN
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
#from textblob_ar import TextBlob as TextBlobAR

Media = "En ce mois de février, le groupe Addoha a fait une nouvelle fois appel au marché obligataire pour s’endetter. Près de 600 millions de dirhams (55 millions d’euros) vont être levés par le promoteur immobilier, pour lesquels les équipes financières du groupe ont négocié un délai de deux ans pour le remboursement du capital afin de se donner un peu de répit. Cette nouvelle ligne de crédit a pour principal objectif de rallonger la maturité de la dette de la société qui s’élève à 4,8 milliards de dirhams au 30 septembre 2020. « À force d’accumuler de la dette, Addoha est obligé d’effectuer des reprofilages de temps en temps. Aujourd’hui, le groupe souffre de la crise liée à la pandémie  – qui a asséché ses caisses – et a besoin de trésorerie pour continuer de fonctionner normalement. Le secteur dans sa globalité traîne un stock d’invendus immense », nous explique un chargé d’affaires auprès d’une banque partenaire du groupe."

i=0
while i < len(Media):
    print(i)
    blob=tb(txt)
    senti = blob.sentiment[0]
    Media.at[i,"sentiment"]=senti
    i+=1



all_media_sentiment  = pd.concat([sentiment_to_update,Media],ignore_index=True)

#%%

