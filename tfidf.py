import pandas as pd
import sklearn as sk
import math

first= "En ce mois de février, le groupe Addoha a fait une nouvelle fois appel au marché obligataire pour s’endetter. Près de 600 millions de dirhams (55 millions d’euros) vont être levés par le promoteur immobilier, pour lesquels les équipes financières du groupe ont négocié un délai de deux ans pour le remboursement du capital afin de se donner un peu de répit. Cette nouvelle ligne de crédit a pour principal objectif de rallonger la maturité de la dette de la société qui s’élève à 4,8 milliards de dirhams au 30 septembre 2020. « À force d’accumuler de la dette, Addoha est obligé d’effectuer des reprofilages de temps en temps. Aujourd’hui, le groupe souffre de la crise liée à la pandémie  – qui a asséché ses caisses – et a besoin de trésorerie pour continuer de fonctionner normalement. Le secteur dans sa globalité traîne un stock d’invendus immense », nous explique un chargé d’affaires auprès d’une banque partenaire du groupe."
second = "Addoha"

first = first.split(" ")
second = second.split(" ")
total = set(first).union(set(second))
print(total)

wordDictA = dict.fromkeys(total, 0)
wordDictB = dict.fromkeys(total, 0)
for word in first:
    wordDictA[word] += 1

for word in second:
    wordDictB[word]+=1

pd.DataFrame([wordDictA, wordDictB])

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict
tfFirst = computeTF(wordDictA, first)
tfSecond = computeTF(wordDictB, second)
tf_df= pd.DataFrame([tfFirst])


def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict



idfs = computeIDF([wordDictA, wordDictB])


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


