from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer



data= "En ce mois de février, le groupe Addoha a fait une nouvelle fois appel au marché obligataire pour s’endetter. Près de 600 millions de dirhams (55 millions d’euros) vont être levés par le promoteur immobilier, pour lesquels les équipes financières du groupe ont négocié un délai de deux ans pour le remboursement du capital afin de se donner un peu de répit. Cette nouvelle ligne de crédit a pour principal objectif de rallonger la maturité de la dette de la société qui s’élève à 4,8 milliards de dirhams au 30 septembre 2020. « À force d’accumuler de la dette, Addoha est obligé d’effectuer des reprofilages de temps en temps. Aujourd’hui, le groupe souffre de la crise liée à la pandémie  – qui a asséché ses caisses – et a besoin de trésorerie pour continuer de fonctionner normalement. Le secteur dans sa globalité traîne un stock d’invendus immense », nous explique un chargé d’affaires auprès d’une banque partenaire du groupe."

cv = CountVectorizer()

words = ['Addoha','groupe','ADH','AD','ahodda']

tfidf_vectorizer = TfidfVectorizer(vocabulary=words)

#tfidf_vectorizer.fit(words)
tfidf_vectorizer.fit_transform(data)



