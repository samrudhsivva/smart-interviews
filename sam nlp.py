import numpy as np
import pandas as pd
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data=[]
for i in range(0,1000):
        review=dataset["Review"][i]
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        data.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(data).toarray()
y=dataset.iloc[:,1].values       
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(output_dim = 1000, init = 'uniform', activation = 'sigmoid', input_dim = 1500))
model.add(Dense(output_dim = 100, init = 'uniform', activation = 'sigmoid'))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train,y_train,batch_size=32,epochs=10)
y=model.predict(X_test)
text="food is bad"
r=model.predict(cv.transform(['good food']))