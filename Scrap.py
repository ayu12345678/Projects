import bs4
import requests
import pandas as pd
import csv
import re
import string
import nltk
import sklearn
import numpy as np

url="https://www.youtube.com/results?search_query=travel&sp=EgIQAQ%253D%253D"
df0=requests.get(url)
soup=bs4.BeautifulSoup(df0.text,'html.parser')


print(soup.prettify())
vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})
videolist=[]
videotitle=[]
category=[]
for v in vids:
    tmp = 'https://www.youtube.com' + v['href']
    videolist.append(tmp)
    category.append('travel')
    videotitle.append(v['title'])

    print(v['title'])
    v = str(v)
    views = ''
    try:
        indx = v.index('views')
        indx = indx - 2
        while v[indx] is not ' ':
            views = views + v[indx]
            indx = indx -1
        print(views[::-1])
    except:
        continue

videodescription=[]
for content in soup.find_all('div', class_= "yt-lockup-content"):
    try:

        description = content.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        videodescription.append(description)

    except Exception as e:
        description = None

    print('\n')
videodescription.append(0)
url1="https://www.youtube.com/results?search_query=science+and+technology&sp=EgIQAQ%253D%253D"
df1=requests.get(url1)
soup1=bs4.BeautifulSoup(df1.text,'html.parser')


print(soup1.prettify())
vids1 = soup1.findAll('a',attrs={'class':'yt-uix-tile-link'})

for v1 in vids1:
    tmp1 = 'https://www.youtube.com' + v1['href']
    videolist.append(tmp1)
    category.append('science and technology')
    videotitle.append(v1['title'])

    print(v1['title'])
    v1 = str(v1)
    views1 = ''
    try:
        indx1 = v1.index('views')
        indx1 = indx1 - 2
        while v1[indx1] is not ' ':
            views1 = views1 + v1[indx1]
            indx1 = indx1 -1
        print(views1[::-1])
    except:
        continue


for content1 in soup1.find_all('div', class_= "yt-lockup-content"):
    try:

        description1 = content1.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        videodescription.append(description1)

    except Exception as e:
        description1 = None

    print('\n')
url2="https://www.youtube.com/results?search_query=food&sp=EgIQAQ%253D%253D"
df2=requests.get(url2)
soup2=bs4.BeautifulSoup(df2.text,'html.parser')


print(soup2.prettify())
vids2 = soup2.findAll('a',attrs={'class':'yt-uix-tile-link'})

for v2 in vids2:
    tmp2 = 'https://www.youtube.com' + v2['href']
    videolist.append(tmp2)
    category.append('food')
    videotitle.append(v2['title'])

    print(v2['title'])
    v2 = str(v2)
    views2 = ''
    try:
        indx2 = v2.index('views')
        indx2 = indx2 - 2
        while v2[indx2] is not ' ':
            views2 = views2 + v2[indx2]
            indx2 = indx2 -1
        print(views2[::-1])
    except:
        continue


for content2 in soup2.find_all('div', class_= "yt-lockup-content"):
    try:

        description2 = content2.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        videodescription.append(description2)

    except Exception as e:
        description2 = None

    print('\n')

url3="https://www.youtube.com/results?search_query=manufacturing&sp=EgIQAQ%253D%253D"
df3=requests.get(url3)
soup3=bs4.BeautifulSoup(df3.text,'html.parser')


print(soup3.prettify())
vids3 = soup3.findAll('a',attrs={'class':'yt-uix-tile-link'})

for v3 in vids3:
    tmp3 = 'https://www.youtube.com' + v3['href']
    videolist.append(tmp3)
    category.append('manufacturing')
    videotitle.append(v3['title'])

    print(v3['title'])
    v3 = str(v3)
    views3 = ''
    try:
        indx3 = v3.index('views')
        indx3 = indx3 - 2
        while v3[indx3] is not ' ':
            views3 = views3 + v3[indx3]
            indx3 = indx3 -1
        print(views3[::-1])
    except:
        continue


for content3 in soup3.find_all('div', class_= "yt-lockup-content"):
    try:

        description3 = content3.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        videodescription.append(description3)

    except Exception as e:
        description3 = None

    print('\n')
videodescription.append(0)
url4="https://www.youtube.com/results?search_query=history&sp=EgIQAQ%253D%253D"
df4=requests.get(url4)
soup4=bs4.BeautifulSoup(df4.text,'html.parser')


print(soup4.prettify())
vids4 = soup4.findAll('a',attrs={'class':'yt-uix-tile-link'})

for v4 in vids4:
    tmp4 = 'https://www.youtube.com' + v4['href']
    videolist.append(tmp4)
    category.append('history')
    videotitle.append(v4['title'])

    print(v4['title'])
    v4 = str(v4)
    views4 = ''
    try:
        indx4 = v4.index('views')
        indx4 = indx4 - 2
        while v4[indx4] is not ' ':
            views4 = views4 + v4[indx4]
            indx4 = indx4 -1
        print(views4[::-1])
    except:
        continue


for content4 in soup4.find_all('div', class_= "yt-lockup-content"):
    try:

        description4 = content4.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        videodescription.append(description4)

    except Exception as e:
        description4 = None

    print('\n')
url5="https://www.youtube.com/results?search_query=art+and+music&sp=EgIQAQ%253D%253D"
df5=requests.get(url5)
soup5=bs4.BeautifulSoup(df5.text,'html.parser')


print(soup5.prettify())
vids5 = soup5.findAll('a',attrs={'class':'yt-uix-tile-link'})

for v5 in vids5:
    tmp5 = 'https://www.youtube.com' + v5['href']
    videolist.append(tmp5)
    category.append('art and music')
    videotitle.append(v5['title'])

    print(v5['title'])
    v5 = str(v5)
    views5 = ''
    try:
        indx5 = v5.index('views')
        indx5 = indx5 - 2
        while v5[indx5] is not ' ':
            views5 = views5 + v5[indx5]
            indx5 = indx5 -1
        print(views5[::-1])
    except:
        continue


for content5 in soup5.find_all('div', class_= "yt-lockup-content"):
    try:

        description5 = content5.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        videodescription.append(description5)

    except Exception as e:
        description5 = None

    print('\n')



print(videolist)
print(videotitle)
print(videodescription)
print(len(videolist))
print(len(videotitle))
print(len(videodescription))

dict = {'Vedio id': videolist, 'Title': videotitle, 'Description': videodescription,'Category': category}

df = pd.DataFrame(dict)


df.to_csv('newscrapper.csv')
num_missing_desc = df.isnull().sum()[2]    # No. of values with msising descriptions
print('Number of missing values: ' + str(num_missing_desc))
df = df.dropna()

df['Title'] = df['Title'].map(lambda x: x.lower())
df['Description'] = df['Description'].map(lambda x: x.lower())

# Remove numbers
df['Title'] = df['Title'].map(lambda x: re.sub(r'\d+', '', x))
df['Description'] = df['Description'].map(lambda x: re.sub(r'\d+', '', x))

# Remove Punctuation
df['Title'] = df['Title'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
df['Description'] = df['Description'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))

# Remove white spaces
df['Title'] = df['Title'].map(lambda x: x.strip())
df['Description'] = df['Description'].map(lambda x: x.strip())

# Tokenize into words
from nltk.tokenize import sent_tokenize, word_tokenize
df['Title'] = df['Title'].map(lambda x: word_tokenize(x))
df['Description'] = df['Description'].map(lambda x: word_tokenize(x))

# Remove non alphabetic tokens
df['Title'] = df['Title'].map(lambda x: [word for word in x if word.isalpha()])
df['Description'] = df['Description'].map(lambda x: [word for word in x if word.isalpha()])
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['Title'] = df['Title'].map(lambda x: [w for w in x if not w in stop_words])
df['Description'] = df['Description'].map(lambda x: [w for w in x if not w in stop_words])

# Word Lemmatization
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
df['Title'] = df['Title'].map(lambda x: [lem.lemmatize(word, "v") for word in x])
df['Description'] = df['Description'].map(lambda x: [lem.lemmatize(word, "v") for word in x])

# Turn lists back to string
df['Title'] = df['Title'].map(lambda x: ' '.join(x))
df['Description'] = df['Description'].map(lambda x: ' '.join(x))



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df.Category)
df.Category = le.transform(df.Category)
df.head(5)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_title = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
tfidf_desc = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
labels = df.Category
features_title = tfidf_title.fit_transform(df.Title).toarray()
features_description = tfidf_desc.fit_transform(df.Description).toarray()
print('Title Features Shape: ' + str(features_title.shape))
print('Description Features Shape: ' + str(features_description.shape))

from sklearn.feature_selection import chi2
import numpy as np

N = 5
for current_class in list(le.classes_):
    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_title, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_title.get_feature_names())[indices]
    unigrams = [vk for vk in feature_names if len(vk.split(' ')) == 1]
    bigrams = [vk for vk in feature_names if len(vk.split(' ')) == 2]
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' * 30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' * 30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")

# Best 5 keywords for each class using Description Features
from sklearn.feature_selection import chi2
import numpy as np

N = 5
for current_class in list(le.classes_):
    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_description, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_desc.get_feature_names())[indices]
    unigrams = [vk for vk in feature_names if len(vk.split(' ')) == 1]
    bigrams = [vk for vk in feature_names if len(vk.split(' ')) == 2]
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' * 30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' * 30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:3], df['Category'], random_state = 0)
X_train_title_features = tfidf_title.transform(X_train['Title']).toarray()
X_train_desc_features = tfidf_desc.transform(X_train['Description']).toarray()
features = np.concatenate([X_train_title_features, X_train_desc_features], axis=1)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(features, y_train)
# SVM
from sklearn import linear_model
svm = linear_model.SGDClassifier(loss='modified_huber',max_iter=1000, tol=1e-3).fit(features,y_train)
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=40,algorithm="SAMME").fit(features,y_train)



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 20000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50
# This is fixed.
EMBEDDING_DIM = 100

# Combining titles and descriptions into a single sentence
titles = df['Title'].values
descriptions = df['Description'].values
df_for_lstms = []
for i in range(len(titles)):
    temp_list = [titles[i], descriptions[i]]
    df_for_lstms.append(' '.join(temp_list))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_for_lstms)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Convert the df to padded sequences
X = tokenizer.texts_to_sequences(df_for_lstms)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of df tensor:', X.shape)

# One-hot Encode labels
Y = pd.get_dummies(df['Category']).values
print('Shape of label tensor:', Y.shape)

# Splitting into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 42)

# Define LSTM Model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Training LSTM Model
epochs = 5
batch_size = 64
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

import matplotlib.pyplot as plt
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();
