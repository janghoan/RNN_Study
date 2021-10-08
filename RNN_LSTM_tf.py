# Data preprocessing

import pandas as pd
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('./train_data/ArticlesApril2018.csv')
df.head()

print('열의 개수 :', len(df.columns))
print(df.columns)

df['headline'].isnull().values.any()

headline = []
headline.extend(list(df.headline.values))
headline[:5]

print('총 샘플의 개수 : {}'.format(len(headline)))

headline = [n for n in headline if n!='Unknown']
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline)))

headline[:5]

def repreprocessing(s):
    s = s.encode('utf8').decode('ascii','ignore')
    return ''.join(c for c in s if c not in punctuation).lower()

text = [repreprocessing(x) for x in headline]


text[:5]

# make train data

t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1
print('단어 집합의 크기 : %d' %vocab_size)

sequences = list()

for line in text:
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print(sequences)

# make train label

index_to_word = {}
for key, value in t.word_index.items():
    index_to_word[value] = key

print('빈도수 상위 582번 단어 : {}'.format(index_to_word[582]))

max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))

sequences = pad_sequences(sequences, maxlen = max_len, padding = 'pre')
print(sequences[:3])

sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

print(X[:3])

print(y[:3]) # 레이블

y = to_categorical(y, num_classes=vocab_size)

# make model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

model = Sequential()






