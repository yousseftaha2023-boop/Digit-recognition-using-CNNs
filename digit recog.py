import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#reading data
df_train = pd.read_csv('C:\\Users\\user\\Downloads\\train_dig.csv')
df_test = pd.read_csv('C:\\Users\\user\\Downloads\\test_dig.csv')
print(df_train.head())
print(df_train.shape)
columns_with_missings = df_train.columns.where(df_train.isnull().sum()/len(df_train) > 0).dropna().to_list()
print(f'count of columns with missings', len(columns_with_missings))
missings= (df_train[columns_with_missings].isnull().sum()/len(df_train)*100).sort_values(ascending=False , inplace=False)
print(f'missing %', missings)
print('descriptive statistics of train set', df_train.describe())
columns_with_missings_test = df_test.columns.where(df_test.isnull().sum()/len(df_test) > 0).dropna().to_list()
print(f'count of columns with missings', len(columns_with_missings_test))
missings_test= (df_test[columns_with_missings_test].isnull().sum()/len(df_test)*100).sort_values(ascending=False , inplace=False)
print(f'missing %', missings_test)




#test_split
from sklearn.model_selection import train_test_split
# df_train['label'] = df_train['label'].astype('object')
df_tran_train , df_train_test = train_test_split(df_train, test_size=0.2, random_state=42 , stratify=df_train['label'])
df_train , df_valid = train_test_split(df_tran_train, test_size=0.12, random_state=42 , stratify=df_tran_train['label'])

x_train = df_train.drop(columns=['label'])
y_train = df_train['label']
x_valid = df_valid.drop(columns=['label'])
y_valid = df_valid['label']
x_train_test = df_train_test.drop(columns=['label'])
y_train_test = df_train_test['label']
x_test = df_test
brightest_pixel_train = x_train.values.max()
x_train_image = (x_train.values.reshape(-1, 28, 28, 1)/brightest_pixel_train).astype(np.float32)  
x_valid_image = (x_valid.values.reshape(-1, 28, 28, 1)/brightest_pixel_train).astype(np.float32)
x_train_test_image = (x_train_test.values.reshape(-1, 28, 28, 1)/brightest_pixel_train).astype(np.float32)
x_test_image = (x_test.values.reshape(-1, 28, 28, 1)/brightest_pixel_train).astype(np.float32)


#neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D , Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import roc_auc_score
import random
tf.random.set_seed(42)
np.random.seed(42)
model = Sequential([
   Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)) ,
   MaxPooling2D(pool_size=(2, 2)) ,
   Conv2D(64, (3, 3), activation='relu') ,
   MaxPooling2D(pool_size=(2, 2)) ,
   Flatten() ,
   Dense(64 ,  activation='relu') ,
   Dense(10 , activation='linear' )
   
])
model.compile(loss = SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(learning_rate=0.0002) , metrics = [tf.keras.metrics.AUC(multi_label=True)])
model.fit(x_train_image, y_train,validation_data=(x_valid_image, y_valid), epochs=20, batch_size=32)
y_train_logits = model.predict(x_train_image)
y_valid_logits = model.predict(x_valid_image)
y_train_test_logits = model.predict(x_train_test_image)
y_test_logits = model.predict(x_test_image)

y_train_probs = tf.keras.activations.softmax(y_train_logits).numpy()
y_valid_probs = tf.keras.activations.softmax(y_valid_logits).numpy()
y_train_test_probs = tf.keras.activations.softmax(y_train_test_logits).numpy()
y_test_probs = tf.keras.activations.softmax(y_test_logits).numpy()
y_tes_labels =np.argmax(y_test_probs, axis=1)
y_test_submission = pd.DataFrame( {'ImageId' : np.arange(1,len(y_test_probs)+1) , 'Label' : y_tes_labels})
y_test_submission.to_csv('C:\\Users\\user\\Downloads\\kaggle\\kaggle_submission_digits.csv', index=False)

train_score = roc_auc_score(y_train, y_train_probs , multi_class='ovr', average='macro')
valid_score = roc_auc_score(y_valid, y_valid_probs , multi_class='ovr', average='macro')
test_score = roc_auc_score(y_train_test, y_train_test_probs , multi_class='ovr', average='macro')

print(f'AUC train set is {train_score:.4f}')
print(f'AUC valid set is {valid_score:.4f}')
print(f'AUC test set is {test_score:.4f}')



