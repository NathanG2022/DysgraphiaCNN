# Compatibility layer between Python 2 and Python 3
from __future__ import print_function
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D


# %%

def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def show_basic_dataframe_info(dataframe,
                              preview_rows=20):
    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    print(dataframe.describe())


def read_data(file_path):
    column_names = ['user-id',
                    'activity',
                    'x-axis',
                    'y-axis',
                    'timestamp',
                    'is-on-surface',
                    'azimuth',
                    'altitude',
                    'pressure']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)

    df['x-axis'] = df['x-axis'].astype(float)
    df['y-axis'] = df['y-axis'].astype(float)
    df['is-on-surface'] = df['is-on-surface'].astype(float)
    df['azimuth'] = df['azimuth'].astype(float)
    df['altitude'] = df['altitude'].astype(float)
    df['pressure'] = df['pressure'].astype(float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)
    return df


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6,
                                                       figsize=(15, 10),
                                                       sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['is-on-surface'], 'is-on-surface')
    plot_axis(ax3, data['timestamp'], data['azimuth'], 'azimuth')
    plot_axis(ax4, data['timestamp'], data['altitude'], 'altitude')
    plot_axis(ax5, data['timestamp'], data['pressure'], 'pressure')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def create_segments_and_labels(df, time_steps, step, label_name):
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        os = df['is-on-surface'].values[i: i + time_steps]
        az = df['azimuth'].values[i: i + time_steps]
        al = df['altitude'].values[i: i + time_steps]
        pr = df['pressure'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        segments.append([xs, ys, os, az, al, pr])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_FEATURES, time_steps).transpose((0, 2, 1))
    labels = np.asarray(labels)
    return reshaped_segments, labels


# %%

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

# The number of steps within one time segment
TIME_PERIODS = 400
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 200

# %%

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df = read_data('./PreProcessing/data.csv')

# Describe the data
show_basic_dataframe_info(df, 20)

df['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()

df['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()

for activity in np.unique(df["activity"]):
    subset = df[df["activity"] == activity][:1800]
    plot_activity(activity, subset)

# Define column name of the label vector
LABEL = "ActivityEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df["activity"].values.ravel())

# %%

print("\n--- Reshape the data into segments ---\n")

# Normalize features for training data set
df['x-axis'] = feature_normalize(df['x-axis'])
df['y-axis'] = feature_normalize(df['y-axis'])
df['is-on-surface'] = feature_normalize(df['is-on-surface'])
df['azimuth'] = feature_normalize(df['azimuth'])
df['altitude'] = feature_normalize(df['altitude'])
df['pressure'] = feature_normalize(df['pressure'])
# Round in order to comply to NSNumber from iOS
df = df.round({'x-axis': 6, 'y-axis': 6, 'is-on-surface': 6, 'azimuth': 6, 'altitude': 6, 'pressure': 6})

# Differentiate between test set and training set
df_test = df[(df['user-id'] < 64) & (df['user-id'] > 18)]
df_train = df[(df['user-id'] > 64) | (df['user-id'] < 18)]

show_basic_dataframe_info(df_train, 20)

# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

# %%

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')

# Inspect y data
print('y_train shape: ', y_train.shape)

# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
print(list(le.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (num_time_periods * num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding of y_train labels (only execute once!)
# y_train = to_categorical(y_train, num_classes)
y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
print('New y_train shape: ', y_train.shape)

# %%

print("\n--- Create neural network model ---\n")

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 20, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 20, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 20, activation='relu'))
model_m.add(Conv1D(160, 20, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(1, activation='sigmoid'))
print(model_m.summary())

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True)
    #    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1

]

model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 64
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

print(history)

# %%

print("\n--- Check against test data ---\n")

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(x_test)

# Take the class with the highest probability from the test predictions
max_y_pred_test = (y_pred_test[:, 0] + 0.5).astype(int)
max_y_pred_test = max_y_pred_test.reshape(-1, 1)
max_y_test = y_test[:, 0].astype(int)
max_y_test = max_y_test.reshape(-1, 1)

max_tt = []
for i in range(0, len(df_test) - TIME_PERIODS, STEP_DISTANCE):
    a = stats.mode(df_test['user-id'][i: i + TIME_PERIODS])[0]
    max_tt.append(a)
max_tt = np.asarray(max_tt).reshape(-1, 1)

arr = np.concatenate((max_tt, max_y_test, max_y_pred_test), axis=1).astype(int)
df2 = pd.DataFrame(arr)
df2.to_csv('result.csv', header=False, index=False)

# %%

print("\n--- Classification report for test data ---\n")

column_names = ['user-id',
                'y-test',
                'y-pred'
                ]
df = pd.read_csv('result.csv',
                 header=None,
                 names=column_names)

for user in df['user-id'].unique():
    t = df.loc[df['user-id'] == user]
    a = stats.mode(t['y-pred'])
    aa = stats.describe(t['y-pred'])
    b = stats.mode(t['y-test'])
    if a.mode == b.mode:
        print('1', a.count, (a.count * aa.mean).astype(int), b.mode)
    else:
        print('0', a.count, (a.count * aa.mean).astype(int), b.mode)
