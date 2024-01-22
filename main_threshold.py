from itertools import combinations
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix ###### pip install sci-learn
from sklearn.model_selection import train_test_split

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# %%
def show_basic_dataframe_info(dataframe, preview_rows=20):
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")


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
    df['x-axis'] = df['x-axis'].astype("float32")
    df['y-axis'] = df['y-axis'].astype("float32")
    df['is-on-surface'] = df['is-on-surface'].astype("float32")
    df['azimuth'] = df['azimuth'].astype("float32")
    df['altitude'] = df['altitude'].astype("float32")
    df['pressure'] = df['pressure'].astype("float32")

    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)
    return df


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


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
    # Set subtitle based on activity value
    if activity == 1:
        subtitle = "Neurotypical"
    elif activity == 0:
        subtitle = "Dysgraphia"
    else:
        subtitle = str(activity)  # Default to the activity number if not 0 or 1
    fig.suptitle(subtitle, fontsize=14)
    plt.subplots_adjust(top=0.90)
    plt.show()



def create_segments(df, time_steps, step):
    segments = []
    labels = []
    person_ids = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        os = df['is-on-surface'].values[i: i + time_steps]
        az = df['azimuth'].values[i: i + time_steps]
        al = df['altitude'].values[i: i + time_steps]
        pr = df['pressure'].values[i: i + time_steps]
        ac = stats.mode(df['activity'][i: i + time_steps])[0]
        id = stats.mode(df['user-id'][i: i + time_steps])[0]
        segments.append([xs, ys, os, az, al, pr])
        labels.append(ac)
        person_ids.append(id)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_FEATURES, time_steps ).transpose((0,2,1))
    reshaped_segments = reshaped_segments.reshape(-1, INPUT_SHAPE)
    labels = np.asarray(labels)
    person_ids = np.asarray(person_ids)
    # Convert type for Keras otherwise Keras cannot process the data
    reshaped_segments = reshaped_segments.astype("float32")
    labels = labels.astype("float32")

    return reshaped_segments, labels, person_ids


class PersonLevelEvaluation(Callback):
    def __init__(self, validation_data, person_ids, threshold):
        super().__init__()
        self.validation_data = validation_data
        self.person_ids = person_ids
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        segment_predictions = self.model.predict(x_val)
        aggregated_preds, aggregated_labels = self.aggregate_predictions(segment_predictions, 
                                                                         y_val, self.person_ids, 
                                                                         self.threshold)
        person_level_accuracy = accuracy_score(aggregated_labels, aggregated_preds)
        print(f"Epoch {epoch+1}: Person-level Accuracy is {person_level_accuracy:.4f}")

    @staticmethod
    def aggregate_predictions(predictions, labels, person_ids, threshold):
        person_predictions = {}
        person_true_labels = {}

        for pred, label, person_id in zip(predictions, labels, person_ids):
            if person_id not in person_predictions:
                person_predictions[person_id] = []
                person_true_labels[person_id] = label  # Assuming all segments of a person have the same label

            # Convert the prediction to a binary value (0 or 1)
            binary_pred = int(np.round(pred[0]))  # Convert numpy float to Python int
            person_predictions[person_id].append(binary_pred)

        aggregated_predictions = []
        aggregated_true_labels = []

        for person_id, binary_preds in person_predictions.items():
            # Calculate the percentage of '1' predictions
            percent_ones = sum(binary_preds) / len(binary_preds)
            # If the percentage of '1's is above the threshold, classify as '1', else '0'
            final_prediction = 1 if percent_ones >= threshold else 0
            aggregated_predictions.append(final_prediction)
            aggregated_true_labels.append(person_true_labels[person_id])

        return np.array(aggregated_predictions), np.array(aggregated_true_labels)


def load_and_evaluate(df_train, df_test, threshold):
    model_path = 'best_model.43-0.8513.tf'
    model_m = keras.models.load_model(model_path)

    print("\n--- Check against test data ---\n")
    x_test, y_test, test_person_ids = create_segments(df_test, TIME_PERIODS, STEP_DISTANCE)
    print('x_test shape:', x_test.shape)
    print('y_test shape: ', y_test.shape)

    segment_predictions = model_m.predict(x_test)

    # Convert probabilities to binary labels if necessary
    binary_segment_predictions = [int(np.round(pred[0])) for pred in segment_predictions]
    # Calculate accuracy if y_test is available
    segment_level_accuracy = accuracy_score(y_test, binary_segment_predictions)
    print(f"Segment-level accuracy on test data: {segment_level_accuracy:.4f}")

    person_level_evaluator = PersonLevelEvaluation(validation_data=None, person_ids=None, threshold=threshold)
    aggregated_preds, aggregated_labels = person_level_evaluator.aggregate_predictions(segment_predictions, 
                                                                                       y_test, 
                                                                                       test_person_ids, 
                                                                                       threshold)
    person_level_accuracy = accuracy_score(aggregated_labels, aggregated_preds)
    print("\nPerson-level accuracy on test data: %0.4f" % person_level_accuracy)

     # Calculate confusion matrix
    conf_matrix_person = confusion_matrix(aggregated_labels, aggregated_preds)
    # Extract FP and FN
    fp_person = conf_matrix_person[0][1]
    fn_person = conf_matrix_person[1][0]
    print(f"Person Level - False Positives: {fp_person}, False Negatives: {fn_person}")
    print(conf_matrix_person)

    return person_level_accuracy


def split_and_evaluate(df):
    # Splitting based on user-id and activity
    unique_users_0 = df[df['activity'] == 0]['user-id'].unique()
    unique_users_1 = df[df['activity'] == 1]['user-id'].unique()

    # Randomly split user-ids into 5 sets while maintaining the activity ratio
    sets_0 = np.array_split(np.random.permutation(unique_users_0), 5)
    sets_1 = np.array_split(np.random.permutation(unique_users_1), 5)
    print("sets_0:", sets_0)
    print("sets_1:", sets_1)

    best_accuracy = 0
    best_split = None

    for training_indices in combinations(range(5), 4):
        testing_indices = [x for x in range(5) if x not in training_indices]

        # Combine user-ids for training and testing
        train_user_ids = np.concatenate([sets_0[i] for i in training_indices] + [sets_1[i] for i in training_indices])
        test_user_ids = np.concatenate([sets_0[i] for i in testing_indices] + [sets_1[i] for i in testing_indices])

        # Create training and testing dataframes
        df_train = df[df['user-id'].isin(train_user_ids)]
        df_test = df[df['user-id'].isin(test_user_ids)]

        # Train the model and evaluate
        person_level_accuracy = train_and_evaluate(df_train, df_test)

        # Store the best result
        if person_level_accuracy > best_accuracy:
            best_accuracy = person_level_accuracy
            best_split = (training_indices, testing_indices)

    print(f"Best person-level accuracy: {best_accuracy}")
    print(f"Best split: Training sets {best_split[0]}, Testing sets {best_split[1]}")


# %%
# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------
# Set some standard parameters upfront
pd.options.display.float_format = '{:.4f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

# The number of steps within one time segment
TIME_PERIODS = 800
# The steps to take from one segment to the next; No overlap between the segments if same as TIME_PERIODS
STEP_DISTANCE = 400
# Hyper-parameters
BATCH_SIZE = 64
EPOCHS = 50
N_FEATURES = 6
INPUT_SHAPE = N_FEATURES * TIME_PERIODS

# %%
print("\n--- Load, inspect and transform data ---\n")
# Load data set containing all the data from csv
df = read_data('data.csv')
# Describe the data
# print("First 20 rows of the df dataframe:\n")
# show_basic_dataframe_info(df, 20)

""" ################ Plotting ###########################
ax = df['activity'].value_counts().plot(kind='bar', title='Data Samples by Participant Diagnosis')
ax.set_xlabel('')
ax.set_xticks(ticks=[0, 1])
ax.set_xticklabels(labels=["Neurotypical", "Dysgraphia"], fontsize=12, rotation=0)
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.show()

ax = df['user-id'].value_counts().plot(kind='bar', title='Data Samples Per Participant')
ax.set_xticklabels([])
ax.set_xlabel('Participants')
ax.set_ylabel('Samples')
plt.show()

for activity in np.unique(df["activity"]):
    subset = df[df["activity"] == activity][:1800]
    plot_activity(activity, subset)
################# End Plotting ####################### """

# %%
print("\n--- Reshape the data into segments ---\n")
# Normalize features for training data set
df['x-axis'] = feature_normalize(df['x-axis'])
df['y-axis'] = feature_normalize(df['y-axis'])
df['is-on-surface'] = feature_normalize(df['is-on-surface'])
df['azimuth'] = feature_normalize(df['azimuth'])
df['altitude'] = feature_normalize(df['altitude'])
df['pressure'] = feature_normalize(df['pressure'])

# split_and_evaluate(df)

train_user_ids = [51,  89,  52, 184, 160,  56,  82, 177,  85, 167,  63,  54, 173, 
                  181, 178,  61,  77,  58, 170, 183, 165, 190,  74, 172, 166, 168, 
                  186,  84,  73, 163,  88, 182, 192,  71, 180,  67, 161,  83,  72, 
                  158,  60,  78, 179,  87, 169,  65, 171, 185, 100, 162, 189,
                  149,   8,  15, 115, 118, 125, 110, 139, 104,  95,  91,  17, 
                  130,  92,   7,  29, 141,  38, 148, 129,  26, 128,  11, 
                  22, 122, 133, 131, 135,  42,  93,  19, 120,  48,  14,
                  124, 109, 108, 114,  21, 113, 107,  49,   6, 121, 119]
test_user_ids = [90,  70, 176,  69,  50, 159,  66, 187, 191,  57,  79,  76,  62,
                 39, 101, 112, 134,  16,  36,  99,  44,  32,  13, 150]
# Create training and testing dataframes
df_train = df[df['user-id'].isin(train_user_ids)]
df_test = df[df['user-id'].isin(test_user_ids)]

# Load the model and evaluate
for threshold in np.arange(0.40, 0.60, 0.05):
    person_level_accuracy = load_and_evaluate(df_train, df_test, threshold)

# for BATCH_SIZE in [64, 128, 400]:
#     for i in range(10):
#         person_level_accuracy = train_and_evaluate(df_train, df_test, 0.5)
#         print(f"BATCH_SIZE {BATCH_SIZE} Run {i} - Person-level accuracy: {person_level_accuracy}")