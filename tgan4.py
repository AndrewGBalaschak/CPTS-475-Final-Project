# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabgan.sampler import GANGenerator
import numpy as np


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.losses import mean_squared_error as old_mean_squared_error


# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# %%
def preprocess_data(df):
    non_numeric_cols = df.select_dtypes(exclude=['number', 'bool']).columns

    le = LabelEncoder()
    df[non_numeric_cols] = df[non_numeric_cols].apply(lambda x: le.fit_transform(x))

    df[non_numeric_cols] = df[non_numeric_cols].astype(int)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

# %%

# Load dataset
dataset_path = "data/cleaned/NF-UQ-NIDS-ATTACKS"
target_column = "Label"  

df_pre = pd.read_csv(dataset_path)
df = preprocess_data(df_pre)


COLS_USED = list(df.columns)
COlS_TRAIN = list(df.columns).remove(target_column)


df = df[COLS_USED]
print(COLS_USED)


# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.head())

# %%

# Split into training and test sets
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df.drop(target_column, axis=1),
    df[target_column],
    test_size=0.20,
    random_state=42
)

# %%

# Create dataframe 
df_x_test, df_y_test = df_x_test.reset_index(drop=True), df_y_test.reset_index(drop=True)
df_y_train = pd.DataFrame(df_y_train)
df_y_test = pd.DataFrame(df_y_test)

# %%
x_train = df_x_train.values
x_test = df_x_test.values
y_train = df_y_train.values
y_test = df_y_test.values

# %%
model = Sequential()
model.add(Dense(50, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# %%
second_row_list = df.iloc[1].tolist()
print(second_row_list)

# %%
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto',
                        restore_best_weights=True)

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          callbacks=[monitor], verbose=2, epochs=10)

# %%
gen_x, gen_y = GANGenerator().generate_data_pipe(df_x_train, df_y_train,
                                                  df_x_test, deep_copy=True,
                                                  only_adversarial=False,
                                                  use_adversarial=True)

# %%
gen_df = pd.concat([gen_x, pd.DataFrame(gen_y, columns=[target_column])], axis=1)
gen_df.to_csv("synthetic_dataset.csv", index=False)

# %%
print(gen_df.head())


# %%
unique_count = gen_df['Attack'].nunique()

print(f"Number of unique entries in the column: {unique_count}")

# %%
def preprocess_data(df):

    non_numeric_cols = df.select_dtypes(exclude=['number', 'bool']).columns


    encoder_mapping = {}

    for column in non_numeric_cols:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoder_mapping[column] = le


    df[non_numeric_cols] = df[non_numeric_cols].astype(int)


    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return encoder_mapping

# %%
def reverse_encoding(df, encoder_mapping):
    reversed_df = df.copy()

    for column, encoder in encoder_mapping.items():
        reversed_df[column] = encoder.inverse_transform(df[column])

    columns_to_exclude = ['Unnamed: 0', 'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'Label']
    bool_cols = reversed_df.select_dtypes(include='int').columns.difference(columns_to_exclude)
    reversed_df[bool_cols] = reversed_df[bool_cols].astype(bool)

    return reversed_df

# %%
encoder = preprocess_data(pd.read_csv(dataset_path))
newdf = reverse_encoding(gen_df, encoder)
newdf.to_csv("synthetic_dataset_decoded.csv", index=False)

# %%
newdf.to_csv("synthetic_dataset_decoded.csv", index=False)

# %%
print(newdf.head())



