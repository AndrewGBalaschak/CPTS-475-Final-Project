import pandas as pd
from tabgan.sampler import GANGenerator

csv_file_path = 'data/cleaned/NF-UQ-NIDS-ATTACKS'
df = pd.read_csv(csv_file_path, nrows=1000)
num_rows = len(df)
print(f"Number of rows in the CSV file: {num_rows}")
print("Finished reading dataset")

numerical_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'Label']

target_col = 'Label'

gen_params = {
    "batch_size": 10,
    "patience": 25,
    "epochs": 100,
}
print("Set values")

# Include the target column in categorical_cols
categorical_cols.append(target_col)

new_train, new_target = GANGenerator(
    gen_x_times=0.1,
    cat_cols=categorical_cols,
    bot_filter_quantile=0.001,
    top_filter_quantile=0.999,
    is_post_process=True,
    gen_params=gen_params
).generate_data_pipe(df.drop(target_col, axis=1), pd.DataFrame(df[target_col]), df.drop(target_col, axis=1))

print("Synthetic Dataset:")
print(new_train.head())

synthetic_csv_path = 'data/cleaned/synthetic_dataset2.csv'
new_train.to_csv(synthetic_csv_path, index=False)

print(f"Synthetic dataset saved to {synthetic_csv_path}")
