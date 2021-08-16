import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/training_data/Cleaned-28/final_cleaned1.csv')
print(df.shape)
df = df[['entid', 'CaseType']]
df.drop_duplicates(keep='first', inplace=True)
print(df.shape)
df = df.sort_values(['CaseType'], ascending=[True])

df_train = pd.DataFrame(columns=['entid', 'CaseType'])
df_test = pd.DataFrame(columns=['entid', 'CaseType'])

for i in range(5):
    print(i)
    temp = df[df['CaseType'] == i]
    num = temp.shape[0]
    print(num)
    mid = int(num * 0.7)
    temp = temp.sample(frac=1).reset_index(drop=True)
    temp_train, temp_test = temp.iloc[:mid, :], temp.iloc[mid:, :]
    df_train = pd.concat([df_train, temp_train], axis=0)
    df_test = pd.concat([df_test, temp_test], axis=0)

df_entire = pd.read_csv('data/training_data/Cleaned-28/final_cleaned1.csv')
df_entire = df_entire.sort_values(['entid'])
ids = df_train.iloc[:, 0]

first = df_train.iloc[0, 0]
df_train_return = df_entire[df_entire['entid'] == first]
for i in tqdm(df_train['entid'][1:]):
    temp = df_entire[df_entire['entid'] == i]
    df_train_return = pd.concat([df_train_return, temp], axis=0)

