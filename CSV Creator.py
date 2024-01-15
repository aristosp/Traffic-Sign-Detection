import pandas as pd
from torch_snippets import Glob

# Read class labels and convert to integers
classIDs = pd.read_csv('signnames.csv')
classIDs.set_index('ClassId', inplace=True)
# Convert to dictionary the SignName column
classIDs = classIDs.to_dict()['SignName']
# Expand the keys to 5 digit (same as the filepaths)
classIDs = {f'{k:05d}': v for k, v in classIDs.items()}
# Create a dictionary containing the string and integer labels
id2int = {v: ix for ix, (k, v) in enumerate(classIDs.items())}
# Convert to dataframe and add column names
df = pd.DataFrame(id2int.items())
df.columns = ['Sign Name', '1 Digit Int']
df2 = pd.DataFrame(classIDs.items())
df2.columns = ['5 Digit Int Label', 'Sign Name']
# Get all training files
all_files = Glob('Final_Training/Images/*/*.ppm')
int_class = {}
mydict = {}
# For each file finds its 5 digit label and its 1 digit label and save it to a dictionary
for file in all_files:
    classID = file.parent.name
    mydict[file] = classID
    # Special cases where strip removes all zeros
    if classID == '00000':
        int_id = 0
    elif classID == '00010':
        int_id = 10
    elif classID == '00020':
        int_id = 20
    elif classID == '00030':
        int_id = 30
    elif classID == '00040':
        int_id = 40
    else:
        int_id = int(classID.strip("0"))
    int_class[file] = int_id
# Convert dictionary to dataframe and name the two columns
df3 = pd.DataFrame(mydict.items())
df3.columns = ['Filepath', '5 Digit Int Label']
df4 = pd.DataFrame(int_class.items())
df4.columns = ['Filepath', '1 Digit Int Label']
# Merge the 2 dataframes, reorder the columns and add 1 digit integer
final_df = pd.merge(df2, df3, on='5 Digit Int Label')
final_df = final_df.reindex(columns=['Filepath', 'Sign Name', '5 Digit Int Label'])
# Convert 1 Digit Int column to Integers
final_df = pd.concat([final_df, pd.to_numeric(df4['1 Digit Int Label'])], axis=1)
final_df['1 Digit Int Label'] = final_df['1 Digit Int Label'].fillna(0).astype('int')
# Save to csv file
final_df.to_csv('train.csv', sep=',', index=False)
