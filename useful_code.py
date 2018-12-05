############################################################
# Read Specific columsn from CSV and extract data
data = pd.read_csv("filename.csv",encoding = "ISO-8859-1")
train_size = int(len(data)*0.8)
feat_len = input_number.
train_test = data['col_name'][:train_size]
X = pd.read_csv("training_data.csv", encoding = "ISO-8859-1", usecols=['col1', 'col2', 'col3'])
y = pd.read_csv("training_data.csv", encoding = "ISO-8859-1", usecols=['col1', 'col2'])

# In case you want to drop some rows:
X.drop([0])
y.drop([0])
###############################################################
