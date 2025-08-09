import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import statistics

df = pd.read_csv('/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/2015.csv')

df['CELLFON3'] = df['CELLFON3'].replace(1, 0) # replace 1 in landline cellphone to 0 (not cellphone)
df['CELLFON3'] = df['CELLFON3'].replace(2, 1) # replace 2 in landline cellphone to 1 (is cellphone)
df['CELLPHONE'] = df['CELLFON3'].fillna(df['CELLFON2']) # combine cellphone columns
df['CELLPHONE'] = df['CELLPHONE'].replace(2, 0)
df.drop(['CELLFON3', 'CELLFON2'], axis=1, inplace=True)

df['PVTRESD_COMBINED'] = df['PVTRESD1'].fillna(df['PVTRESD2']) # combine private residence columns
df.drop(['PVTRESD1', 'PVTRESD2'], axis=1, inplace=True)

df['COLGHOUS_COMBINED'] = df['COLGHOUS'].fillna(df['CCLGHOUS']) # combine college housing columns
df.drop(['COLGHOUS', 'CCLGHOUS'], axis=1, inplace=True)

df['STATERES_COMBINED'] = df['STATERES'].fillna(df['CSTATE']) # combine state residence columns
df.drop(['STATERES', 'CSTATE'], axis=1, inplace=True)

df['LADULT'] = df['LADULT'].fillna(df['CADULT']) # combine adult columns
df.drop(['LADULT', 'CADULT'], axis=1, inplace=True)

df['NUMADULTS'] = df['NUMADULT'].fillna(df['HHADULT']) # combine num adults in household columns
df.drop(['NUMADULT', 'HHADULT'], axis=1, inplace=True)

# look for columns with b'' encoding and drop them
search_val = "b''"
cols = [col for col in df.columns if df[col].eq(search_val).any()]
df.drop(columns=cols, inplace=True)

df.drop(columns=['IDATE', 'IMONTH', 'IDAY', 'IYEAR'], inplace=True)

# rename final weight column to 'WEIGHT' and drop other weight columns
df.rename(columns={'_LLCPWT': 'WEIGHT'}, inplace=True)
df.drop(columns=['_STRWT', '_RAWRAKE', '_WT2RAKE'], inplace=True)

df

# drop any columns with > 25% missing values
threshold = 0.75*len(df)
df_clean = df.dropna(axis=1, thresh=threshold)

# removes 170 columns

# see how much data still missing
with pd.option_context('display.max_rows', None):
    print(df_clean.isna().sum())

# drop unnecessary weighting columns
# df_clean.drop(columns=['MSCODE', '_STSTR', '_DUALUSE','_STRWT', '_RAWRAKE', '_WT2RAKE'], inplace=True)

# drop redundant calculated columns
df_clean.drop(columns=['_RFHLTH', '_HCVU651', '_RFHYPE5', '_CHOLCHK', '_RFCHOL', '_MICHD', '_LTASTH1', 
                          '_CASTHM1', '_ASTHMS1', '_DRDXAR1', '_PRACE1', '_MRACE1','_RACEG21', '_RACEGR3', '_RACE_G1',
                          '_AGE65YR', '_AGEG5YR', '_AGE_G', 'HEIGHT3', 'HTIN4', 'HTM4', 'WEIGHT2', 'WTKG3', '_BMI5', '_RFBMI5',
                          'CHILDREN', 'EDUCA', 'INCOME2', 'SMOKE100', 'USENOW3', '_RFSMOK3', 'ALCDAY5',
                          'DRNKANY5', 'DROCDY3_', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1', 
                          '_MISFRTN', '_MISVEGN', '_FRTRESP', '_VEGRESP', '_FRTLT1', '_VEGLT1', '_FRT16', '_VEG23', '_FRUITEX', '_VEGETEX', 
                          'EXERANY2', 'STRENGTH', 'PAMISS1_', '_PA150R2', '_PA300R2', '_PA30021', '_PASTAE1', '_PAREC1', '_PASTAE1', 
                          '_LMTWRK1', '_LMTSCL1', 'SEATBELT', '_RFSEAT3', 'FLUSHOT6', 'PNEUVAC3', 'HIVTST6'], inplace=True)

# replace NUMADULTS > 6 with 6 and missing with 7
df_clean['NUMADULTS'] = np.minimum(df_clean['NUMADULTS'], 6)
df_clean['NUMADULTS'].fillna(7, inplace=True)

# replace missing values for numerical columns with mean
numeric_cols = ['FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_FRUTSUM', '_VEGESUM']
for col in numeric_cols:
    mean_val = df_clean[col].mean()
    df_clean[col].fillna(mean_val, inplace=True)

# replace missing values with value 10
categorical_cols = ['GENHLTH', 'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'CHOLCHK', 'TOLDHI2', 'CVDCRHD4', 'CHCSCNCR', 'HAVARTH3', 'INTERNET', 'QLACTLM2', 
                'USEEQUIP', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', '_LMTACT1', '_AIDTST3', 'NUMADULTS', 'VETERAN3']
for col in categorical_cols:
    df_clean[col].fillna(10, inplace=True)

df_clean['PHYSHLTH'].fillna(77, inplace=True) # replace missing PHYSHLTH value with 77
df_clean['PHYSHLTH'] = df_clean['PHYSHLTH'].replace(88, 0) # replace 'None' response with 0
df_clean['QSTLANG'].fillna((df_clean['QSTLANG'].mode().iloc[0]), inplace=True) # replace missing QSTLANG with mode
df_clean['STRFREQ_'].fillna(99000, inplace=True) # replace missing PHYSHLTH value with 77
df_clean['STATERES_COMBINED'].fillna((df_clean['STATERES_COMBINED'].mode().iloc[0]), inplace=True) # replace missing STATERES with mode

# clean target column
df_clean = df_clean.dropna(subset=['DIABETE3']) # drop 7 missing diabetes responses
df_clean = df_clean[df_clean['DIABETE3'] != 9] # drop refused to answer values for diabetes col
df_clean = df_clean[df_clean['DIABETE3'] != 7] # drop don't know/not sure answers
df_clean['DIABETE3'] = df_clean['DIABETE3'].replace(2, 3) # change answers 2 (yes, but only when pregnant) to 3 (no)

df_clean['DIABETE3'].hist()
plt.xlabel('Ever told you have diabetes?')
plt.ylabel('Frequency')
plt.show()

# replace values 3, 4 with 0, 2, respectively
df_clean['DIABETE3'] = df_clean['DIABETE3'].replace(3, 0)
df_clean['DIABETE3'] = df_clean['DIABETE3'].replace(4, 2)

# export multiclass target column data to csv
df_clean.to_csv('brfss_clean_multiclass.csv', index=False)

# convert diabetes column to binary (change prediabetes to has diabetes)
df_clean_bin = df_clean
df_clean_bin['DIABETE3'] = df_clean_bin['DIABETE3'].replace(2,1)
df_clean_bin['DIABETE3'].unique()

# export binary target column data
df_clean_bin.to_csv('brfss_clean_binary.csv', index=False)

# create new df with selected features
df_sparse = df_clean[['PHYSHLTH', 'GENHLTH', 'BPHIGH4', 'TOLDHI2', '_AGE80', '_BMI5CAT', 
    'MAXVO2_', 'FC60_', 'CHOLCHK', 'EMPLOY1', 'CVDINFR4', 'DIABETE3', 'WEIGHT']]

# replace values 7, 9, and 10 with nan
df_sparse[['GENHLTH', 'TOLDHI2', 'CHOLCHK', 'CVDINFR4', 'BPHIGH4']] = df[
    ['GENHLTH', 'TOLDHI2', 'CHOLCHK', 'CVDINFR4', 'BPHIGH4']].replace(
        [7, 9, 10], np.nan)

df_sparse['PHYSHLTH'] = df_sparse['PHYSHLTH'].replace([77,99], np.nan) # replace 77 and 99 with nan
df_sparse['MAXVO2_'] = df_sparse['MAXVO2_'].replace(99900, np.nan) # replace 99900 with nan
df_sparse['FC60_'] = df_sparse['FC60_'].replace(99900, np.nan) # replace 99900 with nan
df_sparse['EMPLOY1'] = df_sparse['EMPLOY1'].replace(9, np.nan) # replace 9 with nan
df_sparse['BPHIGH4'] = df_sparse['BPHIGH4'].replace(2, 3) # replace BPHIGH 2 (only during pregnancy) with 3 (no)
df_sparse['BPHIGH4'] = df_sparse['BPHIGH4'].replace(4, 1) # replace BPHIGH 4 (pre-hypertensive) with 1 (yes)
df_sparse['BPHIGH4'] = df_sparse['BPHIGH4'].replace(3, 0) # change no label to 0
df_sparse['TOLDHI2'] = df_sparse['TOLDHI2'].replace(2, 0) # change no label to 0
df_sparse['CVDINFR4'] = df_sparse['CVDINFR4'].replace(2, 0) # change no label to 0

# see how much data missing per column
total_rows = len(df_sparse)
missing_counts = df_sparse.isna().sum()
missing_percentage = (missing_counts / total_rows) * 100
print(missing_percentage)

df_sparse.describe()

df_sparse['PHYSHLTH'].hist()
plt.xlabel('PHYSHLTH')
plt.ylabel('Frequency')
plt.show()

df_sparse['GENHLTH'].hist()
plt.xlabel('GENHLTH')
plt.ylabel('Frequency')
plt.show()

df_sparse['BPHIGH4'].hist()
plt.xlabel('BPHIGH4')
plt.ylabel('Frequency')
plt.show()

df_sparse['TOLDHI2'].hist()
plt.xlabel('TOLDHI2')
plt.ylabel('Frequency')
plt.show()

df_sparse['_AGE80'].hist()
plt.xlabel('_AGE80')
plt.ylabel('Frequency')
plt.show()

df_sparse['MAXVO2_'].hist()
plt.xlabel('MAXVO2')
plt.ylabel('Frequency')
plt.show()

df_sparse['FC60_'].hist()
plt.xlabel('FC60')
plt.ylabel('Frequency')
plt.show()

df_sparse['CHOLCHK'].hist()
plt.xlabel('CHOLCHK')
plt.ylabel('Frequency')
plt.show()

df_sparse['EMPLOY1'].hist()
plt.xlabel('EMPLOY1')
plt.ylabel('Frequency')
plt.show()

df_sparse['CVDINFR4'].hist()
plt.xlabel('CVDINFR4')
plt.ylabel('Frequency')
plt.show()

df_sparse.to_csv('brfss_sparse_with_nan.csv', index=False)

# stratified split, designate weight column
seed = np.random.seed(6)
X = df_sparse.drop(columns=['DIABETE3', 'WEIGHT'],  axis=1)
y = df_sparse['DIABETE3']
w = df_sparse['WEIGHT']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.3, stratify=y, random_state=seed)

# export to csv
X_train.to_csv('X_train_sparse.csv', index=False)
X_test.to_csv('X_test_sparse.csv', index=False)
y_train.to_csv('y_train_sparse.csv', index=False)
y_test.to_csv('y_test_sparse.csv', index=False)
w_train.to_csv('w_train_sparse.csv', index=False)
w_test.to_csv('w_test_sparse.csv', index=False)
