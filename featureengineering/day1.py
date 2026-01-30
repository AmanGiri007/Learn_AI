import pandas as pd

url="https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"

df=pd.read_csv(url)

#display datasets
print("Dataset Info:\n")
print(df.info())

#preview the first few rows
print("\n Dataset Preview:\n")
print(df.head())

#separate featured
categorical_features=df.select_dtypes(include=["object"]).columns
numerical_features=df.select_dtypes(include=["int64","float64"]).columns

print("\nCategorical features:",categorical_features.tolist())
print("\nNumerical features:",numerical_features.tolist())

#display summaru of categorical features
print("\n Categorical Feature Summary:\n")
for col in categorical_features:
    print(f"{col}:\n",df[col].value_counts(),"\n")

#display summary of numerical features
print("\nNumerical Features Summary\n")
print(df[numerical_features].describe())
