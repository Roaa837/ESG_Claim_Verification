import pandas as pd
print("imported")

df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation")

# basic shape
print(df.shape)

# check metadata is filled correctly
print(df[["company_name", "year", "report_type"]].drop_duplicates())

# check a few sentences look clean
print(df["text"].iloc[0])
print(df["text"].iloc[100])
print(df["text"].iloc[1000])

# check segments per company
print(df.groupby("company_name").size())

# check segments per document
print(df.groupby("source_document").size())

# check for empty or very short text
print("Empty rows:", df["text"].isna().sum())
print("Short rows (<20 chars):", (df["text"].str.len() < 20).sum())

# print 10 random sentences
print(df["text"].sample(10).to_list())

# check sentences that mention emissions (should look like real ESG claims)
emissions = df[df["text"].str.contains("emission|carbon|CO2", case=False)]
print(emissions["text"].head(10).to_list())

# check for remaining noise — very repetitive or suspiciously short
print(df[df["text"].str.len() < 60]["text"].sample(10).to_list())