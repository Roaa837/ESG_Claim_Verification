import pandas as pd

df = pd.read_parquet(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation_esg")

# label distribution
print("Label distribution:")
print(df["esg_label"].value_counts())
print()

# check score ranges
print("Score stats:")
print(df.groupby("esg_label")["esg_score"].describe())
print()

# sample 5 sentences from each label — check they make sense
for label in ["Environmental", "Social", "Governance"]:
    print(f"--- {label} samples ---")
    sample = df[df["esg_label"] == label]["text"].sample(5).tolist()
    for s in sample:
        print(f"  - {s[:120]}")
    print()

# check high confidence sentences — should be very clearly ESG
print("--- High confidence Environmental (score > 0.95) ---")
high_e = df[(df["esg_label"] == "Environmental") & (df["esg_score"] > 0.95)]
for s in high_e["text"].sample(5).tolist():
    print(f"  - {s[:120]}")
print()

# check low confidence sentences — borderline cases
print("--- Low confidence labels (score 0.5-0.6) ---")
low = df[(df["esg_score"] >= 0.5) & (df["esg_score"] < 0.6)]
for s in low["text"].sample(5).tolist():
    print(f"  - {s[:120]}")
print()

# check unlabelled — should be non-ESG content
print("--- Unlabelled samples ---")
unlabelled = df[df["esg_label"].isna()]
for s in unlabelled["text"].sample(10).tolist():
    print(f"  - {s[:120]}")