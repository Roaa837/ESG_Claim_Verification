import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

INPUT_FILE  = "data/processed/segmentation"
OUTPUT_FILE = "data/processed/segmentation_esg"

# batch size — lower to 16 if you run out of memory
BATCH_SIZE = 32

# minimum confidence to assign a label, otherwise stays None
MIN_SCORE = 0.5

print("ESG classification running...")


# governance keywords to boost GovernanceBERT where it underperforms
GOVERNANCE_KEYWORDS = [
    "board", "director", "audit", "committee", "shareholder", "executive",
    "compensation", "remuneration", "compliance", "ethics", "transparency",
    "anti-corruption", "whistleblower", "governance", "oversight", "fiduciary",
    "ceo", "cfo", "chairman", "supervisory", "voting", "proxy", "disclosure",
    "accountability", "stewardship", "nomination", "independent director",
]


def load_classifiers():
    # load all three ESGBERT models
    e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
    s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social",              truncation=True, max_length=512)
    g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance",      truncation=True, max_length=512)
    return e_clf, s_clf, g_clf


def get_pillar_score(result, pillar_label):
    # if model label matches the pillar, use score directly
    # if model returned "none", flip the score
    label = result["label"].lower()
    score = result["score"]
    if label == pillar_label.lower():
        return score
    else:
        return 1 - score


def get_governance_score(result, text):
    # start with the corrected model score
    model_score = get_pillar_score(result, "governance")
    # boost with keyword matching where model underperforms
    text_lower = text.lower()
    matches = sum(1 for kw in GOVERNANCE_KEYWORDS if kw in text_lower)
    keyword_boost = min(matches * 0.15, 0.4)
    # take the higher of model score or model + boost, capped at 1.0
    return min(max(model_score, model_score + keyword_boost), 1.0)


def classify_batch(texts, e_clf, s_clf, g_clf):
    e_results = e_clf(texts)
    s_results = s_clf(texts)
    g_results = g_clf(texts)

    labels = []
    scores = []

    for text, e, s, g in zip(texts, e_results, s_results, g_results):
        candidates = {
            "Environmental": get_pillar_score(e, "environmental"),
            "Social":        get_pillar_score(s, "social"),
            "Governance":    get_governance_score(g, text),
        }
        best_label = max(candidates, key=candidates.get)
        best_score = candidates[best_label]

        # only assign label if confidence is above threshold
        if best_score >= MIN_SCORE:
            labels.append(best_label)
            scores.append(round(best_score, 4))
        else:
            labels.append(None)
            scores.append(None)

    return labels, scores


def process():
    # load segments
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} segments\n")

    texts = df["text"].tolist()
    all_labels = []
    all_scores = []

    # load models
    print("Loading ESGBERT models...")
    e_clf, s_clf, g_clf = load_classifiers()
    print("Models loaded\n")

    # classify in batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Classifying"):
        batch = texts[i : i + BATCH_SIZE]
        labels, scores = classify_batch(batch, e_clf, s_clf, g_clf)
        all_labels.extend(labels)
        all_scores.extend(scores)

    # fill in the placeholder columns
    df["esg_label"] = all_labels
    df["esg_score"] = all_scores

    # save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    # summary
    print(f"\nTotal segments: {len(df):,}")
    print(f"Labelled:       {df['esg_label'].notna().sum():,}")
    print(f"Unlabelled:     {df['esg_label'].isna().sum():,}")
    print("\nLabel distribution:")
    print(df["esg_label"].value_counts())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()