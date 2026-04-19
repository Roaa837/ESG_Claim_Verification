import json
from pathlib import Path

JSON_DIR = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\json"

json_paths = list(Path(JSON_DIR).rglob("*.json"))
print(f"Found {len(json_paths)} JSON files\n")

# check metadata across all files
print("--- Metadata check ---")
for json_path in json_paths:
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    print(
        f"{doc['filename'][:60]:<60} | "
        f"company: {doc['company_name']:<25} | "
        f"year: {doc['year']} | "
        f"pages: {doc['pages_extracted']}/{doc['total_pages']} | "
        f"ocr: {doc['pages_ocr']}"
    )

# inspect one document in detail
print("\n--- Detailed check (first file) ---")
with open(json_paths[0], encoding="utf-8") as f:
    doc = json.load(f)

print("Company:", doc["company_name"])
print("Year:", doc["year"])
print("Report type:", doc["report_type"])

# check page 5
page = doc["pages"][5]
print(f"\nPage {page['page_number']} preview:")
print(page["text"][:300])
print("\nTables on this page:")
for row in page["tables"][:5]:
    print(" -", row)