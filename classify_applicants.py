from openai import OpenAI
import pandas as pd
import time
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Load CSV
df = pd.read_csv("applicants.csv")

# Ensure columns exist
if 'classification' not in df.columns:
    df['classification'] = ""
else:
    df['classification'] = df['classification'].fillna("")

if 'description' not in df.columns:
    df['description'] = ""
else:
    df['description'] = df['description'].fillna("")

# Error tracking
error_rows = []

# Build prompt
def build_prompt(name):
    return f"""
You are an expert business research assistant. Classify the given entity and provide a one-line description only if it's a startup or a company.

Classification rules:
- If it's an individual person (a founder, applicant, or individual), classify as "Individual".
- If it's a university or research institution/group, classify as "University".
- If it has raised venture capital (VC) and is not public, classify as "Startup".
- If it's publicly listed, classify as "Company".
- If it's a family-owned business, classify as "Company (FM)".
- If no reliable data is found, classify as "NA".

Respond strictly in this format:
Classification: <Startup/Company/Company (FM)/University/Individual/NA>
Description: <One-line ONLY if Startup or Company/Company (FM); otherwise leave blank>

Entity: {name}
"""

# Valid classification options
valid_tags = {"Startup", "Company", "Company (FM)", "University", "Individual", "NA"}

# Classification + Description function
def classify_name(name):
    try:
        prompt = build_prompt(name)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert business research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        lines = content.split('\n')
        classification = ""
        description = ""

        for line in lines:
            if line.lower().startswith("classification:"):
                classification = line.split(":", 1)[1].strip().rstrip('.').title()
            elif line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip()

        if classification not in valid_tags:
            classification = "NA"
            description = ""

        # Only keep description for startups and companies
        if classification not in {"Startup", "Company", "Company (FM)"}:
            description = ""

        return classification, description

    except Exception as e:
        print(f"Error with {name}: {e}")
        return "ERROR", ""

# Main loop (first 500 rows)
for i, row in tqdm(df.iterrows(), total=min(len(df), 500)):
    if df.at[i, 'classification'] == "":
        tag, desc = classify_name(row['name'])
        df.at[i, 'classification'] = tag
        df.at[i, 'description'] = desc
        if tag == "ERROR":
            error_rows.append({'row_index': i, 'name': row['name']})
        time.sleep(1.2)

# Save results
df.to_csv("classified_applicants.csv", index=False)

# Save error log
if error_rows:
    pd.DataFrame(error_rows).to_csv("error_log.csv", index=False)





