import json

# Define allowed characters
def is_valid_char(char):
    # Allow basic ASCII letters, numbers, common punctuation, and whitespace
    return char.isalnum() or char in ' .,!?;:\'"()[]{}@#$%^&*+-_=<>~/\\’\n\r\t'

def clean_summary(summary):
    # Scan character-by-character and keep only valid ones
    cleaned = ''.join(c for c in summary if is_valid_char(c))
    
    # Replace multiple spaces with single space
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()

def clean_json_data(data):
    for item in data:
        if "positive_summary" in item:
            item["positive_summary"] = clean_summary(item["positive_summary"])
    return data

# File paths
input_file = "positive_summaries.json"
output_file = "cleaned_movies.json"

# Load JSON
try:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: File '{input_file}' not found.")
    exit()

# Clean summaries
cleaned_data = clean_json_data(data)

# Save cleaned JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

print(f"✅ Cleaning complete. Cleaned JSON saved to '{output_file}'.")