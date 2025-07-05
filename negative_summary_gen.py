import os
import json
import re
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# ===== CONFIGURATION =====
HF_TOKEN = ""
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_REVIEWS_PER_MOVIE = 5
CHUNK_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache_dir'

# ===== HuggingFace Login =====
login(HF_TOKEN)

# ===== Load JSON Data =====
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== Filter Top N Negative Reviews (rating ≤ 5) =====
def get_negative_reviews(reviews):
    filtered = []
    for rev in reviews[:MAX_REVIEWS_PER_MOVIE]:
        rating_str = rev.get("review_rating")
        text = rev.get("review_text")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            rating = int(rating_str)
        except (TypeError, ValueError):
            continue
        if rating <= 5:
            filtered.append(text.strip())
    return filtered

# ===== Build a Prompt for Negative Summary =====
def build_negative_prompt(title, year, reviews):
    review_snippets = "\n".join([f"{i}. {r[:250]}" for i, r in enumerate(reviews, 1)])
    return (
        f"<s>[INST] Title: {title} ({year})\n\n"
        f"Reviews:\n{review_snippets}\n\n"
        "Summary: Write a concise negative summary of this movie based only on the above reviews. "
        "Highlight weaknesses like poor acting, bad pacing, confusing plot, weak direction, or lack of emotional impact. "
        "Only include details explicitly mentioned in the reviews. Do not mention reviewers or audience opinions. Keep under 100 words."
        "[/INST]"
    )

# ===== Parse the Summary Response from LLM Output =====
def parse_summary(response):
    summary = response.strip()
    summary = re.sub(r"\s+", " ", summary)  # Normalize whitespace
    return summary[:500]  # Truncate safely

# ===== Summarize with Token Control =====
def summarize(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )
        # Slice only generated part, excluding prompt
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ Error during generation: {e}")
        return ""

# ===== Append to JSON File Safely =====
def append_to_json_file(file_path, result):
    if os.path.exists(file_path):
        with open(file_path, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(result)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([result], f, indent=2)

# ===== Pipeline for Negative Summary =====
def run_negative_pipeline(
    json_path,
    output_file="negative_summaries.json",
    start_offset=0,
    max_movies=None
):
    data = load_data(json_path)
    imdb_ids = list(data.keys())[start_offset:]
    if max_movies:
        imdb_ids = imdb_ids[:max_movies]

    print(f"🧠 Total movies to process: {len(imdb_ids)}")
    print(f"📦 Loading model: {MODEL_NAME} on {DEVICE}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    ).eval()

    for idx, imdb_id in tqdm(enumerate(imdb_ids), total=len(imdb_ids), desc="🔍 Negative Summarizing"):
        reviews = data.get(imdb_id, [])
        selected_reviews = get_negative_reviews(reviews)
        if not selected_reviews:
            continue

        title = reviews[0].get('review_title', imdb_id)[:50] if reviews else imdb_id
        year = "Unknown"

        prompt = build_negative_prompt(title, year, selected_reviews)

        try:
            raw_output = summarize(model, tokenizer, prompt, DEVICE)
            summary = parse_summary(raw_output)
        except Exception as e:
            print(f"❌ Error on {imdb_id}: {e}")
            continue

        result = {
            "imdb_id": imdb_id,
            "negative_summary": summary
        }

        append_to_json_file(output_file, result)
        print(f"💾 Saved negative summary for {imdb_id}")

        if idx % CHUNK_SIZE == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"✅ Final negative summaries saved to {output_file}")


# ===== Entry Point =====
if __name__ == "__main__":
    run_negative_pipeline(
        json_path="/root/SummayGenAi/grouped_reviews.json",
        output_file="negative_summaries.json",
        max_movies=6000
    )
