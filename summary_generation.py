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
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
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

# ===== Filter Top N Reviews with Rating > 1 =====
def get_movie_reviews(reviews):
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
        if rating > 1:
            filtered.append(text.strip())
    return filtered

# ===== Build Prompt for Positive Summary =====
def build_prompt(title, year, reviews):
    review_snippets = "\n".join([f"{i}. {r[:250]}" for i, r in enumerate(reviews, 1)])
    return (
        f"user\n"
        f"Title: {title} ({year})\n\n"
        f"Reviews:\n{review_snippets}\n\n"
        "Summary: Write a detailed positive summary of this movie based only on the above reviews. "
        "Highlight strengths like story, acting, visuals, direction, emotional impact, and reviewer opinions. "
        "Only include details explicitly mentioned in the reviews. "
        "Ensure the summary is **at least 100 words long**. Be thorough and descriptive."
        f"\nassistant\n"
    )

# ===== Parse Summary and Ensure Length =====
def parse_summary(response, min_words=150):
    summary = response.strip()
    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(r"(?<!\.)\.\.(?!\.)", ".", summary)

    # Ensure ends with punctuation
    if summary and not re.search(r"[.!?]$", summary):
        summary += "."  

    # Count words
    word_count = len(summary.split())
    print(f"📝 Generated summary has {word_count} words")

    return summary, word_count

# ===== Summarize with Retry Until Min Word Count =====
def summarize(model, tokenizer, prompt, device, min_words=150, max_attempts=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for attempt in range(max_attempts):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=600 if attempt == 0 else 800,  # Increase aggressively
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
            generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            raw_summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            summary, word_count = parse_summary(raw_summary, min_words=min_words)

            if word_count >= min_words:
                return summary

            print(f"🔄 Attempt {attempt + 1}: Summary too short. Retrying with more tokens...")

        except Exception as e:
            print(f"⚠️ Error during generation (attempt {attempt + 1}): {e}")
            continue

    print("⚠️ Could not reach desired word count.")
    return raw_summary if raw_summary else "Incomplete summary due to generation issues."

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

# ===== Pipeline to Generate Summaries =====
def run_pipeline(
    json_path,
    output_file="positive_summaries.json",
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
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True  # Allow offloading
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()

    model.resize_token_embeddings(len(tokenizer))

    for idx, imdb_id in tqdm(enumerate(imdb_ids), total=len(imdb_ids), desc="🔍 Summarizing"):
        reviews = data.get(imdb_id, [])
        selected_reviews = get_movie_reviews(reviews)
        if not selected_reviews:
            continue

        title = reviews[0].get('review_title', imdb_id)[:50] if reviews else imdb_id
        year = "Unknown"

        prompt = build_prompt(title, year, selected_reviews)

        try:
            summary = summarize(model, tokenizer, prompt, DEVICE, min_words=150)
        except Exception as e:
            print(f"❌ Error on {imdb_id}: {e}")
            continue

        result = {
            "imdb_id": imdb_id,
            "positive_summary": summary
        }

        append_to_json_file(output_file, result)
        print(f"💾 Saved summary for {imdb_id}")

        if idx % CHUNK_SIZE == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"✅ Final summaries saved to {output_file}")


# ===== Entry Point =====
if __name__ == "__main__":
    run_pipeline(
        json_path="grouped_reviews.json",
        output_file="qwen_positive_summaries.json",
        max_movies=6000
    )