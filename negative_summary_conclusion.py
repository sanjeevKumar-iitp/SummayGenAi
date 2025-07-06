import os
import json
import re
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from transformers import GenerationConfig

# ===== CONFIGURATION =====
os.environ["ACCELERATE_USE_FSDP"] = "0"
HF_TOKEN = ""  # ← Add your HuggingFace token here
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
CHUNK_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache_dir'

# ===== HuggingFace Login =====
login(HF_TOKEN)

# ===== Load JSON Data =====
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== Build Prompt Based on Negative Summary =====
def build_prompt(movie_title, negative_summary):
    return (
        f"user\n"
        f"Movie Title: {movie_title}\n\n"
        f"Negative Feedback:\n{negative_summary}\n\n"
        "Summary: Write a concise and professional conclusion summarizing only the negative aspects of this movie. "
        "Highlight issues mentioned in the text such as weak plot, bad acting, poor direction, low production quality, or lack of engagement. "
        "Avoid exaggeration or positive remarks. Keep it under 100 words.\n"
        f"\nassistant\n"
    )

# ===== Clean & Parse LLM Output =====
def parse_summary(response):
    summary = response.strip()
    summary = re.sub(r"\s+", " ", summary)
    if summary and not re.search(r"[.!?]$", summary):
        summary += "."  # Ensure ends with punctuation
    return summary

# ===== Generate Summary Using Qwen Model (Fast Mode) =====
def summarize(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = GenerationConfig(
        max_new_tokens=120,
        min_new_tokens=80,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config
        )
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"⚠️ Error during generation: {e}")
        return ""

# ===== Append Result Safely to JSON File =====
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

# ===== Pipeline to Generate Negative Conclusions =====
def run_conclusion_pipeline(
    input_summary_file="qwen_negative_summaries.json",
    output_file="final_negative_conclusions.json",
    start_offset=0,
    max_movies=None
):
    print("🧠 Loading existing negative summaries...")
    data = load_data(input_summary_file)

    if start_offset > len(data):
        print("⚠️ Start offset exceeds data size.")
        return

    movies = data[start_offset:]
    if max_movies:
        movies = movies[:max_movies]

    print(f"📦 Loading model: {MODEL_NAME} on {DEVICE}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=False
    ).eval()

    model.resize_token_embeddings(len(tokenizer))

    for idx, entry in enumerate(tqdm(movies, desc="🎬 Generating Negative Conclusions")):
        imdb_id = entry.get("imdb_id")
        negative_summary = entry.get("negative_summary", "").strip()

        if not negative_summary:
            print(f"🚫 No negative summary found for {imdb_id}. Skipping...")
            continue

        movie_title = imdb_id  # Replace with actual title if needed

        prompt = build_prompt(movie_title, negative_summary)

        try:
            raw_output = summarize(model, tokenizer, prompt, DEVICE)
            conclusion = parse_summary(raw_output)
        except Exception as e:
            print(f"❌ Error generating conclusion for {imdb_id}: {e}")
            continue

        result = {
            "imdb_id": imdb_id,
            "conclusion": conclusion
        }

        append_to_json_file(output_file, result)
        print(f"💾 Saved negative conclusion for {imdb_id}")

        if idx % CHUNK_SIZE == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"✅ Final negative conclusions saved to {output_file}")


# ===== Entry Point =====
if __name__ == "__main__":
    run_conclusion_pipeline(
        input_summary_file="qwen_negative_summaries.json",
        output_file="final_negative_conclusions.json",
        max_movies=6000
    )
