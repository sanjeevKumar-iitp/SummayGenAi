import os
import json
import re
import gc
import torch
import time
import multiprocessing as mp
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from huggingface_hub import login

# CONFIG
HF_TOKEN = ""  # Optional: insert your token if needed
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
INPUT_FILE = "outputs/qwen_negative_summaries.json"  # <- input with negative_summary fields
OUTPUT_FILE = "outputs/final_negative_conclusions.json"
CHUNK_SIZE = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache_dir"

if HF_TOKEN:
    login(HF_TOKEN)

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(title, summary):
    return (
        f"user\n"
        f"Movie Title: {title}\n\n"
        f"Negative Feedback:\n{summary}\n\n"
        "Summary: Write a concise, engaging conclusion summarizing only the **negative aspects** of this movie. "
        "Focus on criticism related to acting, story, direction, pacing, plot holes, visuals, or emotional impact. "
        "Exclude any positive or neutral points. Keep it under 100 words. Sound analytical and critical, not rude.\n"
        f"\nassistant\n"
    )

def parse_summary(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text if text.endswith((".", "!", "?")) else text + "."

def append_to_json_file(path, batch):
    if os.path.exists(path):
        with open(path, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.extend(batch)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2)

def run_worker(gpu_id, chunk, output_path):
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()

    model.resize_token_embeddings(len(tokenizer))

    results = []

    for entry in tqdm(chunk, desc=f"[GPU {gpu_id}] Generating"):
        imdb_id = entry.get("imdb_id")
        neg_summary = entry.get("negative_summary", "").strip()
        if not neg_summary:
            continue

        prompt = build_prompt(imdb_id, neg_summary)
        inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{gpu_id}")
        gen_config = GenerationConfig(
            max_new_tokens=120,
            min_new_tokens=80,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=gen_config
                )
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"imdb_id": imdb_id, "conclusion": parse_summary(generated)})
        except Exception as e:
            print(f"[GPU {gpu_id}] ❌ Error: {e}")
            continue

        if len(results) % CHUNK_SIZE == 0:
            append_to_json_file(output_path, results)
            results.clear()
            torch.cuda.empty_cache()
            gc.collect()

    if results:
        append_to_json_file(output_path, results)

def parallel_negative_conclusion_pipeline(max_movies=6000):
    data = load_data(INPUT_FILE)
    if max_movies:
        data = data[:max_movies]

    num_gpus = torch.cuda.device_count()
    print(f"🚀 Starting NEGATIVE conclusion generation on {num_gpus} GPUs")

    chunks = [data[i::num_gpus] for i in range(num_gpus)]

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(target=run_worker, args=(gpu_id, chunk, OUTPUT_FILE))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    start = time.time()
    parallel_negative_conclusion_pipeline()
    print(f"✅ Negative conclusions generated in {(time.time() - start) / 60:.2f} minutes")
