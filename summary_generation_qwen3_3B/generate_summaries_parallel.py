import pandas as pd
import ast
import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cached_dir = "/hf_models/"
pref = "user_pref_master.csv"

# Load LLMs
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
tokenizer1 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cached_dir)
model1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True, cache_dir=cached_dir)

tokenizer2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cached_dir)
model2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True, cache_dir=cached_dir)


def compute_cosine_similarity(summary, next_likes_dict):
    try:
        next_likes_text = " ".join(next_likes_dict.values())
    except Exception:
        return 0.0
    embeddings = embedder.encode([summary, next_likes_text], batch_size=1, show_progress_bar=False)
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(sim[0][0])


def query_model(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1500,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content


def generate_best_summary(row, model, tokenizer, local_df):
    likes_dict = row["_likes_dict"]
    prev_summary = row['prev_summary']
    has_next_movie = row['has_next_movie']
    prompt = (
        f"Based on the following liked features:\n\n{likes_dict}"
        + (f" and {prev_summary}" if prev_summary else "") +
        "\n\nGenerate a refined, updated paragraph summarizing what this user generally prefers in movies.\n"
        "Ensure that this response is generated independently and does not rely on or get influenced by any previous summaries or responses."
        "Treat this as a standalone task with no prior context. Limit the summary to a max of 50 words"
    )
    candidates = []
    for _ in range(5):
        output = query_model(prompt, model, tokenizer)
        summary = output.split("features:")[-1].strip()
        if has_next_movie:
            try:
                next_likes_dict = local_df.at[row.name + 1, "_likes_dict"]
                sim = compute_cosine_similarity(summary, next_likes_dict)
            except:
                sim = 0.0
        else:
            sim = 1.0
        candidates.append((summary, sim))
    best = max(candidates, key=lambda x: x[1])
    return best[0]


def process_user_rows(local_df, model, tokenizer):
    # Sort and prepare
    local_df = local_df.sort_values(by=['user_id', 'cluster_id', 'movie_id']).reset_index(drop=True)

    # Pre-parse likes_dict to avoid repeated parsing
    local_df["_likes_dict"] = local_df["likes"].apply(ast.literal_eval)

    # Initialize summary columns
    local_df['prev_summary'] = ""
    local_df['best_summary'] = ""
    summary_memory = {}
    last_cluster = None
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(local_df.iterrows(), total=len(local_df), desc="Thread")
    for idx, row in pbar:
        key = (row['user_id'], row['cluster_id'])
        prev = summary_memory.get(key, "")
        local_df.at[idx, 'prev_summary'] = prev

        logging.info(f"Processing user={row['user_id']} cluster={row['cluster_id']} movie={row['movie_id']}")
        try:
            best = generate_best_summary(row, model, tokenizer, local_df)
        except Exception as e:
            logging.error(f"Summary generation failed for user {key[0]}, cluster {key[1]}, movie {row['movie_id']}: {str(e)}")
            best = "[Summary unavailable due to error]"
        local_df.at[idx, 'best_summary'] = best
        summary_memory[key] = best

        # Save when switching to a new cluster
        if last_cluster is not None and last_cluster != key:
            cluster_df = local_df[(local_df['user_id'] == last_cluster[0]) & (local_df['cluster_id'] == last_cluster[1])]
            cluster_path = os.path.join(output_dir, f"user_{last_cluster[0]}_cluster_{last_cluster[1]}.csv")
            cluster_df.to_csv(cluster_path, index=False, float_format='%.4f')
            logging.info(f"Saved checkpoint: {cluster_path}")
        last_cluster = key

    # Final save for last cluster
    if last_cluster:
        cluster_df = local_df[(local_df['user_id'] == last_cluster[0]) & (local_df['cluster_id'] == last_cluster[1])]
        cluster_path = os.path.join(output_dir, f"user_{last_cluster[0]}_cluster_{last_cluster[1]}.csv")
        cluster_df.to_csv(cluster_path, index=False, float_format='%.4f')
        logging.info(f"Saved checkpoint: {cluster_path}")

    return local_df


# Load and reverse the input data
df = pd.read_csv(pref)
df = df[::-1].reset_index(drop=True)  # <-- Process from last to first row

# Precompute has_next_movie
user_groups = df.groupby('user_id', sort=False)
has_next_movie = []

for _, group in user_groups:
    indices = group.index.tolist()
    has_next = [False] * len(indices)
    for i in range(len(indices) - 1):
        has_next[i] = True
    has_next_movie.extend(has_next)

df['has_next_movie'] = has_next_movie

# Final sort and initialize summary columns
df = df.sort_values(by=["user_id", "cluster_id", "movie_id"]).reset_index(drop=True)
df['prev_summary'] = ""
df['best_summary'] = ""

# Split users for parallel processing
user_ids = df['user_id'].unique()
users1 = set(user_ids[::2])
users2 = set(user_ids[1::2])

df1 = df[df['user_id'].isin(users1)].copy()
df2 = df[df['user_id'].isin(users2)].copy()

# Run in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(process_user_rows, df1, model1, tokenizer1)
    future2 = executor.submit(process_user_rows, df2, model2, tokenizer2)
    local_df1 = future1.result()
    local_df2 = future2.result()

# Merge results
files = glob.glob("checkpoints/user_*_cluster_*.csv")
all_clusters = []
for f in files:
    cluster_df = pd.read_csv(f)
    all_clusters.append(cluster_df)

final_df = pd.concat(all_clusters, axis=0)
final_df = final_df.sort_values(by=["user_id", "cluster_id", "movie_id"]).reset_index(drop=True)
final_df[['user_id', 'cluster_id', 'movie_id', 'best_summary']].to_csv("user_summaries_with_context.csv", index=False)