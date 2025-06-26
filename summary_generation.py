import json
import torch
import gc
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import pandas as pd

# Load JSON data
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Pick up to `max_reviews` reviews per movie
def get_movie_reviews(reviews, max_reviews=5):
    return [rev['review_text'] for rev in reviews[:max_reviews] if 'review_text' in rev]

# Build prompt for model
def build_prompt(movie_title, reviews):
    prompt = f"""You are given {len(reviews)} reviews about the movie "{movie_title}". Read them carefully and generate two summaries:
1. A positive summary highlighting what people liked about the movie.
2. A negative summary outlining the main criticisms or dislikes.

Reviews:
"""
    for i, r in enumerate(reviews, start=1):
        prompt += f"{i}. {r}\n"

    prompt += """
Positive Summary:
"""
    return prompt

# Parse output into positive and negative summaries
def parse_output(output):
    try:
        pos_start = output.find("Positive Summary:") + len("Positive Summary:")
        neg_start = output.find("Negative Summary:")

        positive_summary = output[pos_start:neg_start].strip()
        negative_summary = output[neg_start + len("Negative Summary:"):].strip()

        if "\n\n" in negative_summary:
            negative_summary = negative_summary.split("\n\n")[0]

        return {
            "positive_summary": positive_summary,
            "negative_summary": negative_summary
        }
    except Exception as e:
        print("Error parsing output:", str(e))
        return {"positive_summary": "", "negative_summary": ""}

# Generate summaries using Llama 2.0 3B model
def summarize_with_llama(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return parse_output(response)

# Main pipeline function
def run_pipeline(
    json_path,
    output_file="movie_summaries.json",
    MAX_MOVIES=10,               # Number of movies to process
    START_OFFSET=0,              # Start from this index
    MAX_REVIEWS_PER_MOVIE=5,     # Limit to N reviews per movie
    MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
):
    data = load_data(json_path)

    imdb_ids = list(data.keys())[START_OFFSET:]  # Skip already processed movies
    total_movies = min(MAX_MOVIES, len(imdb_ids))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME).to(device)

    results = []

    for idx, imdb_id in tqdm(enumerate(imdb_ids), desc="Processing Movies", total=total_movies):
        if idx >= total_movies:
            break

        reviews = data[imdb_id]
        selected_reviews = get_movie_reviews(reviews, max_reviews=MAX_REVIEWS_PER_MOVIE)

        if not selected_reviews:
            print(f"⚠️ No valid reviews found for movie {imdb_id}")
            continue

        # Fallback for movie title
        movie_title = imdb_id
        if 'review_title' in reviews[0]:
            movie_title = reviews[0]['review_title'][:50]

        prompt = build_prompt(movie_title, selected_reviews)
        summaries = summarize_with_llama(model, tokenizer, prompt, device)

        results.append({
            "imdb_id": imdb_id,
            "title": movie_title,
            "positive_summary": summaries["positive_summary"],
            "negative_summary": summaries["negative_summary"]
        })

        # Clean memory
        del reviews, selected_reviews, prompt, summaries
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file.replace(".json", ".csv"), index=False)

    print(f"✅ Summaries saved to {output_file} and CSV version")

if __name__ == "__main__":
    run_pipeline(
        json_path="/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation copy/pipeline/Data Prepration pipelines/output/json/grouped_reviews.json",
        output_file="summaries_part1.json",
        MAX_MOVIES=2,               # Change as needed
        START_OFFSET=0,
        MAX_REVIEWS_PER_MOVIE=10
    )
