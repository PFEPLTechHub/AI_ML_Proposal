# -*- coding: utf-8 -*-
"""
🌙 Fairy Nights 🏰 v8.4 (FINAL FIX: Retrieval Threshold and Generation Logic)
Funny storyteller Guard that narrates real stories from:
- Alice in Wonderland
- Arabian Nights
- Gulliver’s Travels

✨ Features:
✅ Retrieves passages using SentenceTransformer + FAISS (Word count check loosened)
✅ Summarizes & rewrites into funny storytelling (Two-part prompt for accuracy)
✅ Generates image (Stable Diffusion via HF API)
✅ Speaks the reply (gTTS)
✅ Clickable sample questions
"""

import os, pickle, faiss, numpy as np, requests, gradio as gr
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from gtts import gTTS
import random

# ---------------- CONFIG ----------------
VALID_STORIES = ["Alice in Wonderland", "Arabian Nights", "Gulliver's Travels"]
THRESHOLD = 0.5
EMBED_MODEL = "all-MiniLM-L6-v2"
SUMMARY_MODEL = "facebook/bart-large-cnn"
REWRITE_MODEL = "google/flan-t5-base"

# 🔑 Hugging Face API (for image generation)


# ---------------- LOAD MODELS & DATA ----------------
print("⚙️ Loading models and data...")
# NOTE: Ensure you have 'fairy_index.faiss' and 'fairy_meta.pkl' in your directory
try:
    model_embed = SentenceTransformer(EMBED_MODEL)

    # Load FAISS + metadata
    index = faiss.read_index("fairy_index.faiss")
    meta = pickle.load(open("fairy_meta.pkl", "rb"))
    texts, titles = meta["texts"], meta["titles"]

    # Load embeddings
    if "embeddings" in meta and meta["embeddings"] is not None:
        embeddings = np.asarray(meta["embeddings"], dtype=np.float32)
        print(f"✅ Loaded {len(embeddings)} precomputed embeddings.")
    else:
        print("⚠️ Computing embeddings (one-time)...")
        embeddings = model_embed.encode(texts, show_progress_bar=True)
        meta["embeddings"] = embeddings
        pickle.dump(meta, open("fairy_meta.pkl", "wb"))
        print("💾 Saved embeddings to fairy_meta.pkl")

    # Load summarizer & rewriter
    print("📖 Loading summarizer and rewriter models...")
    summarizer = pipeline("summarization", model=SUMMARY_MODEL)
    rewriter = pipeline("text2text-generation", model=REWRITE_MODEL)
    print("✅ Models ready!\n")
except Exception as e:
    print(f"FATAL ERROR: Could not load RAG components. Ensure models and files are present: {e}")
    exit()

# ---------------- IMPROVED STORY RETRIEVAL (Tuned merge_chunks) ----------------
def find_story(query, threshold=THRESHOLD, merge_chunks=4): # Reduced from 6 to 4 for better focus
    """
    Smarter retriever:
    - Expands query semantically
    - Ignores metadata chunks (like 'This eBook')
    - Dynamically adjusts threshold if needed
    """
    expansions = " story adventure journey travel voyage sea ship ocean tale magic fantasy dream"
    q_aug = query + " " + expansions

    q_vec = model_embed.encode([q_aug])[0]
    sims = np.dot(embeddings, q_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    )

    idx = int(np.argmax(sims))
    score = float(np.max(sims))
    story_title = titles[idx]
    print(f"🔍 Similarity {score:.3f} → {story_title}")

    # Adaptive threshold — looser if the top result clearly matches a known story
    if score < threshold and any(v.lower() in story_title.lower() for v in VALID_STORIES):
        print("⚠️ Slightly weak match, but correct story detected.")
    elif score < (threshold - 0.05):
        print("⚠️ Weak match — using fallback context.")
        return story_title, ""

    # Merge text chunks for more context
    start = max(0, idx - 1) # Start from 1 chunk back
    end = min(len(texts), idx + merge_chunks)
    merged_text = " ".join(texts[start:end])

    # Clean metadata noise before summarization
    bad_phrases = [
        "project gutenberg", "ebook", "chapter", "illustration", "copyright",
        "table of contents", "this eBook", "end of the project", "license"
    ]
    for bad in bad_phrases:
        merged_text = merged_text.replace(bad, "")
    merged_text = " ".join(merged_text.split())

    # Ensure minimum usable text
    if len(merged_text.split()) < 40:
        print("⚠️ Retrieved passage too short, trying nearby chunk.")
        next_idx = min(idx + 1, len(texts) - 1)
        merged_text += " " + texts[next_idx]

    return story_title, merged_text

# ---------------- STORY GENERATION (FIXED FOR ACCURACY) ----------------
def generate_guard_reply(query, story, passage):
    """Generate a proper funny storytelling narration, constrained to facts."""
    try:
        # Step 1: Summarize (Tuned parameters from previous fix are good)
        token_count = len(passage.split())
        if token_count < 80:
            summary = passage
        else:
            summary = summarizer(
                passage,
                max_length=200,
                min_length=min(80, token_count // 2),
                do_sample=False,
            )[0]["summary_text"]

        # Step 2: STRONGER TWO-PART REWRITE PROMPT (CRITICAL FIX)
        rewrite_prompt = (
            f"Task 1: **FACTUAL SUMMARY.** Analyze the text below and state in a single sentence "
            f"the core event that answers the query: '{query}'.\n\n"
            f"Task 2: **STORYTELLING REWRITE.** You are a friendly, exaggerating medieval guard. "
            f"Using the sentence from Task 1 as your ONLY factual basis, rewrite it into a vivid, "
            f"funny, self-contained story (150-250 words) about '{story}'. "
            f"Use modern English, add humor, and make it sound like you witnessed it. "
            f"DO NOT mention politics, elections, or repetition. Start with 'Well, I remember that day...'.\n\n"
            f"Source Text:\n{summary}\n\n"
            f"**BEGIN OUTPUT:**\nTask 1 Output: "
        )

        # Step 3: Try larger T5 or BlenderBot for creative output (Keep logic as-is)
        try:
            creative_gen = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                truncation=True,
            )
        except Exception:
            creative_gen = rewriter

        # ADJUSTED GENERATION PARAMETERS (Lowered max_new_tokens to prevent repetition)
        rewritten = creative_gen(
            rewrite_prompt,
            max_new_tokens=200,      # Reduced from 250/400 to prevent infinite loop/repetition
            do_sample=True,
            temperature=0.7,         # Keeps tone, avoids severe hallucination
            top_p=0.8                
        )[0]["generated_text"]

        # Step 4: Clean unwanted text
        # ... (cleaning logic remains here) ...
        for junk in ["Task 1 Output:", "Task 2 Output:", "ebook", "chapter", "project gutenberg", "free e-book", "copyright", "Alice's Adventures in Wonderland by Lewis Carroll"]:
            rewritten = rewritten.replace(junk, "").replace(junk.lower(), "")
        rewritten = " ".join(rewritten.split())

        # Step 5: Add humor/tone
        openings = [
            f"Ah, {story}! Gather 'round, traveler —",
            f"Heh! {story} always gets me laughing —",
            f"Listen close! The wonders of {story} unfold like this —",
            f"Oh, {story}? You’ll love this part —",
        ]
        endings = [
            "Quite the commotion, eh? Even my armor rattled from laughter!",
            "And that’s how it went — still makes me smile every time!",
            "Ha! I nearly dropped my spear the first time I heard that tale!",
            "And that, my friend, is why I never trust curious rabbits or flying carpets!",
        ]

        reply = f"🛡️ Guard: {random.choice(openings)} {rewritten.strip()} {random.choice(endings)}"

    except Exception as e:
        print("⚠️ Story generation failed:", e)
        reply = f"🛡️ Guard: My scrolls got soaked in tea again! I can’t recall {story}, but it was a hilarious mess!"
    return reply

# ---------------- IMAGE ----------------
def generate_story_image(story, query):
    prompt = f"storybook fantasy illustration of {story}, {query}, colorful, whimsical art, vibrant lighting"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    print("🎨 Requesting image from Hugging Face...")
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}",
            headers=headers, json={"inputs": prompt}, timeout=120,
        )
        if r.status_code != 200:
            print("⚠️ HF Image Error:", r.text)
            return None
        return Image.open(BytesIO(r.content))
    except requests.exceptions.Timeout:
        print("⚠️ HF Image Request Timed Out.")
        return None
    except Exception as e:
        print("⚠️ Image generation failed:", e)
        return None

# ---------------- AUDIO ----------------
def generate_audio(reply_text):
    try:
        tts = gTTS(text=reply_text, lang="en")
        path = "guard_reply.mp3"
        tts.save(path)
        return path
    except Exception as e:
        print("⚠️ Audio generation failed:", e)
        return None

# ---------------- CHAT LOGIC (FIXED) ----------------
def guard_chat(query, history):
    # Retrieve story passage
    story, passage = find_story(query)
    passage_length = len(passage.split()) # Get the final passage length

    # 🔥 FIX: Loosened the required passage length to 25 words (from 40)
    # This prevents the fallback when the similarity score is high but cleaning resulted in few words.
    if story is None or not any(v.lower() in story.lower() for v in VALID_STORIES) or passage_length < 25:
        
        print(f"DEBUG: Triggered Fallback. Passage length: {passage_length}. Story: {story}.")
        
        funny_reply = random.choice([
            "🛡️ Guard: I don’t know that tale! Maybe it’s from another kingdom. Stick to Alice, Gulliver, or Arabian Nights, traveler!",
            "🛡️ Guard: Hmm, not in my fairy scrolls. I only guard Alice, Arabian Nights, and Gulliver!",
            "🛡️ Guard: Oh dear! That story’s lost beyond the castle walls. Try one I know!",
        ])
        audio_path = generate_audio(funny_reply)
        return funny_reply, None, audio_path

    # Generate reply, image, and audio
    reply = generate_guard_reply(query, story, passage)
    img = generate_story_image(story, query)
    audio_path = generate_audio(reply)
    return reply, img, audio_path

# ---------------- GRADIO UI ----------------
with gr.Blocks(title="🌙 Fairy Nights v8.4") as demo:
    gr.Markdown("## 🏰 **Fairy Nights — Chat with the Storytelling Guard!** (Fixes Applied)")

    chatbot = gr.Chatbot(label="Fairy Guard", height=480)
    msg = gr.Textbox(label="Ask the Guard", placeholder="Ask about Alice, Arabian Nights, or Gulliver...")
    audio_box = gr.Audio(label="Guard’s Voice", interactive=False)
    download_btn = gr.File(label="Download Voice (MP3)")
    img_box = gr.Image(label="Story Illustration")

    # 🔹 Sample clickable questions
    sample_questions = [
        "Who is the White Rabbit?",
        "Where did Alice go?",
        "What happens when Gulliver meets the tiny people?",
        "How did Alice get lost?",
        "What adventure did Gulliver face at sea?",
    ]

    gr.Markdown("### 🎯 **Sample Questions**")
    with gr.Row():
        for q in sample_questions[:3]:
            gr.Button(q).click(fn=lambda x=q: x, outputs=msg)
    with gr.Row():
        for q in sample_questions[3:]:
            gr.Button(q).click(fn=lambda x=q: x, outputs=msg)

    def interact(user_msg, chat_history):
        reply, img, audio = guard_chat(user_msg, chat_history)
        chat_history.append((user_msg, reply))
        
        img_output = None
        if img:
            img_path = "story.png"
            img.save(img_path)
            img_output = img_path # Pass the path for Gradio Image component
            
        return "", chat_history, audio, audio, img_output

    msg.submit(interact, [msg, chatbot], [msg, chatbot, audio_box, download_btn, img_box])

if __name__ == "__main__":
    demo.launch()