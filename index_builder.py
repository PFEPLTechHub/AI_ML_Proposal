# -*- coding: utf-8 -*-
import os
import faiss
import pickle
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INPUT_DIR = "input_stories"
OUTPUT_INDEX = "fairy_index.faiss"
OUTPUT_META = "fairy_meta.pkl"

# File mapping (title → filename)
FILES = [
    ("The Arabian Nights", "The_Arabian_Nights.pdf"),
    ("Gullivers Travels", "Gullivers_Travels.pdf"),
    ("Alice In Wonderland", "Alice_In_Wonderland.pdf"),
]

# ---------------- LOAD & SPLIT ----------------
texts, titles = [], []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for title, filename in FILES:
    pdf_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        continue

    print(f"📖 Reading: {title}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split(splitter)

    for p in pages:
        texts.append(p.page_content)
        titles.append(title)

print(f"✅ Total chunks extracted: {len(texts)}")

# ---------------- EMBEDDINGS ----------------
print("⚙️ Generating embeddings... (this may take a minute)")
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

# ---------------- BUILD FAISS INDEX ----------------
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# ---------------- SAVE TO DISK ----------------
faiss.write_index(index, OUTPUT_INDEX)
pickle.dump({"texts": texts, "titles": titles}, open(OUTPUT_META, "wb"))

print("\n🎉 Index built successfully!")
print(f"🗂  Saved index to: {OUTPUT_INDEX}")
print(f"🗂  Saved metadata to: {OUTPUT_META}")
