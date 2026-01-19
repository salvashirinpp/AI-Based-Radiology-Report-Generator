import os
import faiss
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOC_PATH = "rag/documents"              # where PDFs exist
PROCESSED_PATH = "rag/processed"
VECTOR_PATH = "rag/vector_store"

print("📂 DOC_PATH:", os.path.abspath(DOC_PATH))

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(VECTOR_PATH, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

def extract_text_from_pdf(pdf_path):
    print(f"📄 Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        else:
            print(f"⚠️ Page {i} has NO extractable text")

    return text.strip()

# -------- STEP 1: FIND PDFs --------
pdf_files = [f for f in os.listdir(DOC_PATH) if f.lower().endswith(".pdf")]

if not pdf_files:
    print("❌ NO PDF FILES FOUND in rag/")
    exit()

print(f"✅ Found {len(pdf_files)} PDF files")

# -------- STEP 2: EXTRACT TEXT --------
for file in pdf_files:
    pdf_path = os.path.join(DOC_PATH, file)
    txt_name = file.replace(".pdf", ".txt")
    txt_path = os.path.join(PROCESSED_PATH, txt_name)

    text = extract_text_from_pdf(pdf_path)

    if len(text) < 50:
        print(f"❌ WARNING: Very little text extracted from {file}")
        continue

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Saved extracted text → {txt_path}")

# -------- STEP 3: LOAD TXT FILES --------
txt_files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith(".txt")]

if not txt_files:
    print("❌ NO TEXT FILES FOUND in rag/processed/")
    exit()

print(f"✅ Found {len(txt_files)} processed text files")

for file in txt_files:
    with open(os.path.join(PROCESSED_PATH, file), "r", encoding="utf-8") as f:
        content = f.read()

    documents.append(content)
    metadata.append({"source": file.replace(".txt", "")})

# -------- STEP 4: BUILD FAISS --------
print("🧠 Creating embeddings...")
embeddings = model.encode(documents, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{VECTOR_PATH}/faiss.index")

with open(f"{VECTOR_PATH}/metadata.pkl", "wb") as f:
    pickle.dump((documents, metadata), f)

print("🎉 SUCCESS: FAISS index built!")
