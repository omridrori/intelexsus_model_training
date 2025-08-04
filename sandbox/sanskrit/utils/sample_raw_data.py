import json
import random
from pathlib import Path
from tqdm import tqdm

# --- הגדרות ---

# 1. שנה את הנתיב הזה אם תיקיית הנתונים הגולמית שלך נמצאת במקום אחר
#    הנתיב הנוכחי מניח שהפרויקט שלך מאורגן עם תיקיית 'data' ברמה העליונה
RAW_DATA_PATH = Path("data/sanskrit")

# 2. לאן לשמור את הדגימות
OUTPUT_FILE = Path("sanskrit_samples.txt")

# 3. כמה דגימות אקראיות להוציא
NUM_SAMPLES = 300

# --- לוגיקה ראשית ---

def create_samples():
    """
    סורק את תיקיית הנתונים הגולמית, דוגם מסמכים באופן אקראי ושומר אותם לקובץ.
    """
    print("--- Starting Raw Data Sampler ---")
    if not RAW_DATA_PATH.exists() or not RAW_DATA_PATH.is_dir():
        print(f"❌ ERROR: Raw data directory not found at '{RAW_DATA_PATH}'")
        return

    # rglob('**/*.jsonl') ימצא את כל הקבצים בכל תת-התיקיות
    all_files = list(RAW_DATA_PATH.rglob("*.jsonl"))
    if not all_files:
        print(f"❌ ERROR: No .jsonl files were found in '{RAW_DATA_PATH}'")
        return
    
    print(f"Found {len(all_files)} source files. Reading all documents into memory...")

    all_documents = []
    # tqdm יוסיף סרגל התקדמות בזמן קריאת הקבצים
    for file_path in tqdm(all_files, desc="Reading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # טוען כל שורה כ-JSON ומחלץ את השדה 'text'
                    text = json.loads(line).get("text", "")
                    if text.strip(): # מוסיף רק אם הטקסט אינו ריק
                        all_documents.append(text)
                except json.JSONDecodeError:
                    # מתעלם משורות שאינן בפורמט JSON תקין
                    continue

    if not all_documents:
        print("Could not find any documents to sample.")
        return

    print(f"\nFound a total of {len(all_documents):,} documents.")
    
    # ודא שלא ננסה לדגום יותר מסמכים ממה שיש
    actual_samples_to_take = min(NUM_SAMPLES, len(all_documents))
    
    print(f"Randomly selecting {actual_samples_to_take} documents...")
    # דגימה אקראית מהרשימה
    sampled_docs = random.sample(all_documents, actual_samples_to_take)
    
    # כתיבת הדגימות לקובץ פלט
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, doc in enumerate(sampled_docs, 1):
            f_out.write(f"--- SAMPLE #{i:02d} ---\n")
            f_out.write(doc)
            f_out.write(f"\n\n{'='*70}\n\n")

    print(f"\n✅ Success! {actual_samples_to_take} samples have been written to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    create_samples()
