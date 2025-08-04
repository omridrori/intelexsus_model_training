import json
import re
from pathlib import Path
from tqdm import tqdm

# --- 1. הגדרות דיבוג ---
# הנתיב לקבצי הדאטה הגולמיים שלך
RAW_DATA_PATH = Path(r"C:\Users\omrid\Desktop\university\intelexsus\data\sanskrit")
# המונחים שאנחנו רוצים לחקור. למה הם הופכים לשורה נפרדת?
TERMS_TO_DEBUG = ['153', '38', '27', '8', '42']
# כמה דוגמאות למצוא לפני שהסקריפט עוצר
MAX_FINDS = 5

def debug_verse_segmentation():
    """
    סקריפט זה סורק את קבצי המקור, מאתר מסמכים המכילים מונחים בעייתיים,
    ומדגים כיצד תהליך הפילוח לפסוקים משפיע עליהם.
    """
    print("--- Starting Verse Segmentation Debugger ---")
    
    all_files = list(RAW_DATA_PATH.glob("**/*.jsonl"))
    if not all_files:
        print(f"Error: No .jsonl files found in '{RAW_DATA_PATH}'.")
        return

    found_count = 0

    for file_path in all_files:
        if found_count >= MAX_FINDS:
            break
            
        print(f"\nScanning file: {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if found_count >= MAX_FINDS:
                    break
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if not text:
                        continue
                    
                    # בדוק אם אחד המונחים הבעייתיים מופיע בטקסט כמילה שלמה
                    for term in TERMS_TO_DEBUG:
                        # שימוש ב-regex \b (word boundary) כדי למצוא את המילה המדויקת
                        if re.search(r'\b' + re.escape(term) + r'\b', text):
                            found_count += 1
                            print(f"\n--- [MATCH #{found_count}] Found term '{term}' ---")
                            
                            # 1. הדפס את המסמך המקורי המלא
                            print(f"[Original Document]: {text.strip()}")
                            
                            # 2. בצע את תהליך הפילוח בדיוק כפי שעשית בסקריפט הראשי
                            unified_text = text.replace('//', '||')
                            verses = unified_text.split('||')
                            
                            # 3. הדפס את התוצאות
                            print("\n[Segmentation Result]:")
                            for i, verse in enumerate(verses):
                                verse = verse.strip()
                                # מדגיש את הפסוק הבעייתי שגרם לבעיה
                                if verse == term:
                                    print(f"  Verse {i+1}: '{verse}'   <--- ⚠️ PROBLEM VERSE")
                                else:
                                    print(f"  Verse {i+1}: '{verse}'")
                            
                            print("-" * 50)
                            # מצאנו התאמה, עבור לשורה הבאה בקובץ המקורי
                            break 

                except (json.JSONDecodeError, AttributeError):
                    continue

    if found_count == 0:
        print("\nCould not find any of the debug terms in the corpus.")
    else:
        print(f"\n--- Debugging finished. Found {found_count} examples. ---")


if __name__ == "__main__":
    debug_verse_segmentation()
