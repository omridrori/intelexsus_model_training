import os
from tqdm import tqdm

# --- הגדרות ---

# שנה את הנתיבים האלה אם צריך.
# הקוד מניח שהסקריפט נמצא בתיקיית הפרויקט הראשית או בתיקיית sandbox.
INPUT_FILE_PATH = os.path.join('data', 'sanskrit_preprocessed', 'sanskrit_latin_only.txt')
OUTPUT_FILE_PATH = os.path.join('data', 'sanskrit_preprocessed', 'sanskrit_final_for_bert.txt')

# ודא שהתיקייה לקובץ הפלט קיימת
os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

# --- לוגיקת הסינון ---

total_lines = 0
removed_lines = 0

print(f"Reading from: {INPUT_FILE_PATH}")
print(f"Writing to:   {OUTPUT_FILE_PATH}")

try:
    # פתיחת שני הקבצים בו-זמנית
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f_out:
        
        # tqdm ייתן לנו סרגל התקדמות יפה
        for line in tqdm(f_in, desc="Filtering file"):
            total_lines += 1
            
            # ליבת הלוגיקה:
            # line.strip() -> מסיר רווחים ותווי ירידת שורה מההתחלה והסוף
            # .isdigit()  -> מחזיר True אם כל התווים בשורה הם ספרות, אחרת False
            
            # אם השורה, אחרי ניקוי רווחים, היא *לא* רק מספרים - נכתוב אותה לקובץ החדש
            if not line.strip().isdigit():
                f_out.write(line)
            else:
                removed_lines += 1

    print("\n--- Filtering Complete! ---")
    print(f"Total lines read: {total_lines:,}")
    print(f"Lines containing only numbers (removed): {removed_lines:,}")
    print(f"Total lines written to new file: {total_lines - removed_lines:,}")
    print(f"✅ Clean file is ready at: {OUTPUT_FILE_PATH}")

except FileNotFoundError:
    print(f"\n❌ ERROR: Input file not found at '{INPUT_FILE_PATH}'. Please check the path.")
