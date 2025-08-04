import os

# --- הגדרות ---
# ודא שהנתיב הזה מצביע על קובץ הפלט *החדש* שיצרת עם הסקריפט הקודם
FILE_TO_VERIFY = os.path.join('data', 'sanskrit_preprocessed', 'sanskrit_final_for_bert.txt')

# כמה שורות להדפיס כמדגם
NUM_SAMPLE_LINES = 150
# אחרי כמה שורות שגויות להפסיק את הבדיקה
MAX_ERRORS_TO_SHOW = 10


def run_verification(file_path):
    """
    מריץ בדיקות על קובץ הטקסט כדי לוודא ששורות המכילות מספרים בלבד סוננו בהצלחה.
    """
    print(f"--- Starting Verification for: {file_path} ---")
    
    # --- בדיקה 1: קיום הקובץ ---
    print("\n[Test 1: File Existence]")
    if not os.path.exists(file_path):
        print(f"❌ FAIL: The file was not found at '{file_path}'.")
        print("Please make sure you ran the filter script successfully and the path is correct.")
        return
    print("✅ PASS: File found.")

    # --- בדיקה 2: חיפוש שורות עם מספרים בלבד ---
    print("\n[Test 2: Searching for Numeric-Only Lines]")
    problematic_lines = []
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line_count += 1
                if line.strip().isdigit():
                    problematic_lines.append((i, line.strip()))
                    if len(problematic_lines) >= MAX_ERRORS_TO_SHOW:
                        break # מצאנו מספיק דוגמאות, אין צורך להמשיך
    
    except Exception as e:
        print(f"❌ ERROR: Could not read the file. Error: {e}")
        return

    if not problematic_lines:
        print("✅ PASS: No lines containing only numbers were found.")
        final_result = True
    else:
        print("❌ FAIL: Found lines that should have been filtered out!")
        print("Examples of problematic lines found:")
        for line_num, content in problematic_lines:
            print(f"  - Line {line_num}: '{content}'")
        final_result = False

    # --- בדיקה 3: הצגת מדגם מהקובץ ---
    print("\n[Test 3: Content Sample]")
    print(f"Displaying the first {NUM_SAMPLE_LINES} lines from the file:")
    print("-" * 40)
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLE_LINES:
                break
            print(f"  {i+1:02d}: {line.strip()}")
    print("-" * 40)

    # --- סיכום סופי ---
    print("\n--- Verification Summary ---")
    if final_result:
        print("✅✅✅ The file seems to be correctly filtered and is ready for the next step!")
    else:
        print("❌❌❌ The file still contains errors. The filtering was not successful.")


if __name__ == "__main__":
    run_verification(FILE_TO_VERIFY)
