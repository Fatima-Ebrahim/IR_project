# file: check_dependencies.py
import sys
import os
import nltk

def print_environment_info():
    """
    (جديد) يطبع معلومات تشخيصية حول بيئة بايثون المستخدمة.
    """
    print("--- Environment Diagnostics ---")
    python_executable = sys.executable
    print(f"🐍 Python Executable Path: {python_executable}")
    
    # التحقق مما إذا كان المسار يشير إلى بيئة افتراضية
    if 'venv' in python_executable or 'env' in python_executable:
        print("✅ It looks like you are running from a virtual environment.")
    else:
        print("⚠️ WARNING: It does not look like you are running from a virtual environment.")
        print("   This can cause 'module not found' errors.")
        print("   Please make sure to activate it first (e.g., .\\venv\\Scripts\\activate on Windows).")
    print("-" * 30)

def check_and_download_nltk_data():
    """
    Checks for necessary NLTK data packages and downloads them if missing.
    This is the recommended way, run this script once before starting the service.
    """
    required_packages = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }
    
    print("\nChecking NLTK data packages...")
    all_good = True
    
    for path, pkg_id in required_packages.items():
        try:
            nltk.data.find(path)
            print(f"✅ '{pkg_id}' is already downloaded.")
        except LookupError:
            print(f"🟡 '{pkg_id}' not found. Downloading...")
            try:
                nltk.download(pkg_id, quiet=True)
                print(f"✅ Successfully downloaded '{pkg_id}'.")
            except Exception as e:
                print(f"❌ Failed to download '{pkg_id}'. Error: {e}")
                all_good = False
                
    return all_good

def check_libraries():
    """
    Checks if essential Python libraries are installed and shows where they are located.
    """
    print("\nChecking required Python libraries...")
    try:
        import pyspellchecker
        print("✅ 'pyspellchecker' is installed.")
        # (جديد) إظهار مسار المكتبة لتأكيد أنها من البيئة الافتراضية
        library_path = os.path.dirname(pyspellchecker.__file__)
        print(f"   └── Location: {library_path}")
        return True
    except ImportError:
        print("❌ 'pyspellchecker' is NOT installed.")
        print("   Please install it by running: pip install pyspellchecker")
        return False

if __name__ == "__main__":
    print("--- Dependency Verification Tool (v2.0 with Diagnostics) ---")
    print_environment_info()
    libs_ok = check_libraries()
    nltk_ok = check_and_download_nltk_data()
    
    print("\n--- Summary ---")
    if libs_ok and nltk_ok:
        print("✅ All dependencies are correctly set up. You are ready to go!")
    else:
        print("❌ Some dependencies are missing. Please follow the instructions above.")

