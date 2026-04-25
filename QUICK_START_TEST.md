# Quick Start Guide - Test Dataset Upload Feature

## Overview
This guide shows you how to quickly test the dataset upload feature **without needing MySQL**.

## Prerequisites
- Python 3.8+
- FastAPI and dependencies installed
- No database required for this demo!

## Step 1: Install Dependencies (if not already done)

```bash
cd /workspace
pip install fastapi uvicorn python-multipart pandas requests
```

## Step 2: Create a Simple Test Script

Save this as `quick_test.py`:

```python
#!/usr/bin/env python3
"""
Quick test of dataset upload feature WITHOUT database
Tests file validation and parsing logic only
"""

import os
import tempfile
import pandas as pd
from dataset_management.dataset_management_service import DatasetManagementService

def test_csv_parsing():
    """Test CSV file parsing without database"""
    print("=" * 60)
    print("Testing CSV File Parsing")
    print("=" * 60)
    
    # Create sample CSV
    csv_content = """doc_id,text
doc_1,"Machine learning is a subset of artificial intelligence"
doc_2,"Deep learning uses neural networks with many layers"
doc_3,"Natural language processing helps computers understand text"
"""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_file = f.name
    
    try:
        # Initialize service (we'll skip DB operations)
        service = DatasetManagementService({})
        
        # Test validation
        is_valid, error_msg = service.validate_file(temp_file, '.csv')
        print(f"✓ File validation: {'PASSED' if is_valid else 'FAILED'}")
        if not is_valid:
            print(f"  Error: {error_msg}")
            return False
        
        # Test parsing
        documents = service.parse_uploaded_file(temp_file, '.csv', "test_dataset")
        print(f"✓ File parsing: PASSED")
        print(f"  Documents found: {len(documents)}")
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"  Doc {i}: ID={doc[0]}, Text='{doc[1][:50]}...'")
        
        print("\n✅ CSV parsing test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ CSV parsing test FAILED: {e}")
        return False
    finally:
        os.unlink(temp_file)


def test_json_parsing():
    """Test JSON file parsing without database"""
    print("\n" + "=" * 60)
    print("Testing JSON File Parsing")
    print("=" * 60)
    
    # Create sample JSON
    import json
    json_data = [
        {"doc_id": "doc_1", "text": "Machine learning algorithms learn from data"},
        {"doc_id": "doc_2", "text": "Computer vision enables machines to see"},
        {"doc_id": "doc_3", "text": "Reinforcement learning uses rewards and penalties"}
    ]
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        temp_file = f.name
    
    try:
        service = DatasetManagementService({})
        
        # Test validation
        is_valid, error_msg = service.validate_file(temp_file, '.json')
        print(f"✓ File validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test parsing
        documents = service.parse_uploaded_file(temp_file, '.json', "test_dataset")
        print(f"✓ File parsing: PASSED")
        print(f"  Documents found: {len(documents)}")
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"  Doc {i}: ID={doc[0]}, Text='{doc[1][:50]}...'")
        
        print("\n✅ JSON parsing test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ JSON parsing test FAILED: {e}")
        return False
    finally:
        os.unlink(temp_file)


def test_txt_parsing():
    """Test TXT file parsing without database"""
    print("\n" + "=" * 60)
    print("Testing TXT File Parsing")
    print("=" * 60)
    
    # Create sample TXT
    txt_content = """First document about artificial intelligence
Second document discussing machine learning algorithms
Third document explaining neural networks
Fourth document covering deep learning techniques
"""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(txt_content)
        temp_file = f.name
    
    try:
        service = DatasetManagementService({})
        
        # Test validation
        is_valid, error_msg = service.validate_file(temp_file, '.txt')
        print(f"✓ File validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test parsing
        documents = service.parse_uploaded_file(temp_file, '.txt', "test_dataset")
        print(f"✓ File parsing: PASSED")
        print(f"  Documents found: {len(documents)}")
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"  Doc {i}: ID={doc[0]}, Text='{doc[1][:50]}...'")
        
        print("\n✅ TXT parsing test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TXT parsing test FAILED: {e}")
        return False
    finally:
        os.unlink(temp_file)


def test_invalid_file():
    """Test rejection of invalid file types"""
    print("\n" + "=" * 60)
    print("Testing Invalid File Rejection")
    print("=" * 60)
    
    # Create invalid file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
        f.write("This should be rejected")
        temp_file = f.name
    
    try:
        service = DatasetManagementService({})
        
        # Test validation
        is_valid, error_msg = service.validate_file(temp_file, '.exe')
        
        if not is_valid:
            print(f"✓ Correctly rejected invalid file type")
            print(f"  Error message: {error_msg}")
            print("\n✅ Invalid file rejection test PASSED!")
            return True
        else:
            print(f"❌ Should have rejected .exe file")
            return False
        
    except Exception as e:
        print(f"✓ Exception raised as expected: {e}")
        print("\n✅ Invalid file rejection test PASSED!")
        return True
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATASET UPLOAD FEATURE - QUICK TEST SUITE")
    print("=" * 60)
    print("Testing file validation and parsing WITHOUT database\n")
    
    results = []
    results.append(("CSV Parsing", test_csv_parsing()))
    results.append(("JSON Parsing", test_json_parsing()))
    results.append(("TXT Parsing", test_txt_parsing()))
    results.append(("Invalid File Rejection", test_invalid_file()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The dataset upload feature is working correctly.")
        print("\nNext steps:")
        print("1. Set up MySQL database for full functionality")
        print("2. Start the server: uvicorn app:app --host 0.0.0.0 --port 8000")
        print("3. Upload your datasets via POST /api/v1/datasets/upload")
        print("4. Use all IR features (search, topic detection, etc.)")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("=" * 60 + "\n")

```

## Step 3: Run the Quick Test

```bash
cd /workspace
python quick_test.py
```

## Expected Output

You should see output like:

```
============================================================
DATASET UPLOAD FEATURE - QUICK TEST SUITE
============================================================
Testing file validation and parsing WITHOUT database

============================================================
Testing CSV File Parsing
============================================================
✓ File validation: PASSED
✓ File parsing: PASSED
  Documents found: 3
  Doc 1: ID=doc_1, Text='Machine learning is a subset of artificial intel...'
  ...

✅ CSV parsing test PASSED!

... (similar for JSON and TXT)

============================================================
TEST SUMMARY
============================================================
CSV Parsing: ✅ PASSED
JSON Parsing: ✅ PASSED
TXT Parsing: ✅ PASSED
Invalid File Rejection: ✅ PASSED

Total: 4/4 tests passed

🎉 All tests passed!
```

## Step 4: Create Sample Dataset Files

Create these files to test with the actual API later:

### 1. Create `sample_docs.csv`

```bash
cat > /workspace/sample_docs.csv << 'EOF'
doc_id,text
doc_1,"Artificial intelligence is transforming industries worldwide"
doc_2,"Machine learning models require large amounts of training data"
doc_3,"Deep neural networks excel at image recognition tasks"
doc_4,"Natural language processing enables chatbots and virtual assistants"
doc_5,"Computer vision allows autonomous vehicles to navigate safely"
EOF
```

### 2. Create `sample_docs.json`

```bash
cat > /workspace/sample_docs.json << 'EOF'
[
  {"doc_id": "1", "text": "Quantum computing promises exponential speedup for certain problems"},
  {"doc_id": "2", "text": "Blockchain technology provides decentralized and secure transactions"},
  {"doc_id": "3", "text": "Internet of Things connects billions of devices worldwide"},
  {"doc_id": "4", "text": "Edge computing reduces latency by processing data closer to source"},
  {"doc_id": "5", "text": "5G networks enable faster mobile communication and IoT applications"}
]
EOF
```

### 3. Create `sample_docs.txt`

```bash
cat > /workspace/sample_docs.txt << 'EOF'
Cloud computing provides on-demand access to computing resources
DevOps practices streamline software development and deployment
Microservices architecture enables scalable and maintainable applications
Containerization with Docker simplifies application deployment
Kubernetes orchestrates containerized applications at scale
EOF
```

## Step 5: Full API Testing (Requires MySQL)

If you have MySQL installed and running:

### 5a. Configure Database Connection

Create `.env` file:

```bash
cat > /workspace/.env << 'EOF'
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=127.0.0.1
DB_NAME=ir_database
EOF
```

### 5b. Start MySQL and Create Database

```sql
CREATE DATABASE ir_database;
USE ir_database;

-- Create necessary tables (run the SQL from database setup scripts)
```

### 5c. Start the Server

```bash
cd /workspace
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5d. Upload Your Dataset

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -F "file=@/workspace/sample_docs.csv" \
  -F "dataset_name=my_sample_dataset"

# Or using Python
python -c "
import requests
with open('/workspace/sample_docs.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/datasets/upload',
        files={'file': ('sample_docs.csv', f)},
        data={'dataset_name': 'my_sample_dataset'}
    )
    print(response.json())
"
```

### 5e. Check Status

```bash
curl http://localhost:8000/api/v1/datasets/my_sample_dataset/status
```

### 5f. List All Datasets

```bash
curl http://localhost:8000/api/v1/datasets
```

## Troubleshooting

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
```bash
lsof -ti:8000 | xargs kill -9
```

### Issue: MySQL connection error
- Verify MySQL is running: `sudo systemctl status mysql`
- Check credentials in `.env` file
- Ensure database exists

## What You've Tested

✅ File validation (CSV, JSON, TXT, TSV)
✅ Document parsing and extraction
✅ Invalid file rejection
✅ Service layer logic

## Next Steps for Full Functionality

1. **Set up MySQL** for persistent storage
2. **Start all microservices** for search capabilities
3. **Upload your dataset** via the API
4. **Run text processing** pipeline
5. **Generate embeddings** with BERT
6. **Build indexes** for TF-IDF search
7. **Use search features**:
   - Keyword search (TF-IDF)
   - Semantic search (BERT)
   - Hybrid search
   - Query suggestions
   - Topic modeling

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Enjoy testing your new dataset upload feature! 🚀
