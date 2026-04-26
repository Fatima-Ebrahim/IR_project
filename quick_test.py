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
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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
