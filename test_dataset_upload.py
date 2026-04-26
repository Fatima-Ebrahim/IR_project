"""
Test Script for Dataset Upload and Management Feature

This script tests the complete workflow of uploading and processing custom datasets.
"""

import requests
import time
import json
import os

BASE_URL = "http://127.0.0.1:8000"


def create_test_csv(filename="test_dataset.csv"):
    """Create a sample CSV file for testing"""
    content = """doc_id,text
doc_1,"Machine learning is a subset of artificial intelligence that enables systems to learn from data."
doc_2,"Natural language processing helps computers understand human language."
doc_3,"Information retrieval systems help users find relevant documents in large collections."
doc_4,"Deep learning uses neural networks with multiple layers to extract features from data."
doc_5,"Search engines use ranking algorithms to order results by relevance."
"""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"✓ Created test CSV file: {filename}")
    return filename


def create_test_json(filename="test_dataset.json"):
    """Create a sample JSON file for testing"""
    data = [
        {"doc_id": "doc_1", "text": "Machine learning is a subset of artificial intelligence."},
        {"doc_id": "doc_2", "text": "Natural language processing helps computers understand language."},
        {"doc_id": "doc_3", "text": "Information retrieval systems find relevant documents."},
        {"doc_id": "doc_4", "text": "Deep learning uses neural networks with multiple layers."},
        {"doc_id": "doc_5", "text": "Search engines use ranking algorithms for relevance."}
    ]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Created test JSON file: {filename}")
    return filename


def create_test_txt(filename="test_dataset.txt"):
    """Create a sample TXT file for testing (one document per line)"""
    content = """Machine learning is a subset of artificial intelligence.
Natural language processing helps computers understand language.
Information retrieval systems find relevant documents.
Deep learning uses neural networks with multiple layers.
Search engines use ranking algorithms for relevance.
"""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"✓ Created test TXT file: {filename}")
    return filename


def test_health_check():
    """Test API health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    print(f"✓ Health check passed: {response.json()}")
    return True


def test_upload_csv():
    """Test uploading a CSV dataset"""
    print("\n=== Testing CSV Upload ===")
    
    # Create test file
    filename = create_test_csv()
    
    # Upload file
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/csv')}
        data = {'dataset_name': 'test_ml_docs_csv'}
        response = requests.post(f"{BASE_URL}/api/v1/datasets/upload", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
    
    assert response.status_code == 200
    result = response.json()
    assert result['dataset_name'] == 'test_ml_docs_csv'
    assert 'documents_uploaded' in result
    
    print("✓ CSV upload test passed")
    return result


def test_upload_json():
    """Test uploading a JSON dataset"""
    print("\n=== Testing JSON Upload ===")
    
    # Create test file
    filename = create_test_json()
    
    # Upload file
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'application/json')}
        data = {'dataset_name': 'test_ml_docs_json'}
        response = requests.post(f"{BASE_URL}/api/v1/datasets/upload", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
    
    assert response.status_code == 200
    result = response.json()
    assert result['dataset_name'] == 'test_ml_docs_json'
    
    print("✓ JSON upload test passed")
    return result


def test_upload_txt():
    """Test uploading a TXT dataset"""
    print("\n=== Testing TXT Upload ===")
    
    # Create test file
    filename = create_test_txt()
    
    # Upload file
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/plain')}
        data = {'dataset_name': 'test_ml_docs_txt'}
        response = requests.post(f"{BASE_URL}/api/v1/datasets/upload", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
    
    assert response.status_code == 200
    result = response.json()
    assert result['dataset_name'] == 'test_ml_docs_txt'
    
    print("✓ TXT upload test passed")
    return result


def test_list_datasets():
    """Test listing all datasets"""
    print("\n=== Testing List Datasets ===")
    
    response = requests.get(f"{BASE_URL}/api/v1/datasets")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    result = response.json()
    assert 'datasets' in result
    assert 'count' in result
    
    print(f"✓ Found {result['count']} datasets")
    return result


def test_get_dataset_status(dataset_name):
    """Test getting dataset status"""
    print(f"\n=== Testing Dataset Status for '{dataset_name}' ===")
    
    response = requests.get(f"{BASE_URL}/api/v1/datasets/{dataset_name}/status")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    result = response.json()
    assert 'dataset_name' in result
    assert 'total_documents' in result
    
    print(f"✓ Status retrieved: {result['total_documents']} documents")
    return result


def test_trigger_processing(dataset_name):
    """Test triggering manual processing"""
    print(f"\n=== Testing Processing Trigger for '{dataset_name}' ===")
    
    response = requests.post(f"{BASE_URL}/api/v1/datasets/{dataset_name}/process")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    
    print("✓ Processing triggered successfully")
    return response.json()


def test_delete_dataset(dataset_name):
    """Test deleting a dataset"""
    print(f"\n=== Testing Delete Dataset '{dataset_name}' ===")
    
    response = requests.delete(f"{BASE_URL}/api/v1/datasets/{dataset_name}")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    
    print("✓ Dataset deleted successfully")
    return response.json()


def test_invalid_file_type():
    """Test uploading an invalid file type"""
    print("\n=== Testing Invalid File Type ===")
    
    # Create invalid file
    filename = "test_invalid.xml"
    with open(filename, 'w') as f:
        f.write("<xml>Invalid format</xml>")
    
    # Try to upload
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'application/xml')}
        data = {'dataset_name': 'test_invalid'}
        response = requests.post(f"{BASE_URL}/api/v1/datasets/upload", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
    
    assert response.status_code == 400
    
    print("✓ Invalid file type correctly rejected")
    return response.json()


def run_all_tests():
    """Run complete test suite"""
    print("=" * 60)
    print("DATASET UPLOAD AND MANAGEMENT - TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2: Upload CSV
        csv_result = test_upload_csv()
        
        # Test 3: Upload JSON
        json_result = test_upload_json()
        
        # Test 4: Upload TXT
        txt_result = test_upload_txt()
        
        # Test 5: List datasets
        list_result = test_list_datasets()
        
        # Test 6: Get status
        status_result = test_get_dataset_status('test_ml_docs_csv')
        
        # Test 7: Invalid file type
        test_invalid_file_type()
        
        # Test 8: Trigger processing (optional - may take time)
        # test_trigger_processing('test_ml_docs_csv')
        
        # Test 9: Delete test datasets
        test_delete_dataset('test_ml_docs_csv')
        test_delete_dataset('test_ml_docs_json')
        test_delete_dataset('test_ml_docs_txt')
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR: Is the API server running?")
        print("Start the server with: python app.py")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if test_name == "health":
            test_health_check()
        elif test_name == "csv":
            test_upload_csv()
        elif test_name == "json":
            test_upload_json()
        elif test_name == "txt":
            test_upload_txt()
        elif test_name == "list":
            test_list_datasets()
        elif test_name == "status":
            test_get_dataset_status(sys.argv[2] if len(sys.argv) > 2 else "test_ml_docs_csv")
        elif test_name == "delete":
            test_delete_dataset(sys.argv[2] if len(sys.argv) > 2 else "test_ml_docs_csv")
        elif test_name == "invalid":
            test_invalid_file_type()
        elif test_name == "process":
            test_trigger_processing(sys.argv[2] if len(sys.argv) > 2 else "test_ml_docs_csv")
        else:
            print(f"Unknown test: {test_name}")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
