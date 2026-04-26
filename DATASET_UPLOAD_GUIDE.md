# Dataset Upload and Management Feature - Implementation Guide

## Overview
This implementation adds the ability for users to upload their own datasets and use all features of the Information Retrieval System.

## What Was Implemented

### 1. **Dataset Management Service** (`/workspace/dataset_management/dataset_management_service.py`)
Core business logic for:
- File validation (CSV, JSON, TXT, TSV)
- File parsing into document tuples
- Dataset registration in database
- Document insertion
- Status tracking
- CRUD operations (list, get status, delete)

### 2. **Dataset Management Handler** (`/workspace/dataset_management/dataset_management_handler.py`)
FastAPI REST API endpoints:
- `POST /api/v1/datasets/upload` - Upload new dataset
- `GET /api/v1/datasets` - List all datasets
- `GET /api/v1/datasets/{name}/status` - Get dataset status
- `DELETE /api/v1/datasets/{name}` - Delete dataset
- `POST /api/v1/datasets/{name}/process` - Trigger processing manually

### 3. **Updated Main App** (`/workspace/app.py`)
- Integrated dataset management router
- Updated version to 3.0.0
- Added feature documentation to health endpoint

### 4. **Test Suite** (`/workspace/test_dataset_upload.py`)
Comprehensive tests for:
- Health check
- CSV upload
- JSON upload
- TXT upload
- Dataset listing
- Status checking
- Invalid file rejection
- Dataset deletion

## How to Use

### Step 1: Start the API Server
```bash
cd /workspace
python app.py
```

The server will start on `http://127.0.0.1:8000`

### Step 2: Upload Your Dataset

#### Option A: Using cURL
```bash
# Upload CSV file
curl -X POST "http://127.0.0.1:8000/api/v1/datasets/upload" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "dataset_name=my_custom_dataset"
```

#### Option B: Using Python Requests
```python
import requests

with open('my_dataset.csv', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/api/v1/datasets/upload',
        files={'file': ('my_dataset.csv', f, 'text/csv')},
        data={'dataset_name': 'my_custom_dataset'}
    )
    print(response.json())
```

#### Option C: Using the Test Script
```bash
python test_dataset_upload.py csv
```

### Step 3: Check Dataset Status
```bash
curl http://127.0.0.1:8000/api/v1/datasets/my_custom_dataset/status
```

### Step 4: Trigger Processing (if not automatic)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/datasets/my_custom_dataset/process
```

### Step 5: Use All IR Features
Once processing is complete, your dataset is available for:
- TF-IDF search
- BERT semantic search
- Hybrid search
- Query suggestions
- Topic detection
- All other IR features

## Supported File Formats

### CSV Format
```csv
doc_id,text
doc_1,"Your document text here"
doc_2,"Another document"
```
- Must have at least 2 columns
- Automatically detects 'doc_id' and 'text' columns
- Falls back to first column as ID, second as text

### JSON Format
```json
[
  {"doc_id": "doc_1", "text": "Your document text"},
  {"doc_id": "doc_2", "text": "Another document"}
]
```
- Array of objects
- Automatically detects 'doc_id' and 'text' fields

### TXT Format
```
One document per line
Each line is treated as a separate document
Auto-generates doc_ids
```

### TSV Format
```
doc_id<TAB>document text
doc_1<TAB>Your document text here
```
- Tab-separated values
- Same structure as CSV but with tabs

## Testing

### Run Full Test Suite
```bash
python test_dataset_upload.py
```

### Run Individual Tests
```bash
python test_dataset_upload.py health    # Health check
python test_dataset_upload.py csv       # CSV upload only
python test_dataset_upload.py json      # JSON upload only
python test_dataset_upload.py txt       # TXT upload only
python test_dataset_upload.py list      # List datasets
python test_dataset_upload.py status my_dataset  # Check status
python test_dataset_upload.py delete my_dataset  # Delete dataset
python test_dataset_upload.py invalid   # Test invalid file rejection
```

## API Endpoints Reference

### Upload Dataset
```
POST /api/v1/datasets/upload
Content-Type: multipart/form-data

Parameters:
- file: Dataset file (CSV, JSON, TXT, TSV)
- dataset_name: Unique name for your dataset

Response:
{
  "message": "Dataset uploaded successfully",
  "dataset_name": "my_dataset",
  "dataset_id": 1,
  "documents_uploaded": 100,
  "status": "processing_started"
}
```

### List Datasets
```
GET /api/v1/datasets

Response:
{
  "datasets": [
    {"id": 1, "name": "my_dataset", "document_count": 100}
  ],
  "count": 1
}
```

### Get Dataset Status
```
GET /api/v1/datasets/{dataset_name}/status

Response:
{
  "dataset_name": "my_dataset",
  "dataset_id": 1,
  "total_documents": 100,
  "processed_documents": 80,
  "bert_processed_documents": 60,
  "processing_progress": 80.0,
  "bert_processing_progress": 60.0,
  "ready_for_search": false
}
```

### Delete Dataset
```
DELETE /api/v1/datasets/{dataset_name}

Response:
{
  "message": "Dataset 'my_dataset' deleted successfully"
}
```

### Trigger Processing
```
POST /api/v1/datasets/{dataset_name}/process

Response:
{
  "message": "Processing started for dataset 'my_dataset'",
  "status": "processing_queued"
}
```

## Architecture

```
User Upload → API Handler → Service Layer → Database
                ↓
         Background Tasks
                ↓
    Text Processing → BERT → Indexing
```

### Components:
1. **Handler Layer**: FastAPI endpoints, request validation
2. **Service Layer**: Business logic, file parsing, DB operations
3. **Background Tasks**: Async processing pipeline
4. **Database**: MySQL storage for documents and metadata

## Error Handling

The system handles:
- Invalid file types (returns 400)
- Empty files (returns 400)
- Malformed data (skips invalid rows)
- Database errors (rollback and error message)
- Missing datasets (returns 404)

## Next Steps for Production

1. **Add Authentication**: Secure upload endpoints
2. **Add Celery**: For robust background task queue
3. **Add Progress Tracking**: WebSocket or polling for real-time updates
4. **Add File Size Limits**: Configure max upload size
5. **Add Monitoring**: Logging and metrics
6. **Add Docker**: Containerize the application

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill process if needed
kill -9 <PID>
```

### Database connection error
```bash
# Ensure MySQL is running
sudo systemctl status mysql
# Check credentials in .env file
```

### Upload fails
- Check file format matches supported types
- Ensure file has at least 2 columns/fields
- Verify file is not empty
- Check logs for detailed error messages

## Example Usage Workflow

```python
import requests
import time

BASE_URL = "http://127.0.0.1:8000"

# 1. Upload dataset
with open('my_docs.csv', 'rb') as f:
    response = requests.post(
        f'{BASE_URL}/api/v1/datasets/upload',
        files={'file': ('my_docs.csv', f)},
        data={'dataset_name': 'my_research_papers'}
    )
print(f"Uploaded: {response.json()}")

# 2. Monitor status
while True:
    status = requests.get(f'{BASE_URL}/api/v1/datasets/my_research_papers/status')
    data = status.json()
    print(f"Progress: {data['processing_progress']:.1f}%")
    
    if data['ready_for_search']:
        print("Ready for search!")
        break
    
    time.sleep(10)  # Wait 10 seconds

# 3. Now use search features
search_query = {"query": "machine learning", "dataset": "my_research_papers"}
results = requests.post(f'{BASE_URL}/search/hybrid', json=search_query)
print(f"Found {len(results.json())} results")
```
