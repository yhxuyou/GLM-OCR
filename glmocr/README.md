# GLM-OCR Async Service Documentation

## Overview

The GLM-OCR async service provides asynchronous document processing with real-time progress updates via WebSocket. Unlike the synchronous Flask server, the async service:

- Processes documents in the background without blocking
- Provides real-time progress tracking via WebSocket
- Supports multiple concurrent document processing tasks
- Uses Redis for distributed state management and result aggregation
- Ideal for long-running OCR tasks and production deployments

## Installation Requirements

### System Requirements

- Python 3.10 or higher
- Redis server (for state management and result aggregation)

### Python Dependencies

Install the async server dependencies:

```bash
pip install 'glmocr[server]'
```

Or install dependencies individually:

```bash
pip install fastapi uvicorn[standard] websockets redis
```

### Redis Setup

The async service requires Redis for:
- Tracking document processing progress
- Storing intermediate results
- Managing distributed state

**Install Redis:**

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

**Start Redis:**

```bash
# Start Redis server
redis-server

# Or run in background
redis-server --daemonize yes
```

**Verify Redis is running:**

```bash
redis-cli ping
# Should return: PONG
```

## Configuration

### Redis Configuration

Configure Redis connection in your `config.yaml`:

```yaml
pipeline:
  redis:
    # Redis connection URL
    url: redis://localhost:6379/0
    
    # Prefix for all Redis keys (useful for multi-tenant deployments)
    key_prefix: glmocr
    
    # Maximum connections in the connection pool
    # Should be >= pipeline max_workers
    max_connections: 10
```

**Environment Variables:**

You can also configure Redis via environment variables:

```bash
export GLMOCR_REDIS_URL=redis://localhost:6379/0
```

**Redis URL Formats:**

```bash
# Local Redis
redis://localhost:6379/0

# Redis with password
redis://:password@localhost:6379/0

# Redis with SSL/TLS
rediss://localhost:6379/0

# Unix socket
unix:///path/to/socket?db=0
```

### Server Configuration

```yaml
server:
  host: "0.0.0.0"  # Bind to all interfaces
  port: 5002        # Server port
  debug: false      # Enable debug mode (development only)
```

## Starting the Async Server

### Method 1: Using the CLI (Recommended)

```bash
# Start with default configuration
glmocr server

# Start with custom config file
glmocr server --config my_config.yaml

# Override host and port
glmocr server --host 127.0.0.1 --port 8080

# Set log level
glmocr server --log-level DEBUG
```

### Method 2: Using Python Module

```bash
python -m glmocr server
```

### Method 3: Direct Python Script

```python
from glmocr.async_server import main
import sys

# Set command-line arguments
sys.argv = ["async_server", "--config", "config.yaml"]
main()
```

### Method 4: Using uvicorn Directly

```bash
# Create the app programmatically
uvicorn glmocr.async_server:create_app --factory --host 0.0.0.0 --port 5002
```

## API Endpoints

### 1. Submit Document for Processing

**Endpoint:** `POST /parse/async`

**Description:** Submit a document for asynchronous processing. Returns a document ID that can be used to track progress and retrieve results.

**Request:**

```bash
curl -X POST http://localhost:5002/parse/async \
  -F "file=@document.pdf"
```

**Python Example:**

```python
import httpx

with httpx.Client() as client:
    with open("document.pdf", "rb") as f:
        response = client.post(
            "http://localhost:5002/parse/async",
            files={"file": ("document.pdf", f, "application/pdf")}
        )
    
    result = response.json()
    doc_id = result["doc_id"]
    print(f"Document submitted: {doc_id}")
```

**Response:**

```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

### 2. Query Processing Status

**Endpoint:** `GET /parse/status/{doc_id}`

**Description:** Query the processing progress for a document.

**Request:**

```bash
curl http://localhost:5002/parse/status/550e8400-e29b-41d4-a716-446655440000
```

**Python Example:**

```python
import httpx

doc_id = "550e8400-e29b-41d4-a716-446655440000"

with httpx.Client() as client:
    response = client.get(f"http://localhost:5002/parse/status/{doc_id}")
    status = response.json()
    
    print(f"Progress: {status['completed']}/{status['total']} regions")
    print(f"Status: {status['status']}")
```

**Response:**

```json
{
  "completed": 5,
  "total": 10,
  "status": "processing"
}
```

**Status Values:**

- `processing` - Document is being processed
- `completed` - Processing is complete
- `not_found` - Document ID not found

### 3. Get Final Result

**Endpoint:** `GET /parse/result/{doc_id}`

**Description:** Retrieve the final OCR result for a completed document.

**Request:**

```bash
curl http://localhost:5002/parse/result/550e8400-e29b-41d4-a716-446655440000
```

**Python Example:**

```python
import httpx

doc_id = "550e8400-e29b-41d4-a716-446655440000"

with httpx.Client() as client:
    response = client.get(f"http://localhost:5002/parse/result/{doc_id}")
    
    if response.status_code == 200:
        result = response.json()
        print("OCR Result:", result)
    elif response.status_code == 202:
        print("Document still processing")
    elif response.status_code == 404:
        print("Document not found")
```

**Response (Success - 200):**

```json
{
  "unit_0": {
    "json_result": [...],
    "markdown_result": "# Document Title\n\nContent...",
    "original_images": ["file:///path/to/image.png"]
  },
  "unit_1": {
    "json_result": [...],
    "markdown_result": "...",
    "original_images": ["file:///path/to/image2.png"]
  }
}
```

**Error Responses:**

- `404 Not Found` - Document ID not found
- `202 Accepted` - Document processing not yet complete

### 4. WebSocket Real-time Updates

**Endpoint:** `WS /ws/{doc_id}`

**Description:** Connect to a WebSocket for real-time progress updates. The server sends progress messages every 1-2 seconds until processing is complete.

**Python Example:**

```python
import asyncio
import websockets
import json

async def monitor_progress(doc_id):
    uri = f"ws://localhost:5002/ws/{doc_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "progress":
                print(f"Progress: {data['completed']}/{data['total']}")
            
            elif data["type"] == "complete":
                print("Processing complete!")
                result = data["result"]
                print("Final result:", result)
                break
            
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                break

# Run the monitor
doc_id = "550e8400-e29b-41d4-a716-446655440000"
asyncio.run(monitor_progress(doc_id))
```

**Message Types:**

**Progress Update:**

```json
{
  "type": "progress",
  "completed": 5,
  "total": 10
}
```

**Completion:**

```json
{
  "type": "complete",
  "result": {
    "unit_0": {
      "json_result": [...],
      "markdown_result": "...",
      "original_images": [...]
    }
  }
}
```

**Error:**

```json
{
  "type": "error",
  "message": "Document not found"
}
```

### 5. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the server is running.

**Request:**

```bash
curl http://localhost:5002/health
```

**Response:**

```json
{
  "status": "ok"
}
```

## Comparison: Sync vs Async Server

| Feature | Sync Server (Flask) | Async Server (FastAPI) |
|---------|---------------------|------------------------|
| **Framework** | Flask | FastAPI |
| **Processing** | Synchronous (blocking) | Asynchronous (non-blocking) |
| **Progress Tracking** | No | Yes (via WebSocket) |
| **Concurrent Tasks** | Limited | Unlimited |
| **State Management** | In-memory | Redis (distributed) |
| **Long-running Tasks** | Not ideal | Excellent |
| **Production Ready** | Development/Testing | Production |
| **WebSocket Support** | No | Yes |
| **Result Aggregation** | Single response | Distributed aggregation |
| **Use Case** | Simple scripts, testing | Production deployments, batch processing |

### When to Use Each

**Use Sync Server when:**
- Simple testing or development
- Processing single documents
- No need for progress tracking
- Quick prototyping

**Use Async Server when:**
- Production deployment
- Processing multiple documents concurrently
- Need real-time progress updates
- Long-running OCR tasks
- Distributed processing requirements
- Need fault tolerance and state persistence

## Complete Workflow Example

### 1. Submit Document and Poll for Result

```python
import httpx
import time

def process_document_async(file_path, server_url="http://localhost:5002"):
    """Submit a document and wait for the result using polling."""
    
    # Step 1: Submit document
    with open(file_path, "rb") as f:
        response = httpx.post(
            f"{server_url}/parse/async",
            files={"file": (file_path, f)}
        )
    
    doc_id = response.json()["doc_id"]
    print(f"Submitted document: {doc_id}")
    
    # Step 2: Poll for completion
    while True:
        status_response = httpx.get(f"{server_url}/parse/status/{doc_id}")
        status = status_response.json()
        
        print(f"Progress: {status['completed']}/{status['total']}")
        
        if status["status"] == "completed":
            break
        
        time.sleep(2)  # Wait 2 seconds before next check
    
    # Step 3: Get final result
    result_response = httpx.get(f"{server_url}/parse/result/{doc_id}")
    result = result_response.json()
    
    print("Processing complete!")
    return result

# Usage
result = process_document_async("document.pdf")
```

### 2. Submit Document and Monitor via WebSocket

```python
import asyncio
import httpx
import websockets
import json

async def process_with_websocket(file_path, server_url="http://localhost:5002"):
    """Submit a document and monitor progress via WebSocket."""
    
    # Step 1: Submit document
    with open(file_path, "rb") as f:
        response = httpx.post(
            f"{server_url}/parse/async",
            files={"file": (file_path, f)}
        )
    
    doc_id = response.json()["doc_id"]
    print(f"Submitted document: {doc_id}")
    
    # Step 2: Connect to WebSocket for real-time updates
    ws_url = server_url.replace("http://", "ws://")
    async with websockets.connect(f"{ws_url}/ws/{doc_id}") as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "progress":
                print(f"Progress: {data['completed']}/{data['total']}")
            
            elif data["type"] == "complete":
                print("Processing complete!")
                return data["result"]
            
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                return None

# Usage
result = asyncio.run(process_with_websocket("document.pdf"))
```

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```python
import httpx
from httpx import HTTPError

def safe_submit_document(file_path, server_url="http://localhost:5002"):
    try:
        with open(file_path, "rb") as f:
            response = httpx.post(
                f"{server_url}/parse/async",
                files={"file": (file_path, f)},
                timeout=30.0  # Set timeout
            )
        response.raise_for_status()
        return response.json()
    
    except httpx.TimeoutException:
        print("Request timed out")
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None
```

### 2. Connection Pooling

For high-throughput scenarios, reuse HTTP clients:

```python
import httpx

# Create a client with connection pooling
client = httpx.Client(
    timeout=30.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)

# Reuse the client for multiple requests
for file_path in file_list:
    with open(file_path, "rb") as f:
        response = client.post(
            "http://localhost:5002/parse/async",
            files={"file": (file_path, f)}
        )
    doc_id = response.json()["doc_id"]
    # ... process document

# Close when done
client.close()
```

### 3. Retry Logic

Implement retry logic for transient failures:

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def submit_with_retry(file_path, server_url="http://localhost:5002"):
    with open(file_path, "rb") as f:
        response = httpx.post(
            f"{server_url}/parse/async",
            files={"file": (file_path, f)}
        )
    response.raise_for_status()
    return response.json()
```

### 4. Resource Cleanup

Redis keys are automatically cleaned up after 24 hours. For manual cleanup:

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Clean up specific document
doc_id = "550e8400-e29b-41d4-a716-446655440000"
r.delete(f"glmocr:doc:{doc_id}:meta")
r.delete(f"glmocr:doc:{doc_id}:regions")

# Clean up all glmocr keys (use with caution)
keys = r.keys("glmocr:*")
if keys:
    r.delete(*keys)
```

## Troubleshooting

### Redis Connection Issues

**Problem:** Cannot connect to Redis

**Solutions:**
- Verify Redis is running: `redis-cli ping`
- Check Redis URL in config
- Ensure firewall allows Redis port (default: 6379)
- Check Redis logs for errors

### WebSocket Connection Fails

**Problem:** WebSocket connection refused

**Solutions:**
- Verify server is running with async mode
- Check WebSocket URL format (use `ws://` not `http://`)
- Ensure no proxy is blocking WebSocket connections
- Check browser console for errors

### Document Processing Stuck

**Problem:** Document status remains "processing" indefinitely

**Solutions:**
- Check server logs for errors
- Verify Redis is accessible
- Check pipeline workers are running
- Restart the async server

### High Memory Usage

**Problem:** Server consumes too much memory

**Solutions:**
- Reduce `max_workers` in config
- Reduce `region_maxsize` in config
- Process fewer documents concurrently
- Increase Redis memory if using result caching

## Additional Resources

- [GLM-OCR Main Documentation](../README.md)
- [Configuration Guide](config.yaml)
- [API Reference](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/zai-org/glm-ocr/issues
- Documentation: https://github.com/zai-org/glm-ocr#readme
