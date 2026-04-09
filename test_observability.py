from fastapi.testclient import TestClient
from app.main import app
import sys

client = TestClient(app)

print("Starting tests focused on observability (Logging)...\n", file=sys.stderr)

# Test 1: Home endpoint
print("--- Requesting / ---", file=sys.stderr)
response = client.get("/")
print("Response:", response.json())
print()

try:
    # Test 2: Chat endpoint (we will send dummy data to force the agent to fail or process it, to see standard error/info logs)
    print("--- Requesting /chat ---", file=sys.stderr)
    response = client.post("/chat", json={"user_id": "test_user_123", "message": "hello"})
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print(f"Chat request raised exception: {e}")

print("\nFinished tests.")
