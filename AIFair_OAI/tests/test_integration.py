# tests/test_integration.py

import requests

def test_api_response():
    response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
    assert response.status_code == 200
    assert "userId" in response.json()


# Run the test
# pytest tests/test_integration.py