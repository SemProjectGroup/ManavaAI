import requests
import json

def test_humanizer_endpoint():
    url = "http://127.0.0.1:8000/api/humanize"
    payload = {"text": "AI generated content is often very predictable and lacks burstiness."}
    
    print(f"Testing connection to {url}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS! Humanity Score: {data['humanity_score']}%")
        assert data['humanity_score'] == 95.5
    else:
        print(f"FAILED with status code: {response.status_code}")

if __name__ == "__main__":
    test_humanizer_endpoint()