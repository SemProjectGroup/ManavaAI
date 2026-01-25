
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app.main import app

def test_api():
    print("Setting up mocks...")
    
    # Mock the CombinedEvaluator
    with patch('app.main.CombinedEvaluator') as MockEvaluator:
        # Setup mock instance
        mock_instance = MockEvaluator.return_value
        
        # Mock evaluate return
        mock_score = MagicMock()
        mock_score.to_dict.return_value = {
            "perplexity": 10.5,
            "overall_human_score": 85.0,
            "likely_source": "Human-written"
        }
        mock_instance.evaluate.return_value = mock_score
        
        # Mock compare return
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "before_score": {},
            "after_score": {},
            "overall_improvement": 10.0
        }
        mock_instance.compare.return_value = mock_result
        
        print("Initializing TestClient...")
        with TestClient(app) as client:
            
            # Test Root
            print("Testing GET / ...")
            response = client.get("/")
            assert response.status_code == 200
            print("Root endpoint OK:", response.json())
            
            # Test Analyze
            print("\nTesting POST /analyze ...")
            analyze_payload = {"text": "This is a test text."}
            response = client.post("/analyze", json=analyze_payload)
            assert response.status_code == 200
            print("Analyze endpoint OK:", response.json())
            
            # Test Compare
            print("\nTesting POST /compare ...")
            compare_payload = {
                "original_text": "AI text",
                "humanized_text": "Human text"
            }
            response = client.post("/compare", json=compare_payload)
            assert response.status_code == 200
            print("Compare endpoint OK:", response.json())

if __name__ == "__main__":
    try:
        test_api()
        print("\n✅ Verification Successful!")
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
