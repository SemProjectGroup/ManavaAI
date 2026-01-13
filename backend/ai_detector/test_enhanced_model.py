# Praful Bhatt roll 61
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AIDetectorTester:
    def __init__(self, model_path="trained_model_enhanced/best"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.classes = ["Human", "AI"]

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

        return {
            "prediction": self.classes[pred.item()],
            "confidence": conf.item(),
            "probabilities": {
                "human": probs[0][0].item(),
                "ai": probs[0][1].item()
            }
        }

def main():
    tester = AIDetectorTester()
    
    print("-" * 30)
    print("AI Detector Test Interface")
    print("-" * 30)
    
    while True:
        text = input("\nEnter text to test (or 'q' to quit): ")
        if text.lower() == 'q':
            break
            
        if len(text.strip()) < 10:
            print("Please enter a longer text for better accuracy.")
            continue

        result = tester.predict(text)
        
        print(f"\nResult: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Human Prob: {result['probabilities']['human']:.4f}")
        print(f"AI Prob: {result['probabilities']['ai']:.4f}")

if __name__ == "__main__":
    main()