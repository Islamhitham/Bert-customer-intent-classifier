import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

class EmotionClassifier:
    def __init__(self, model_path="./emotion_bert_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_class_id]
        confidence = probabilities[0][predicted_class_id].item()
        
        return {
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": {
                self.model.config.id2label[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from text.")
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument("--model_path", type=str, default="./emotion_bert_model", help="Path to saved model")
    
    args = parser.parse_args()
    
    try:
        classifier = EmotionClassifier(model_path=args.model_path)
        result = classifier.predict(args.text)
        
        print("\n--- Prediction Result ---")
        print(f"Text: {args.text}")
        print(f"Emotion: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nFull Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
            
    except OSError:
        print(f"Error: Model not found at {args.model_path}. Please run train.py first.")
