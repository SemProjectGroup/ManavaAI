# inference.py
"""
Inference utilities for AI Detection
"""

import os
from typing import Dict, List, Optional
from colorama import init, Fore, Style

from model import AIDetector
from config import Config

init()  # Initialize colorama


class Detector:
    """Simple inference interface"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize detector
        
        Args:
            model_path: Path to saved model directory
        """
        # Find model path
        if model_path is None:
            model_path = self._find_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Please train the model first using: python train.py"
            )
        
        # Load model
        config = Config()
        self.detector = AIDetector(config=config, model_path=model_path)
        self.detector.eval_mode()
    
    def _find_model(self) -> str:
        """Find the best available model"""
        default_dir = "trained_model"
        
        # Try best model first
        best_path = os.path.join(default_dir, 'best')
        if os.path.exists(best_path):
            return best_path
        
        # Try final model
        final_path = os.path.join(default_dir, 'final')
        if os.path.exists(final_path):
            return final_path
        
        # Try root model dir
        if os.path.exists(default_dir):
            return default_dir
        
        return best_path  # Return default path even if not exists
    
    def detect(self, text: str) -> Dict:
        """
        Detect if text is AI-generated
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with:
                - prediction: "Human" or "AI"
                - is_ai: Boolean
                - ai_percentage: AI probability as percentage
                - human_percentage: Human probability as percentage
                - confidence: Confidence score
        """
        result = self.detector.predict(text)
        
        return {
            'prediction': result['prediction'],
            'is_ai': result['is_ai'],
            'ai_percentage': round(result['ai_probability'] * 100, 2),
            'human_percentage': round(result['human_probability'] * 100, 2),
            'confidence': round(result['confidence'] * 100, 2)
        }
    
    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect multiple texts"""
        results = self.detector.predict_batch(texts)
        
        return [{
            'prediction': r['prediction'],
            'is_ai': r['is_ai'],
            'ai_percentage': round(r['ai_probability'] * 100, 2),
            'human_percentage': round(r['human_probability'] * 100, 2),
            'confidence': round(r['confidence'] * 100, 2)
        } for r in results]
    
    def analyze(self, text: str, show_bar: bool = True) -> Dict:
        """
        Analyze text and return detailed results
        
        Args:
            text: Text to analyze
            show_bar: Whether to include visual bar representation
            
        Returns:
            Detailed analysis results
        """
        result = self.detect(text)
        
        if show_bar:
            # Create visual bar
            ai_pct = result['ai_percentage']
            bar_length = 20
            filled = int(ai_pct / 100 * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            result['visual_bar'] = bar
        
        return result


def print_result(result: Dict, text: str = None):
    """Pretty print detection result"""
    
    print("\n" + "="*50)
    
    if text:
        display_text = text[:100] + "..." if len(text) > 100 else text
        print(f"Text: {display_text}")
        print("-"*50)
    
    # Prediction with color
    if result['is_ai']:
        label = f"{Fore.RED}🤖 AI GENERATED{Style.RESET_ALL}"
    else:
        label = f"{Fore.GREEN}👤 HUMAN WRITTEN{Style.RESET_ALL}"
    
    print(f"\nPrediction: {label}")
    print(f"Confidence: {result['confidence']}%")
    
    # Probability bars
    human_bar_len = int(result['human_percentage'] / 5)
    ai_bar_len = int(result['ai_percentage'] / 5)
    
    print(f"\n  Human: {Fore.GREEN}{'█' * human_bar_len}{'░' * (20 - human_bar_len)}{Style.RESET_ALL} {result['human_percentage']}%")
    print(f"  AI:    {Fore.RED}{'█' * ai_bar_len}{'░' * (20 - ai_bar_len)}{Style.RESET_ALL} {result['ai_percentage']}%")
    
    print("="*50)


def interactive_mode(detector: Detector):
    """Run interactive detection mode"""
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print("   AI TEXT DETECTOR - INTERACTIVE MODE")
    print(f"{'='*60}{Style.RESET_ALL}")
    print("\nEnter text to analyze (press Enter twice to submit)")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            print(f"{Fore.YELLOW}Enter text:{Style.RESET_ALL}")
            
            lines = []
            empty_count = 0
            
            while True:
                line = input()
                
                if line.lower() in ['quit', 'exit']:
                    print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                    return
                
                if line == '':
                    empty_count += 1
                    if empty_count >= 1 and lines:
                        break
                else:
                    empty_count = 0
                    lines.append(line)
            
            text = '\n'.join(lines).strip()
            
            if len(text) < 10:
                print(f"{Fore.RED}Text too short. Please enter at least 10 characters.{Style.RESET_ALL}\n")
                continue
            
            # Detect
            print(f"\n{Fore.CYAN}Analyzing...{Style.RESET_ALL}")
            result = detector.detect(text)
            
            # Print result
            print_result(result, text)
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")