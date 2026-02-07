"""
Unified pipeline for ManavAI.
Combines detection and humanization in a single interface.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from inference.detector_inference import DetectorInference
from inference.humanizer_inference import HumanizerInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManavAIPipeline:
    """
    Unified pipeline combining AI detection and humanization.
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to saved models directory
            config: Configuration object
        """
        self.config = config or Config()
        self.model_path = model_path or self.config.MODEL_SAVE_DIR
        
        self._detector = None
        self._humanizer = None
    
    @property
    def detector(self) -> DetectorInference:
        """Lazy load detector."""
        if self._detector is None:
            self._detector = DetectorInference(self.model_path, self.config)
        return self._detector
    
    @property
    def humanizer(self) -> HumanizerInference:
        """Lazy load humanizer."""
        if self._humanizer is None:
            self._humanizer = HumanizerInference(self.model_path, self.config)
        return self._humanizer
    
    def analyze(self, text: str) -> Dict:
        """
        Perform comprehensive analysis on text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Analysis results including detection and recommendations
        """
        # Get detection results
        detection = self.detector.detect(text, return_details=True)
        
        # Get per-sentence analysis
        sentence_analysis = self.detector.detect_sentences(text)
        
        # Calculate summary stats
        ai_sentences = sum(1 for s in sentence_analysis 
                          if s.get('ai_probability') and s['ai_probability'] > 0.5)
        total_sentences = sum(1 for s in sentence_analysis 
                             if s.get('ai_probability') is not None)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detection, sentence_analysis)
        
        return {
            'overall_detection': detection,
            'sentence_analysis': sentence_analysis,
            'summary': {
                'total_sentences': total_sentences,
                'ai_flagged_sentences': ai_sentences,
                'human_like_sentences': total_sentences - ai_sentences,
                'ai_sentence_ratio': ai_sentences / max(total_sentences, 1)
            },
            'recommendations': recommendations
        }
    
    def _generate_recommendations(
        self, 
        detection: Dict, 
        sentence_analysis: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        ai_prob = detection['ai_probability']
        
        if ai_prob > 0.8:
            recommendations.append(
                "Text is highly likely AI-generated. Consider aggressive humanization."
            )
        elif ai_prob > 0.5:
            recommendations.append(
                "Text shows some AI characteristics. Medium humanization recommended."
            )
        else:
            recommendations.append(
                "Text appears mostly human-written. Light touch-ups may help."
            )
        
        # Check for specific patterns
        high_ai_sentences = [s for s in sentence_analysis 
                           if s.get('ai_probability') and s['ai_probability'] > 0.7]
        
        if len(high_ai_sentences) > 0:
            recommendations.append(
                f"Found {len(high_ai_sentences)} sentences with high AI probability. "
                "Focus on rewriting these."
            )
        
        # Linguistic recommendations
        if detection.get('linguistic_analysis'):
            ling = detection['linguistic_analysis']
            
            if ling.get('sentence_length_variance', 0) < 20:
                recommendations.append(
                    "Sentence lengths are too uniform. Vary sentence structure more."
                )
            
            if ling.get('vocabulary_richness', 0) < 0.5:
                recommendations.append(
                    "Vocabulary could be more diverse. Use more varied word choices."
                )
        
        return recommendations
    
    def detect_and_humanize(
        self,
        text: str,
        style: str = 'casual',
        intensity: str = 'medium',
        target_score: float = 0.3,
        max_iterations: int = 3
    ) -> Dict:
        """
        Detect AI content and humanize until target score is reached.
        
        Args:
            text: Text to process
            style: Humanization style
            intensity: Humanization intensity
            target_score: Target AI detection score (lower is more human-like)
            max_iterations: Maximum humanization attempts
        
        Returns:
            Results including original analysis, humanized text, and final analysis
        """
        # Initial detection
        original_detection = self.detector.detect(text, return_details=True)
        
        result = {
            'original_text': text,
            'original_detection': original_detection,
            'iterations': [],
            'success': False
        }
        
        current_text = text
        current_score = original_detection['ai_probability']
        
        for i in range(max_iterations):
            if current_score <= target_score:
                result['success'] = True
                break
            
            # Humanize
            humanized = self.humanizer.humanize(
                current_text,
                style=style,
                intensity=intensity
            )
            
            # Check new score
            new_detection = self.detector.detect(humanized, return_details=True)
            new_score = new_detection['ai_probability']
            
            iteration_result = {
                'iteration': i + 1,
                'text': humanized,
                'score_before': current_score,
                'score_after': new_score,
                'improvement': current_score - new_score
            }
            result['iterations'].append(iteration_result)
            
            current_text = humanized
            current_score = new_score
            
            # Increase intensity if not improving
            if new_score >= current_score and intensity != 'aggressive':
                intensity = 'aggressive'
        
        result['final_text'] = current_text
        result['final_detection'] = self.detector.detect(current_text, return_details=True)
        result['total_improvement'] = (
            original_detection['ai_probability'] - 
            result['final_detection']['ai_probability']
        )
        
        if current_score <= target_score:
            result['success'] = True
        
        return result
    
    def batch_process(
        self,
        texts: List[str],
        operation: str = 'detect',
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple texts.
        
        Args:
            texts: List of texts to process
            operation: 'detect', 'humanize', or 'both'
            **kwargs: Additional arguments for operations
        
        Returns:
            List of results for each text
        """
        results = []
        
        for text in texts:
            if operation == 'detect':
                result = self.detector.detect(text, return_details=True)
            elif operation == 'humanize':
                humanized = self.humanizer.humanize(text, **kwargs)
                result = {
                    'original': text,
                    'humanized': humanized
                }
            else:  # both
                result = self.detect_and_humanize(text, **kwargs)
            
            results.append(result)
        
        return results
    
    def interactive_session(self):
        """Run an interactive session."""
        print("\n" + "="*60)
        print("ManavAI Interactive Session")
        print("="*60)
        print("\nCommands:")
        print("  detect <text>   - Detect AI content")
        print("  humanize <text> - Humanize text")
        print("  analyze <text>  - Full analysis")
        print("  auto <text>     - Auto detect and humanize")
        print("  quit            - Exit session")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("ManavAI> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                text = parts[1] if len(parts) > 1 else ""
                
                if not text and command in ['detect', 'humanize', 'analyze', 'auto']:
                    print("Please provide text after the command.")
                    continue
                
                if command == 'detect':
                    result = self.detector.detect(text, return_details=True)
                    print(f"\nAI Probability: {result['ai_probability']:.2%}")
                    print(f"Classification: {'AI-Generated' if result['is_ai_generated'] else 'Human-Written'}")
                    print(f"Confidence: {result['confidence']}\n")
                
                elif command == 'humanize':
                    style = input("Style (casual/formal/academic) [casual]: ").strip() or 'casual'
                    intensity = input("Intensity (light/medium/aggressive) [medium]: ").strip() or 'medium'
                    
                    humanized = self.humanizer.humanize(text, style=style, intensity=intensity)
                    print(f"\nHumanized Text:\n{humanized}\n")
                
                elif command == 'analyze':
                    result = self.analyze(text)
                    print(f"\nOverall AI Probability: {result['overall_detection']['ai_probability']:.2%}")
                    print(f"AI Flagged Sentences: {result['summary']['ai_flagged_sentences']}/{result['summary']['total_sentences']}")
                    print("\nRecommendations:")
                    for rec in result['recommendations']:
                        print(f"  - {rec}")
                    print()
                
                elif command == 'auto':
                    print("\nProcessing... (this may take a moment)")
                    result = self.detect_and_humanize(text)
                    print(f"\nOriginal AI Score: {result['original_detection']['ai_probability']:.2%}")
                    print(f"Final AI Score: {result['final_detection']['ai_probability']:.2%}")
                    print(f"Improvement: {result['total_improvement']:.2%}")
                    print(f"Success: {'Yes' if result['success'] else 'No'}")
                    print(f"\nFinal Text:\n{result['final_text']}\n")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Use: detect, humanize, analyze, auto, or quit")
                
            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Test the pipeline."""
    pipeline = ManavAIPipeline()
    pipeline.interactive_session()


if __name__ == '__main__':
    main()