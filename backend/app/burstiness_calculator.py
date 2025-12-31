"""
Burstiness Calculator for Text Analysis
Measures variation in sentence lengths to detect AI-generated text
AI text tends to have low burstiness (uniform sentences)
Human text tends to have high burstiness (varied sentences)
"""

import re
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BurstinessScore:
    """Data class to store burstiness metrics"""
    burstiness: float
    sentence_count: int
    avg_sentence_length: float
    std_dev: float
    coefficient_of_variation: float
    max_consecutive_diff: float
    avg_consecutive_diff: float
    sentence_lengths: List[int]
    interpretation: str


class BurstinessCalculator:
    """Calculate burstiness metrics for text analysis"""
    
    def __init__(self):
        """Initialize the burstiness calculator"""
        logger.info("BurstinessCalculator initialized")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        
        Returns:
            List of sentences
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries (., !, ?)
        # Handle common abbreviations and edge cases
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_sentence_lengths(self, text: str, unit: str = 'words') -> List[int]:
        """
        Get length of each sentence
        
        Args:
            text: Input text
            unit: 'words' or 'characters'
        
        Returns:
            List of sentence lengths
        """
        sentences = self.split_into_sentences(text)
        
        if unit == 'words':
            lengths = [len(sentence.split()) for sentence in sentences]
        elif unit == 'characters':
            lengths = [len(sentence) for sentence in sentences]
        else:
            raise ValueError("Unit must be 'words' or 'characters'")
        
        return lengths
    
    def calculate_burstiness(self, sentence_lengths: List[int]) -> float:
        """
        Calculate burstiness score using standard formula
        
        Burstiness = (σ - μ) / (σ + μ)
        where σ is standard deviation and μ is mean
        
        Score ranges from -1 to 1:
        - Close to 1: High burstiness (varied, human-like)
        - Close to 0: Medium burstiness
        - Close to -1: Low burstiness (uniform, AI-like)
        
        Args:
            sentence_lengths: List of sentence lengths
        
        Returns:
            Burstiness score (-1 to 1)
        """
        if len(sentence_lengths) < 2:
            logger.warning("Need at least 2 sentences to calculate burstiness")
            return 0.0
        
        mean = np.mean(sentence_lengths)
        std_dev = np.std(sentence_lengths)
        
        # Avoid division by zero
        if mean + std_dev == 0:
            return 0.0
        
        burstiness = (std_dev - mean) / (std_dev + mean)
        
        return round(burstiness, 4)
    
    def calculate_coefficient_of_variation(self, sentence_lengths: List[int]) -> float:
        """
        Calculate coefficient of variation (CV)
        CV = (std_dev / mean) * 100
        
        Higher CV indicates more variation (more human-like)
        
        Returns:
            Coefficient of variation as percentage
        """
        if len(sentence_lengths) < 2:
            return 0.0
        
        mean = np.mean(sentence_lengths)
        std_dev = np.std(sentence_lengths)
        
        if mean == 0:
            return 0.0
        
        cv = (std_dev / mean) * 100
        
        return round(cv, 2)
    
    def calculate_consecutive_differences(self, sentence_lengths: List[int]) -> Dict[str, float]:
        """
        Calculate differences between consecutive sentences
        
        Returns:
            Dictionary with max and average consecutive differences
        """
        if len(sentence_lengths) < 2:
            return {
                'max_diff': 0.0,
                'avg_diff': 0.0,
                'differences': []
            }
        
        # Calculate absolute differences between consecutive sentences
        differences = [abs(sentence_lengths[i+1] - sentence_lengths[i]) 
                      for i in range(len(sentence_lengths) - 1)]
        
        return {
            'max_diff': round(max(differences), 2),
            'avg_diff': round(np.mean(differences), 2),
            'differences': differences
        }
    
    def interpret_burstiness(self, burstiness_score: float) -> str:
        """
        Interpret burstiness score
        
        Args:
            burstiness_score: Score between -1 and 1
        
        Returns:
            Human-readable interpretation
        """
        if burstiness_score >= 0.5:
            return "Very High (Strong human characteristics)"
        elif burstiness_score >= 0.2:
            return "High (Likely human-written)"
        elif burstiness_score >= 0:
            return "Moderate (Mixed or edited content)"
        elif burstiness_score >= -0.2:
            return "Low (Possibly AI-generated)"
        else:
            return "Very Low (Likely AI-generated)"
    
    def analyze(self, text: str, unit: str = 'words') -> BurstinessScore:
        """
        Perform complete burstiness analysis
        
        Args:
            text: Input text to analyze
            unit: 'words' or 'characters'
        
        Returns:
            BurstinessScore object with all metrics
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return BurstinessScore(
                burstiness=0.0,
                sentence_count=0,
                avg_sentence_length=0.0,
                std_dev=0.0,
                coefficient_of_variation=0.0,
                max_consecutive_diff=0.0,
                avg_consecutive_diff=0.0,
                sentence_lengths=[],
                interpretation="N/A"
            )
        
        # Get sentence lengths
        sentence_lengths = self.get_sentence_lengths(text, unit)
        
        if len(sentence_lengths) < 2:
            logger.warning("Text has fewer than 2 sentences")
            return BurstinessScore(
                burstiness=0.0,
                sentence_count=len(sentence_lengths),
                avg_sentence_length=sentence_lengths[0] if sentence_lengths else 0.0,
                std_dev=0.0,
                coefficient_of_variation=0.0,
                max_consecutive_diff=0.0,
                avg_consecutive_diff=0.0,
                sentence_lengths=sentence_lengths,
                interpretation="Insufficient data (need 2+ sentences)"
            )
        
        # Calculate metrics
        burstiness = self.calculate_burstiness(sentence_lengths)
        cv = self.calculate_coefficient_of_variation(sentence_lengths)
        consecutive_diffs = self.calculate_consecutive_differences(sentence_lengths)
        
        # Create score object
        score = BurstinessScore(
            burstiness=burstiness,
            sentence_count=len(sentence_lengths),
            avg_sentence_length=round(np.mean(sentence_lengths), 2),
            std_dev=round(np.std(sentence_lengths), 2),
            coefficient_of_variation=cv,
            max_consecutive_diff=consecutive_diffs['max_diff'],
            avg_consecutive_diff=consecutive_diffs['avg_diff'],
            sentence_lengths=sentence_lengths,
            interpretation=self.interpret_burstiness(burstiness)
        )
        
        return score
    
    def compare_texts(
        self,
        text1: str,
        text2: str,
        label1: str = "Text 1",
        label2: str = "Text 2"
    ) -> Dict[str, any]:
        """
        Compare burstiness of two texts
        
        Returns:
            Dictionary with comparison results
        """
        score1 = self.analyze(text1)
        score2 = self.analyze(text2)
        
        return {
            label1: {
                'burstiness': score1.burstiness,
                'interpretation': score1.interpretation,
                'avg_sentence_length': score1.avg_sentence_length,
                'std_dev': score1.std_dev,
                'cv': score1.coefficient_of_variation
            },
            label2: {
                'burstiness': score2.burstiness,
                'interpretation': score2.interpretation,
                'avg_sentence_length': score2.avg_sentence_length,
                'std_dev': score2.std_dev,
                'cv': score2.coefficient_of_variation
            },
            'difference': {
                'burstiness_diff': abs(score1.burstiness - score2.burstiness),
                'more_human_like': label1 if score1.burstiness > score2.burstiness else label2
            }
        }
    
    def visualize_sentence_lengths(
        self,
        text: str,
        title: str = "Sentence Length Variation",
        save_path: str = None
    ) -> None:
        """
        Create visualization of sentence length variation
        
        Args:
            text: Input text
            title: Plot title
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            score = self.analyze(text)
            sentence_lengths = score.sentence_lengths
            
            if len(sentence_lengths) < 2:
                logger.warning("Not enough sentences to visualize")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Line graph of sentence lengths
            ax1.plot(range(1, len(sentence_lengths) + 1), sentence_lengths, 
                    marker='o', linewidth=2, markersize=6, color='#2E86AB')
            ax1.axhline(y=score.avg_sentence_length, color='red', 
                       linestyle='--', label=f'Average: {score.avg_sentence_length:.1f}')
            ax1.fill_between(range(1, len(sentence_lengths) + 1),
                           score.avg_sentence_length - score.std_dev,
                           score.avg_sentence_length + score.std_dev,
                           alpha=0.2, color='red', label='±1 Std Dev')
            ax1.set_xlabel('Sentence Number', fontsize=12)
            ax1.set_ylabel('Sentence Length (words)', fontsize=12)
            ax1.set_title(f'{title}\nBurstiness: {score.burstiness:.3f} ({score.interpretation})', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Bar chart of sentence lengths
            colors = ['#A23B72' if length > score.avg_sentence_length else '#2E86AB' 
                     for length in sentence_lengths]
            ax2.bar(range(1, len(sentence_lengths) + 1), sentence_lengths, color=colors)
            ax2.axhline(y=score.avg_sentence_length, color='red', 
                       linestyle='--', linewidth=2)
            ax2.set_xlabel('Sentence Number', fontsize=12)
            ax2.set_ylabel('Sentence Length (words)', fontsize=12)
            ax2.set_title('Bar Chart View', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not installed. Visualization skipped.")
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
    
    def print_analysis(self, score: BurstinessScore, text_label: str = "Text") -> None:
        """Print formatted burstiness analysis"""
        print("\n" + "="*60)
        print(f"BURSTINESS ANALYSIS: {text_label}")
        print("="*60)
        print(f"\n📊 BURSTINESS METRICS")
        print(f"  Burstiness Score:         {score.burstiness:.4f}")
        print(f"  Interpretation:           {score.interpretation}")
        print(f"  Coefficient of Variation: {score.coefficient_of_variation:.2f}%")
        
        print(f"\n📝 SENTENCE STATISTICS")
        print(f"  Sentence Count:           {score.sentence_count}")
        print(f"  Average Length:           {score.avg_sentence_length:.2f} words")
        print(f"  Standard Deviation:       {score.std_dev:.2f} words")
        print(f"  Min Length:               {min(score.sentence_lengths) if score.sentence_lengths else 0}")
        print(f"  Max Length:               {max(score.sentence_lengths) if score.sentence_lengths else 0}")
        
        print(f"\n🔄 CONSECUTIVE VARIATION")
        print(f"  Max Consecutive Diff:     {score.max_consecutive_diff:.2f} words")
        print(f"  Avg Consecutive Diff:     {score.avg_consecutive_diff:.2f} words")
        
        print("="*60 + "\n")
    
    def get_sentence_length_distribution(self, text: str) -> Dict[str, int]:
        """
        Get distribution of sentence lengths in ranges
        
        Returns:
            Dictionary with length ranges and counts
        """
        sentence_lengths = self.get_sentence_lengths(text)
        
        distribution = {
            'very_short (1-5)': 0,
            'short (6-10)': 0,
            'medium (11-20)': 0,
            'long (21-30)': 0,
            'very_long (31+)': 0
        }
        
        for length in sentence_lengths:
            if length <= 5:
                distribution['very_short (1-5)'] += 1
            elif length <= 10:
                distribution['short (6-10)'] += 1
            elif length <= 20:
                distribution['medium (11-20)'] += 1
            elif length <= 30:
                distribution['long (21-30)'] += 1
            else:
                distribution['very_long (31+)'] += 1
        
        return distribution


def main():
    """Example usage and testing"""
    
    # Initialize calculator
    calculator = BurstinessCalculator()
    
    # Example texts
    ai_text = """
    Machine learning is transforming industries worldwide. Companies are 
    adopting artificial intelligence to improve efficiency. Data analysis 
    helps organizations make better decisions. Technology continues to 
    advance rapidly. Automation is becoming increasingly important.
    """
    
    human_text = """
    I went to the store yesterday. Got some milk and bread, but they were 
    out of my favorite cereal which was really annoying! So I had to get 
    a different brand. Have you ever noticed how some stores just never 
    have what you need when you need it? Anyway, I ended up spending way 
    more than I planned because everything was on sale and I couldn't 
    resist. Classic me!
    """
    
    # Analyze AI text
    print("Analyzing AI-generated text...")
    ai_score = calculator.analyze(ai_text)
    calculator.print_analysis(ai_score, "AI Text")
    
    # Analyze human text
    print("Analyzing Human-written text...")
    human_score = calculator.analyze(human_text)
    calculator.print_analysis(human_score, "Human Text")
    
    # Compare texts
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    comparison = calculator.compare_texts(ai_text, human_text, "AI Text", "Human Text")
    print(f"\nAI Text Burstiness:    {comparison['AI Text']['burstiness']:.4f}")
    print(f"Human Text Burstiness: {comparison['Human Text']['burstiness']:.4f}")
    print(f"Difference:            {comparison['difference']['burstiness_diff']:.4f}")
    print(f"More human-like:       {comparison['difference']['more_human_like']}")
    print("="*60 + "\n")
    
    # Show sentence length distribution
    print("Sentence Length Distribution (Human Text):")
    distribution = calculator.get_sentence_length_distribution(human_text)
    for range_name, count in distribution.items():
        print(f"  {range_name}: {count}")
    
    # Optional: Create visualization (requires matplotlib)
    # calculator.visualize_sentence_lengths(human_text, "Human Text Analysis")


if __name__ == "__main__":
    main()