"""
Text Complexity Metrics Calculator
Analyzes readability, grammar, and sentence structure of text
"""

import re
import numpy as np
from typing import Dict, List, Tuple
import language_tool_python
from textstat import flesch_reading_ease, flesch_kincaid_grade
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComplexityScore:
    """Data class to store all complexity metrics"""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    grammar_errors: int
    grammar_error_rate: float
    sentence_count: int
    avg_sentence_length: float
    sentence_length_variance: float
    sentence_length_std: float
    readability_interpretation: str
    grade_interpretation: str
    grammar_quality: str


class TextComplexityAnalyzer:
    """Analyze text complexity using multiple metrics"""
    
    def __init__(self, language: str = 'en-US', use_grammar_check: bool = True):
        """
        Initialize the complexity analyzer
        
        Args:
            language: Language code for grammar checking (default: 'en-US')
            use_grammar_check: Whether to enable grammar checking (can be slow)
        """
        self.language = language
        self.use_grammar_check = use_grammar_check
        self.grammar_tool = None
        
        if self.use_grammar_check:
            try:
                logger.info(f"Initializing grammar checker for {language}...")
                self.grammar_tool = language_tool_python.LanguageTool(language)
                logger.info("Grammar checker initialized successfully!")
            except Exception as e:
                logger.warning(f"Failed to initialize grammar checker: {str(e)}")
                logger.warning("Grammar checking will be disabled")
                self.use_grammar_check = False
    
    def calculate_flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score
        
        Score interpretation:
        90-100: Very Easy (5th grade)
        80-89: Easy (6th grade)
        70-79: Fairly Easy (7th grade)
        60-69: Standard (8th-9th grade)
        50-59: Fairly Difficult (10th-12th grade)
        30-49: Difficult (College)
        0-29: Very Difficult (College graduate)
        
        Returns:
            Flesch Reading Ease score (0-100+)
        """
        try:
            score = flesch_reading_ease(text)
            return round(score, 2)
        except Exception as e:
            logger.error(f"Error calculating Flesch Reading Ease: {str(e)}")
            return 0.0
    
    def calculate_flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level
        
        Returns the US school grade level needed to understand the text
        (e.g., 8.0 = 8th grade level)
        
        Returns:
            Grade level (typically 0-18+)
        """
        try:
            grade = flesch_kincaid_grade(text)
            return round(grade, 2)
        except Exception as e:
            logger.error(f"Error calculating Flesch-Kincaid Grade: {str(e)}")
            return 0.0
    
    def check_grammar(self, text: str) -> Tuple[int, List[Dict]]:
        """
        Check grammar using LanguageTool
        
        Returns:
            Tuple of (error_count, list_of_errors)
        """
        if not self.use_grammar_check or self.grammar_tool is None:
            return 0, []
        
        try:
            matches = self.grammar_tool.check(text)
            
            errors = []
            for match in matches:
                errors.append({
                    'message': match.message,
                    'context': match.context,
                    'offset': match.offset,
                    'error_length': match.errorLength,
                    'category': match.category,
                    'rule_id': match.ruleId,
                    'suggestions': match.replacements[:3]  # Top 3 suggestions
                })
            
            return len(matches), errors
            
        except Exception as e:
            logger.error(f"Error checking grammar: {str(e)}")
            return 0, []
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        
        Returns:
            List of sentences
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries (., !, ?)
        # This is a simplified approach; for production, consider using nltk or spacy
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def calculate_sentence_length_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate sentence length statistics
        
        Returns:
            Dictionary with avg_length, variance, and std_dev
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return {
                'sentence_count': 0,
                'avg_sentence_length': 0.0,
                'variance': 0.0,
                'std_dev': 0.0,
                'min_length': 0,
                'max_length': 0
            }
        
        # Count words in each sentence
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': round(np.mean(sentence_lengths), 2),
            'variance': round(np.var(sentence_lengths), 2),
            'std_dev': round(np.std(sentence_lengths), 2),
            'min_length': min(sentence_lengths),
            'max_length': max(sentence_lengths)
        }
    
    def interpret_flesch_reading_ease(self, score: float) -> str:
        """Interpret Flesch Reading Ease score"""
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8th-9th grade)"
        elif score >= 50:
            return "Fairly Difficult (10th-12th grade)"
        elif score >= 30:
            return "Difficult (College level)"
        else:
            return "Very Difficult (Graduate level)"
    
    def interpret_grade_level(self, grade: float) -> str:
        """Interpret Flesch-Kincaid Grade Level"""
        if grade <= 5:
            return "Elementary school level"
        elif grade <= 8:
            return "Middle school level"
        elif grade <= 12:
            return "High school level"
        elif grade <= 16:
            return "College level"
        else:
            return "Graduate/Professional level"
    
    def interpret_grammar_quality(self, error_rate: float) -> str:
        """Interpret grammar error rate"""
        if error_rate == 0:
            return "Excellent (no errors)"
        elif error_rate < 1:
            return "Very Good (minimal errors)"
        elif error_rate < 3:
            return "Good (few errors)"
        elif error_rate < 5:
            return "Fair (some errors)"
        elif error_rate < 10:
            return "Poor (many errors)"
        else:
            return "Very Poor (excessive errors)"
    
    def analyze(self, text: str, include_grammar_details: bool = False) -> ComplexityScore:
        """
        Perform complete complexity analysis on text
        
        Args:
            text: Input text to analyze
            include_grammar_details: Return detailed grammar error list
        
        Returns:
            ComplexityScore object with all metrics
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return ComplexityScore(
                flesch_reading_ease=0.0,
                flesch_kincaid_grade=0.0,
                grammar_errors=0,
                grammar_error_rate=0.0,
                sentence_count=0,
                avg_sentence_length=0.0,
                sentence_length_variance=0.0,
                sentence_length_std=0.0,
                readability_interpretation="N/A",
                grade_interpretation="N/A",
                grammar_quality="N/A"
            )
        
        # Calculate readability scores
        flesch_ease = self.calculate_flesch_reading_ease(text)
        flesch_grade = self.calculate_flesch_kincaid_grade(text)
        
        # Check grammar
        grammar_error_count, grammar_errors = self.check_grammar(text)
        
        # Calculate sentence metrics
        sentence_metrics = self.calculate_sentence_length_metrics(text)
        
        # Calculate error rate (errors per 100 words)
        word_count = len(text.split())
        grammar_error_rate = round((grammar_error_count / word_count * 100), 2) if word_count > 0 else 0.0
        
        # Create score object
        score = ComplexityScore(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=flesch_grade,
            grammar_errors=grammar_error_count,
            grammar_error_rate=grammar_error_rate,
            sentence_count=sentence_metrics['sentence_count'],
            avg_sentence_length=sentence_metrics['avg_sentence_length'],
            sentence_length_variance=sentence_metrics['variance'],
            sentence_length_std=sentence_metrics['std_dev'],
            readability_interpretation=self.interpret_flesch_reading_ease(flesch_ease),
            grade_interpretation=self.interpret_grade_level(flesch_grade),
            grammar_quality=self.interpret_grammar_quality(grammar_error_rate)
        )
        
        if include_grammar_details:
            # Add grammar error details to the score object (as dict for easier handling)
            score_dict = score.__dict__.copy()
            score_dict['grammar_error_details'] = grammar_errors
            return score_dict
        
        return score
    
    def compare_texts(
        self,
        text1: str,
        text2: str,
        label1: str = "Text 1",
        label2: str = "Text 2"
    ) -> Dict[str, any]:
        """
        Compare complexity metrics of two texts
        
        Returns:
            Dictionary with comparison results
        """
        score1 = self.analyze(text1)
        score2 = self.analyze(text2)
        
        return {
            label1: {
                'flesch_reading_ease': score1.flesch_reading_ease,
                'flesch_kincaid_grade': score1.flesch_kincaid_grade,
                'grammar_errors': score1.grammar_errors,
                'avg_sentence_length': score1.avg_sentence_length,
                'sentence_length_variance': score1.sentence_length_variance
            },
            label2: {
                'flesch_reading_ease': score2.flesch_reading_ease,
                'flesch_kincaid_grade': score2.flesch_kincaid_grade,
                'grammar_errors': score2.grammar_errors,
                'avg_sentence_length': score2.avg_sentence_length,
                'sentence_length_variance': score2.sentence_length_variance
            },
            'differences': {
                'flesch_ease_diff': abs(score1.flesch_reading_ease - score2.flesch_reading_ease),
                'grade_diff': abs(score1.flesch_kincaid_grade - score2.flesch_kincaid_grade),
                'grammar_diff': abs(score1.grammar_errors - score2.grammar_errors)
            }
        }
    
    def print_analysis(self, score: ComplexityScore, text_label: str = "Text") -> None:
        """Print formatted analysis results"""
        print("\n" + "="*60)
        print(f"COMPLEXITY ANALYSIS: {text_label}")
        print("="*60)
        print(f"\n📊 READABILITY METRICS")
        print(f"  Flesch Reading Ease:  {score.flesch_reading_ease:.2f} - {score.readability_interpretation}")
        print(f"  Flesch-Kincaid Grade: {score.flesch_kincaid_grade:.2f} - {score.grade_interpretation}")
        
        print(f"\n✍️  GRAMMAR METRICS")
        print(f"  Grammar Errors:       {score.grammar_errors}")
        print(f"  Error Rate:           {score.grammar_error_rate:.2f} per 100 words")
        print(f"  Grammar Quality:      {score.grammar_quality}")
        
        print(f"\n📝 SENTENCE METRICS")
        print(f"  Sentence Count:       {score.sentence_count}")
        print(f"  Avg Sentence Length:  {score.avg_sentence_length:.2f} words")
        print(f"  Length Variance:      {score.sentence_length_variance:.2f}")
        print(f"  Length Std Dev:       {score.sentence_length_std:.2f}")
        
        print("="*60 + "\n")
    
    def __del__(self):
        """Cleanup grammar tool on deletion"""
        if self.grammar_tool is not None:
            self.grammar_tool.close()


def main():
    """Example usage and testing"""
    
    # Initialize analyzer
    analyzer = TextComplexityAnalyzer(use_grammar_check=True)
    
    # Example texts
    ai_text = """
    Artificial intelligence is revolutionizing the way we approach 
    problem-solving in the modern era. Machine learning algorithms 
    are becoming increasingly sophisticated, enabling computers to 
    perform tasks that were once thought to be exclusively human. 
    This technological advancement has significant implications for 
    various industries and sectors.
    """
    
    human_text = """
    So I was thinking about getting a new laptop, but honestly I'm 
    not sure if I really need one right now. My old one still works 
    fine, even though it's kinda slow sometimes. Maybe I should just 
    wait until it completely dies? What do you think? I mean, they're 
    so expensive these days!
    """
    
    # Analyze AI text
    print("Analyzing AI-generated text...")
    ai_score = analyzer.analyze(ai_text)
    analyzer.print_analysis(ai_score, "AI Text")
    
    # Analyze human text
    print("Analyzing Human-written text...")
    human_score = analyzer.analyze(human_text)
    analyzer.print_analysis(human_score, "Human Text")
    
    # Compare texts
    print("Comparison Summary:")
    comparison = analyzer.compare_texts(ai_text, human_text, "AI Text", "Human Text")
    print(f"\nFlesch Reading Ease Difference: {comparison['differences']['flesch_ease_diff']:.2f}")
    print(f"Grade Level Difference: {comparison['differences']['grade_diff']:.2f}")
    print(f"Grammar Error Difference: {comparison['differences']['grammar_diff']}")


if __name__ == "__main__":
    main()