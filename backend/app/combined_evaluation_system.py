"""
Combined Evaluation System
Integrates perplexity, complexity, and burstiness metrics
Provides comprehensive text analysis and before/after comparison
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import language_tool_python
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationScore:
    """Comprehensive evaluation score containing all metrics"""
    
    # Perplexity metrics
    perplexity: float
    perplexity_interpretation: str
    
    # Readability metrics
    flesch_reading_ease: float
    flesch_reading_ease_interpretation: str
    flesch_kincaid_grade: float
    flesch_kincaid_grade_interpretation: str
    
    # Grammar metrics
    grammar_errors: int
    grammar_error_rate: float
    grammar_quality: str
    
    # Sentence structure metrics
    sentence_count: int
    avg_sentence_length: float
    sentence_length_variance: float
    sentence_length_std: float
    
    # Burstiness metrics
    burstiness: float
    burstiness_interpretation: str
    coefficient_of_variation: float
    max_consecutive_diff: float
    avg_consecutive_diff: float
    
    # Overall assessment
    overall_human_score: float  # 0-100 score
    likely_source: str  # "AI-generated" or "Human-written"
    confidence: str  # "High", "Medium", "Low"
    
    # Metadata
    text_length: int  # word count
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class ComparisonResult:
    """Results of before/after comparison"""
    
    before_score: EvaluationScore
    after_score: EvaluationScore
    
    improvements: Dict[str, float]
    degradations: Dict[str, float]
    
    overall_improvement: float  # percentage
    humanization_success: bool
    summary: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'before_score': self.before_score.to_dict(),
            'after_score': self.after_score.to_dict(),
            'improvements': self.improvements,
            'degradations': self.degradations,
            'overall_improvement': self.overall_improvement,
            'humanization_success': self.humanization_success,
            'summary': self.summary
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class CombinedEvaluator:
    """Combined evaluation system integrating all metrics"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        use_grammar_check: bool = True,
        language: str = 'en-US'
    ):
        """
        Initialize the combined evaluator
        
        Args:
            model_name: GPT-2 model variant for perplexity
            device: Device for model ("cuda", "cpu", or None for auto)
            use_grammar_check: Enable grammar checking
            language: Language code for grammar checking
        """
        logger.info("Initializing Combined Evaluator...")
        
        self.model_name = model_name
        self.use_grammar_check = use_grammar_check
        self.language = language
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load models and tools
        self._load_perplexity_model()
        self._load_grammar_tool()
        
        logger.info("Combined Evaluator initialized successfully!")
    
    def _load_perplexity_model(self) -> None:
        """Load GPT-2 for perplexity calculation"""
        logger.info(f"Loading perplexity model: {self.model_name}")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_grammar_tool(self) -> None:
        """Load grammar checking tool"""
        if self.use_grammar_check:
            try:
                logger.info("Loading grammar checker...")
                self.grammar_tool = language_tool_python.LanguageTool(self.language)
            except Exception as e:
                logger.warning(f"Failed to load grammar checker: {str(e)}")
                self.use_grammar_check = False
                self.grammar_tool = None
        else:
            self.grammar_tool = None
    
    # ==================== PERPLEXITY METHODS ====================
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity score"""
        try:
            encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = encodings.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return round(perplexity, 2)
        except Exception as e:
            logger.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')
    
    def _interpret_perplexity(self, perplexity: float) -> str:
        """Interpret perplexity score"""
        if perplexity < 50:
            return "Very low (likely AI-generated)"
        elif perplexity < 100:
            return "Low (possibly AI-generated)"
        elif perplexity < 200:
            return "Moderate (mixed or edited)"
        elif perplexity < 500:
            return "High (likely human-written)"
        else:
            return "Very high (likely human-written)"
    
    # ==================== COMPLEXITY METHODS ====================
    
    def _calculate_flesch_scores(self, text: str) -> tuple:
        """Calculate Flesch readability scores"""
        try:
            ease = round(flesch_reading_ease(text), 2)
            grade = round(flesch_kincaid_grade(text), 2)
            return ease, grade
        except Exception as e:
            logger.error(f"Error calculating Flesch scores: {str(e)}")
            return 0.0, 0.0
    
    def _interpret_flesch_ease(self, score: float) -> str:
        """Interpret Flesch Reading Ease"""
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
    
    def _interpret_grade_level(self, grade: float) -> str:
        """Interpret Flesch-Kincaid Grade"""
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
    
    def _check_grammar(self, text: str) -> tuple:
        """Check grammar and return error count and rate"""
        if not self.use_grammar_check or self.grammar_tool is None:
            return 0, 0.0
        
        try:
            matches = self.grammar_tool.check(text)
            word_count = len(text.split())
            error_rate = round((len(matches) / word_count * 100), 2) if word_count > 0 else 0.0
            return len(matches), error_rate
        except Exception as e:
            logger.error(f"Error checking grammar: {str(e)}")
            return 0, 0.0
    
    def _interpret_grammar(self, error_rate: float) -> str:
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
    
    # ==================== SENTENCE ANALYSIS METHODS ====================
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        text = re.sub(r'\s+', ' ', text.strip())
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_sentence_metrics(self, text: str) -> tuple:
        """Calculate sentence-related metrics"""
        sentences = self._split_sentences(text)
        
        if not sentences:
            return 0, 0.0, 0.0, 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        return (
            len(sentences),
            round(np.mean(sentence_lengths), 2),
            round(np.var(sentence_lengths), 2),
            round(np.std(sentence_lengths), 2)
        )
    
    # ==================== BURSTINESS METHODS ====================
    
    def _calculate_burstiness(self, text: str) -> tuple:
        """Calculate burstiness metrics"""
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        # Burstiness score
        mean = np.mean(sentence_lengths)
        std_dev = np.std(sentence_lengths)
        burstiness = (std_dev - mean) / (std_dev + mean) if (std_dev + mean) > 0 else 0.0
        
        # Coefficient of variation
        cv = (std_dev / mean * 100) if mean > 0 else 0.0
        
        # Consecutive differences
        differences = [abs(sentence_lengths[i+1] - sentence_lengths[i]) 
                      for i in range(len(sentence_lengths) - 1)]
        max_diff = max(differences) if differences else 0.0
        avg_diff = np.mean(differences) if differences else 0.0
        
        return (
            round(burstiness, 4),
            round(cv, 2),
            round(max_diff, 2),
            round(avg_diff, 2)
        )
    
    def _interpret_burstiness(self, burstiness: float) -> str:
        """Interpret burstiness score"""
        if burstiness >= 0.5:
            return "Very High (Strong human characteristics)"
        elif burstiness >= 0.2:
            return "High (Likely human-written)"
        elif burstiness >= 0:
            return "Moderate (Mixed or edited content)"
        elif burstiness >= -0.2:
            return "Low (Possibly AI-generated)"
        else:
            return "Very Low (Likely AI-generated)"
    
    # ==================== OVERALL ASSESSMENT ====================
    
    def _calculate_human_score(
        self,
        perplexity: float,
        burstiness: float,
        grammar_error_rate: float,
        cv: float
    ) -> float:
        """
        Calculate overall human-likeness score (0-100)
        
        Weighted combination of metrics:
        - High perplexity = more human
        - High burstiness = more human
        - Low grammar errors = more polished (could be AI or human)
        - High CV = more variation (more human)
        """
        # Normalize perplexity (0-500 range to 0-100)
        perplexity_score = min(perplexity / 5, 100)
        
        # Normalize burstiness (-1 to 1 range to 0-100)
        burstiness_score = (burstiness + 1) * 50
        
        # Normalize grammar (inverse - fewer errors is good but too perfect might be AI)
        grammar_score = max(0, 100 - grammar_error_rate * 10)
        
        # Normalize CV
        cv_score = min(cv * 2, 100)
        
        # Weighted average (adjust weights based on importance)
        weights = {
            'perplexity': 0.35,
            'burstiness': 0.35,
            'grammar': 0.15,
            'cv': 0.15
        }
        
        overall = (
            perplexity_score * weights['perplexity'] +
            burstiness_score * weights['burstiness'] +
            grammar_score * weights['grammar'] +
            cv_score * weights['cv']
        )
        
        return round(overall, 2)
    
    def _determine_source(self, human_score: float) -> tuple:
        """
        Determine likely source and confidence
        
        Returns:
            Tuple of (source, confidence)
        """
        if human_score >= 70:
            return "Human-written", "High"
        elif human_score >= 55:
            return "Human-written", "Medium"
        elif human_score >= 45:
            return "Mixed/Unclear", "Low"
        elif human_score >= 30:
            return "AI-generated", "Medium"
        else:
            return "AI-generated", "High"
    
    # ==================== MAIN EVALUATION METHOD ====================
    
    def evaluate(self, text: str) -> EvaluationScore:
        """
        Perform comprehensive evaluation of text
        
        Args:
            text: Input text to evaluate
        
        Returns:
            EvaluationScore object with all metrics
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            raise ValueError("Text cannot be empty")
        
        logger.info("Starting comprehensive evaluation...")
        
        # Calculate all metrics
        perplexity = self._calculate_perplexity(text)
        perplexity_interp = self._interpret_perplexity(perplexity)
        
        flesch_ease, flesch_grade = self._calculate_flesch_scores(text)
        flesch_ease_interp = self._interpret_flesch_ease(flesch_ease)
        flesch_grade_interp = self._interpret_grade_level(flesch_grade)
        
        grammar_errors, grammar_error_rate = self._check_grammar(text)
        grammar_quality = self._interpret_grammar(grammar_error_rate)
        
        sentence_count, avg_len, variance, std_dev = self._calculate_sentence_metrics(text)
        
        burstiness, cv, max_diff, avg_diff = self._calculate_burstiness(text)
        burstiness_interp = self._interpret_burstiness(burstiness)
        
        # Calculate overall human score
        human_score = self._calculate_human_score(perplexity, burstiness, grammar_error_rate, cv)
        likely_source, confidence = self._determine_source(human_score)
        
        # Get text length
        word_count = len(text.split())
        
        # Create score object
        score = EvaluationScore(
            perplexity=perplexity,
            perplexity_interpretation=perplexity_interp,
            flesch_reading_ease=flesch_ease,
            flesch_reading_ease_interpretation=flesch_ease_interp,
            flesch_kincaid_grade=flesch_grade,
            flesch_kincaid_grade_interpretation=flesch_grade_interp,
            grammar_errors=grammar_errors,
            grammar_error_rate=grammar_error_rate,
            grammar_quality=grammar_quality,
            sentence_count=sentence_count,
            avg_sentence_length=avg_len,
            sentence_length_variance=variance,
            sentence_length_std=std_dev,
            burstiness=burstiness,
            burstiness_interpretation=burstiness_interp,
            coefficient_of_variation=cv,
            max_consecutive_diff=max_diff,
            avg_consecutive_diff=avg_diff,
            overall_human_score=human_score,
            likely_source=likely_source,
            confidence=confidence,
            text_length=word_count,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("Evaluation completed successfully!")
        return score
    
    # ==================== COMPARISON METHOD ====================
    
    def compare(
        self,
        before_text: str,
        after_text: str,
        before_label: str = "Original",
        after_label: str = "Humanized"
    ) -> ComparisonResult:
        """
        Compare before and after texts
        
        Args:
            before_text: Original text
            after_text: Humanized text
            before_label: Label for original text
            after_label: Label for humanized text
        
        Returns:
            ComparisonResult object
        """
        logger.info("Starting before/after comparison...")
        
        # Evaluate both texts
        before_score = self.evaluate(before_text)
        after_score = self.evaluate(after_text)
        
        # Calculate improvements and degradations
        improvements = {}
        degradations = {}
        
        metrics_to_compare = {
            'perplexity': (before_score.perplexity, after_score.perplexity, 'higher_is_better'),
            'burstiness': (before_score.burstiness, after_score.burstiness, 'higher_is_better'),
            'flesch_reading_ease': (before_score.flesch_reading_ease, after_score.flesch_reading_ease, 'higher_is_better'),
            'grammar_errors': (before_score.grammar_errors, after_score.grammar_errors, 'lower_is_better'),
            'coefficient_of_variation': (before_score.coefficient_of_variation, after_score.coefficient_of_variation, 'higher_is_better'),
            'overall_human_score': (before_score.overall_human_score, after_score.overall_human_score, 'higher_is_better')
        }
        
        for metric_name, (before_val, after_val, direction) in metrics_to_compare.items():
            if before_val == 0:
                continue
            
            change = after_val - before_val
            percent_change = (change / abs(before_val)) * 100 if before_val != 0 else 0
            
            if direction == 'higher_is_better':
                if change > 0:
                    improvements[metric_name] = round(percent_change, 2)
                elif change < 0:
                    degradations[metric_name] = round(abs(percent_change), 2)
            else:  # lower_is_better
                if change < 0:
                    improvements[metric_name] = round(abs(percent_change), 2)
                elif change > 0:
                    degradations[metric_name] = round(percent_change, 2)
        
        # Calculate overall improvement
        overall_improvement = after_score.overall_human_score - before_score.overall_human_score
        humanization_success = overall_improvement > 5  # At least 5 point improvement
        
        # Generate summary
        summary = self._generate_comparison_summary(
            before_score,
            after_score,
            improvements,
            degradations,
            overall_improvement,
            humanization_success
        )
        
        result = ComparisonResult(
            before_score=before_score,
            after_score=after_score,
            improvements=improvements,
            degradations=degradations,
            overall_improvement=round(overall_improvement, 2),
            humanization_success=humanization_success,
            summary=summary
        )
        
        logger.info("Comparison completed successfully!")
        return result
    
    def _generate_comparison_summary(
        self,
        before: EvaluationScore,
        after: EvaluationScore,
        improvements: Dict,
        degradations: Dict,
        overall_improvement: float,
        success: bool
    ) -> str:
        """Generate human-readable comparison summary"""
        summary_parts = []
        
        # Overall assessment
        if success:
            summary_parts.append(f"✅ Humanization successful! Overall human score improved by {overall_improvement:.1f} points.")
        else:
            summary_parts.append(f"⚠️ Humanization had limited impact. Overall change: {overall_improvement:.1f} points.")
        
        # Key improvements
        if improvements:
            top_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts.append(f"Main improvements: {', '.join([f'{k} (+{v:.1f}%)' for k, v in top_improvements])}")
        
        # Key degradations
        if degradations:
            top_degradations = sorted(degradations.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts.append(f"Areas of concern: {', '.join([f'{k} (-{v:.1f}%)' for k, v in top_degradations])}")
        
        # Source classification change
        if before.likely_source != after.likely_source:
            summary_parts.append(f"Classification changed: {before.likely_source} → {after.likely_source}")
        
        return " ".join(summary_parts)
    
    def print_evaluation(self, score: EvaluationScore, label: str = "Text") -> None:
        """Print formatted evaluation results"""
        print("\n" + "="*70)
        print(f"COMPREHENSIVE EVALUATION: {label}")
        print("="*70)
        
        print(f"\n🎯 OVERALL ASSESSMENT")
        print(f"  Human-likeness Score:  {score.overall_human_score:.2f}/100")
        print(f"  Likely Source:         {score.likely_source}")
        print(f"  Confidence:            {score.confidence}")
        print(f"  Text Length:           {score.text_length} words")
        
        print(f"\n📊 PERPLEXITY")
        print(f"  Score:                 {score.perplexity:.2f}")
        print(f"  Interpretation:        {score.perplexity_interpretation}")
        
        print(f"\n📖 READABILITY")
        print(f"  Flesch Reading Ease:   {score.flesch_reading_ease:.2f} - {score.flesch_reading_ease_interpretation}")
        print(f"  Flesch-Kincaid Grade:  {score.flesch_kincaid_grade:.2f} - {score.flesch_kincaid_grade_interpretation}")
        
        print(f"\n✍️  GRAMMAR")
        print(f"  Errors:                {score.grammar_errors}")
        print(f"  Error Rate:            {score.grammar_error_rate:.2f} per 100 words")
        print(f"  Quality:               {score.grammar_quality}")
        
        print(f"\n📝 SENTENCE STRUCTURE")
        print(f"  Sentence Count:        {score.sentence_count}")
        print(f"  Avg Length:            {score.avg_sentence_length:.2f} words")
        print(f"  Variance:              {score.sentence_length_variance:.2f}")
        print(f"  Std Deviation:         {score.sentence_length_std:.2f}")
        
        print(f"\n🔄 BURSTINESS")
        print(f"  Burstiness Score:      {score.burstiness:.4f}")
        print(f"  Interpretation:        {score.burstiness_interpretation}")
        print(f"  Coefficient Variation: {score.coefficient_of_variation:.2f}%")
        print(f"  Max Consecutive Diff:  {score.max_consecutive_diff:.2f}")
        print(f"  Avg Consecutive Diff:  {score.avg_consecutive_diff:.2f}")
        
        print("="*70 + "\n")
    
    def print_comparison(self, result: ComparisonResult) -> None:
        """Print formatted comparison results"""
        print("\n" + "="*70)
        print("BEFORE/AFTER COMPARISON")
        print("="*70)
        
        print(f"\n{result.summary}")
        
        print(f"\n📊 SCORE COMPARISON")
        print(f"  Before: {result.before_score.overall_human_score:.2f}/100 ({result.before_score.likely_source})")
        print(f"  After:  {result.after_score.overall_human_score:.2f}/100 ({result.after_score.likely_source})")
        print(f"  Change: {result.overall_improvement:+.2f} points")
        
        if result.improvements:
            print(f"\n✅ IMPROVEMENTS")
            for metric, improvement in sorted(result.improvements.items(), key=lambda x: x[1], reverse=True):
                print(f"  {metric}: +{improvement:.2f}%")
        
        if result.degradations:
            print(f"\n⚠️  DEGRADATIONS")
            for metric, degradation in sorted(result.degradations.items(), key=lambda x: x[1], reverse=True):
                print(f"  {metric}: -{degradation:.2f}%")
        
        print("="*70 + "\n")
    
    def __del__(self):
        """Cleanup"""
        if self.grammar_tool is not None:
            self.grammar_tool.close()


def main():
    """Example usage"""
    
    # Initialize evaluator
    evaluator = CombinedEvaluator(model_name="gpt2", use_grammar_check=True)
    
    # Example texts
    ai_text = """
    Artificial intelligence is revolutionizing the way we approach 
    problem-solving in the modern era. Machine learning algorithms 
    are becoming increasingly sophisticated. Companies are adopting 
    these technologies to improve efficiency. Data analysis helps 
    organizations make better decisions. Technology continues to 
    advance rapidly in various sectors.
    """
    
    human_text = """
    So I was thinking about AI the other day, and honestly? It's pretty 
    wild how much it's changing everything. I mean, some companies are 
    using it for like, everything now. Makes you wonder where it's all 
    heading, right? My friend works in tech and he says it's crazy how 
    fast things are moving. Kinda exciting and scary at the same time!
    """
    
    # Evaluate AI text
    print("Evaluating AI-generated text...")
    ai_score = evaluator.evaluate(ai_text)
    evaluator.print_evaluation(ai_score, "AI Text")
    
    # Evaluate human text
    print("Evaluating Human-written text...")
    human_score = evaluator.evaluate(human_text)
    evaluator.print_evaluation(human_score, "Human Text")
    
    # Compare before/after
    print("Comparing AI text (before) vs Human text (after)...")
    comparison = evaluator.compare(ai_text, human_text, "AI Original", "Humanized Version")
    evaluator.print_comparison(comparison)
    
    # Save results as JSON
    print("Saving results to files...")
    with open("ai_evaluation.json", "w") as f:
        f.write(ai_score.to_json())
    
    with open("comparison_results.json", "w") as f:
        f.write(comparison.to_json())
    
    print("Results saved!")


if __name__ == "__main__":
    main()