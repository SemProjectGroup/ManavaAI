"""
Linguistic feature extraction for AI detection.
Analyzes text for patterns that distinguish AI from human writing.
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class LinguisticAnalyzer:
    """
    Analyze linguistic features of text for AI detection.
    """
    
    # Common filler words and hedging phrases (often more frequent in human writing)
    FILLER_WORDS = {
        'um', 'uh', 'like', 'you know', 'i mean', 'actually', 'basically',
        'literally', 'honestly', 'frankly', 'obviously', 'clearly'
    }
    
    # Transitional phrases (often overused in AI writing)
    TRANSITION_PHRASES = {
        'furthermore', 'moreover', 'additionally', 'consequently', 
        'therefore', 'thus', 'hence', 'accordingly', 'nevertheless',
        'nonetheless', 'however', 'in conclusion', 'in summary',
        'to summarize', 'in other words', 'for instance', 'for example'
    }
    
    # AI-typical formal phrases
    AI_TYPICAL_PHRASES = {
        'it is important to note', 'it should be noted', 'it is worth mentioning',
        'in today\'s world', 'in this day and age', 'plays a crucial role',
        'plays a vital role', 'it is essential', 'it is imperative',
        'leverage', 'utilize', 'facilitate', 'optimize', 'implement',
        'comprehensive', 'robust', 'seamless', 'innovative', 'cutting-edge'
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        self._load_word_frequencies()
    
    def _load_word_frequencies(self):
        """Load common word frequency data."""
        # Simple frequency approximation (in real implementation, load from file)
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she'
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Perform comprehensive linguistic analysis.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary of linguistic features
        """
        if not text or len(text.strip()) < 10:
            return self._empty_features()
        
        features = {}
        
        # Basic stats
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = self._avg_word_length(words)
        features['avg_sentence_length'] = len(words) / max(len(sentences), 1)
        
        # Vocabulary analysis
        features['vocabulary_richness'] = self._vocabulary_richness(words)
        features['hapax_ratio'] = self._hapax_ratio(words)
        features['type_token_ratio'] = self._type_token_ratio(words)
        
        # Sentence analysis
        features['sentence_length_variance'] = self._sentence_length_variance(sentences)
        features['sentence_starter_diversity'] = self._sentence_starter_diversity(sentences)
        
        # Complexity metrics
        features['syllables_per_word'] = self._avg_syllables(words)
        features['flesch_reading_ease'] = self._flesch_reading_ease(text, words, sentences)
        
        # Burstiness (variation in word/phrase usage patterns)
        features['burstiness'] = self._calculate_burstiness(words)
        
        # Perplexity approximation (using simple n-gram model)
        features['perplexity'] = self._approximate_perplexity(words)
        
        # Pattern detection
        features['transition_density'] = self._count_phrases(text, self.TRANSITION_PHRASES)
        features['ai_phrase_density'] = self._count_phrases(text, self.AI_TYPICAL_PHRASES)
        features['filler_word_density'] = self._count_phrases(text, self.FILLER_WORDS)
        
        # Punctuation analysis
        features['comma_ratio'] = text.count(',') / max(len(words), 1)
        features['question_ratio'] = text.count('?') / max(len(sentences), 1)
        features['exclamation_ratio'] = text.count('!') / max(len(sentences), 1)
        
        # First person usage
        features['first_person_ratio'] = self._first_person_ratio(words)
        
        # Repetition analysis
        features['word_repetition'] = self._word_repetition_score(words)
        features['phrase_repetition'] = self._phrase_repetition_score(text)
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'vocabulary_richness': 0,
            'hapax_ratio': 0,
            'type_token_ratio': 0,
            'sentence_length_variance': 0,
            'sentence_starter_diversity': 0,
            'syllables_per_word': 0,
            'flesch_reading_ease': 0,
            'burstiness': 0,
            'perplexity': 0,
            'transition_density': 0,
            'ai_phrase_density': 0,
            'filler_word_density': 0,
            'comma_ratio': 0,
            'question_ratio': 0,
            'exclamation_ratio': 0,
            'first_person_ratio': 0,
            'word_repetition': 0,
            'phrase_repetition': 0
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if w]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _avg_word_length(self, words: List[str]) -> float:
        """Calculate average word length."""
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)
    
    def _vocabulary_richness(self, words: List[str]) -> float:
        """Calculate vocabulary richness (unique words / total words)."""
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _hapax_ratio(self, words: List[str]) -> float:
        """Calculate ratio of words appearing only once."""
        if not words:
            return 0.0
        word_counts = Counter(words)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        return hapax / len(words)
    
    def _type_token_ratio(self, words: List[str]) -> float:
        """Calculate type-token ratio."""
        return self._vocabulary_richness(words)
    
    def _sentence_length_variance(self, sentences: List[str]) -> float:
        """Calculate variance in sentence lengths."""
        if len(sentences) < 2:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return np.var(lengths)
    
    def _sentence_starter_diversity(self, sentences: List[str]) -> float:
        """Calculate diversity of sentence starters."""
        if not sentences:
            return 0.0
        starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        starters = [s for s in starters if s]
        if not starters:
            return 0.0
        return len(set(starters)) / len(starters)
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        
        return max(count, 1)
    
    def _avg_syllables(self, words: List[str]) -> float:
        """Calculate average syllables per word."""
        if not words:
            return 0.0
        return sum(self._count_syllables(w) for w in words) / len(words)
    
    def _flesch_reading_ease(
        self, 
        text: str, 
        words: List[str], 
        sentences: List[str]
    ) -> float:
        """Calculate Flesch Reading Ease score."""
        if not words or not sentences:
            return 0.0
        
        asl = len(words) / len(sentences)  # Average sentence length
        asw = self._avg_syllables(words)    # Average syllables per word
        
        score = 206.835 - (1.015 * asl) - (84.6 * asw)
        return max(0, min(100, score))
    
    def _calculate_burstiness(self, words: List[str]) -> float:
        """
        Calculate burstiness - variation in word spacing.
        Human writing tends to have more burstiness (varied patterns).
        AI tends to be more uniform.
        """
        if len(words) < 10:
            return 0.0
        
        # Calculate inter-word intervals for common words
        word_positions = {}
        for i, word in enumerate(words):
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(i)
        
        # Calculate variance in intervals
        variances = []
        for word, positions in word_positions.items():
            if len(positions) >= 3:
                intervals = [positions[i+1] - positions[i] 
                           for i in range(len(positions)-1)]
                if intervals:
                    variances.append(np.var(intervals))
        
        return np.mean(variances) if variances else 0.0
    
    def _approximate_perplexity(self, words: List[str]) -> float:
        """
        Approximate perplexity using simple bigram model.
        Lower perplexity often indicates AI-generated text.
        """
        if len(words) < 5:
            return 0.0
        
        # Build bigram counts
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)
        
        # Calculate log probability
        log_prob = 0
        n = 0
        
        for (w1, w2), count in bigram_counts.items():
            prob = count / max(unigram_counts[w1], 1)
            if prob > 0:
                log_prob += math.log(prob)
                n += count
        
        if n == 0:
            return 0.0
        
        avg_log_prob = log_prob / n
        perplexity = math.exp(-avg_log_prob)
        
        return min(perplexity, 1000)  # Cap at 1000
    
    def _count_phrases(self, text: str, phrases: set) -> float:
        """Count density of phrases in text."""
        text_lower = text.lower()
        count = sum(1 for phrase in phrases if phrase in text_lower)
        words = len(text.split())
        return count / max(words, 1) * 100  # Per 100 words
    
    def _first_person_ratio(self, words: List[str]) -> float:
        """Calculate ratio of first-person pronouns."""
        first_person = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'}
        count = sum(1 for w in words if w in first_person)
        return count / max(len(words), 1)
    
    def _word_repetition_score(self, words: List[str]) -> float:
        """Calculate word repetition score."""
        if len(words) < 10:
            return 0.0
        
        word_counts = Counter(words)
        # Count words appearing more than expected
        expected_freq = 1 / len(set(words)) if words else 0
        
        repetition_score = 0
        for word, count in word_counts.items():
            if word not in self.common_words:
                actual_freq = count / len(words)
                if actual_freq > expected_freq * 2:
                    repetition_score += actual_freq - expected_freq
        
        return repetition_score
    
    def _phrase_repetition_score(self, text: str) -> float:
        """Calculate phrase repetition score."""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Check for repeated 3-grams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        
        repeated = sum(1 for count in trigram_counts.values() if count > 1)
        return repeated / max(len(trigrams), 1)
    
    def get_ai_score_estimate(self, features: Dict[str, float]) -> float:
        """
        Estimate AI probability based on linguistic features.
        This is a simple heuristic - the trained model will be more accurate.
        
        Args:
            features: Dictionary of linguistic features
        
        Returns:
            Estimated AI probability (0-1)
        """
        score = 0.5  # Start neutral
        
        # Low sentence length variance suggests AI
        if features.get('sentence_length_variance', 50) < 20:
            score += 0.1
        
        # Low burstiness suggests AI
        if features.get('burstiness', 10) < 5:
            score += 0.1
        
        # High transition phrase density suggests AI
        if features.get('transition_density', 0) > 2:
            score += 0.1
        
        # High AI phrase density suggests AI
        if features.get('ai_phrase_density', 0) > 1:
            score += 0.15
        
        # Low first-person usage suggests AI
        if features.get('first_person_ratio', 0.05) < 0.02:
            score += 0.05
        
        # Very uniform vocabulary suggests AI
        if features.get('vocabulary_richness', 0.5) > 0.7:
            score += 0.05
        
        # Low filler word usage suggests AI
        if features.get('filler_word_density', 0) < 0.5:
            score += 0.05
        
        return min(max(score, 0.0), 1.0)


def main():
    """Test the linguistic analyzer."""
    analyzer = LinguisticAnalyzer()
    
    # Test with AI-like text
    ai_text = """
    Artificial intelligence has fundamentally transformed the landscape of modern 
    technology. Furthermore, the implementation of machine learning algorithms has 
    enabled unprecedented levels of automation and efficiency. It is important to 
    note that these advancements have significant implications for various industries. 
    Moreover, the integration of AI systems continues to accelerate innovation across 
    multiple sectors. In conclusion, artificial intelligence represents a paradigm 
    shift in how we approach complex problems.
    """
    
    # Test with human-like text
    human_text = """
    I've been thinking about AI lately, and honestly? It's kind of wild how much 
    things have changed. Like, remember when we thought robots were just sci-fi 
    stuff? Now my phone can literally finish my sentences. Sometimes it gets it 
    wrong though - autocorrect fails are the worst! But yeah, I guess it's pretty 
    cool overall. My friend was telling me about ChatGPT the other day and I was 
    like, wait, it can write essays?? That's nuts.
    """
    
    print("="*60)
    print("LINGUISTIC ANALYSIS TEST")
    print("="*60)
    
    print("\n--- AI-like Text Analysis ---")
    ai_features = analyzer.analyze(ai_text)
    for key, value in ai_features.items():
        print(f"  {key}: {value:.3f}")
    print(f"  Estimated AI Score: {analyzer.get_ai_score_estimate(ai_features):.2%}")
    
    print("\n--- Human-like Text Analysis ---")
    human_features = analyzer.analyze(human_text)
    for key, value in human_features.items():
        print(f"  {key}: {value:.3f}")
    print(f"  Estimated AI Score: {analyzer.get_ai_score_estimate(human_features):.2%}")


if __name__ == '__main__':
    main()