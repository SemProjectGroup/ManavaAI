"""
Text preprocessing utilities for ManavAI.
Handles text cleaning, normalization, and preparation.
"""

import re
import unicodedata
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing class for cleaning and normalizing text.
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = False,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
        min_length: int = 10,
        max_length: int = 10000
    ):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters
            normalize_whitespace: Normalize whitespace
            normalize_unicode: Normalize unicode characters
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|ftp://\S+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.special_char_pattern = re.compile(
            r'[^\w\s.,!?;:\'"()\-]'
        )
        self.whitespace_pattern = re.compile(r'\s+')
        self.repeated_punct_pattern = re.compile(r'([.!?])\1+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = self.special_char_pattern.sub(' ', text)
        
        # Normalize repeated punctuation
        text = self.repeated_punct_pattern.sub(r'\1', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
            text = text.strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def validate_text(self, text: str) -> Tuple[bool, str]:
        """
        Validate text meets requirements.
        
        Args:
            text: Text to validate
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if not text:
            return False, "Empty text"
        
        if len(text) < self.min_length:
            return False, f"Text too short (min: {self.min_length})"
        
        if len(text) > self.max_length:
            return False, f"Text too long (max: {self.max_length})"
        
        # Check for mostly non-alphabetic content
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return False, "Text contains too few alphabetic characters"
        
        # Check for repeated content
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False, "Text contains too much repetition"
        
        return True, "Valid"
    
    def preprocess_for_training(
        self, 
        text: str, 
        max_words: int = 500
    ) -> Optional[str]:
        """
        Preprocess text specifically for model training.
        
        Args:
            text: Raw text
            max_words: Maximum number of words
        
        Returns:
            Preprocessed text or None if invalid
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Validate
        is_valid, reason = self.validate_text(cleaned)
        if not is_valid:
            return None
        
        # Truncate to max words
        words = cleaned.split()
        if len(words) > max_words:
            # Try to truncate at sentence boundary
            truncated = ' '.join(words[:max_words])
            last_period = truncated.rfind('.')
            if last_period > len(truncated) * 0.7:
                truncated = truncated[:last_period + 1]
            cleaned = truncated
        
        return cleaned
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1<PERIOD> ', text)
        text = re.sub(r'\b(vs|etc|i\.e|e\.g)\.\s', r'\1<PERIOD> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
        
        Returns:
            List of paragraphs
        """
        # Split on double newlines or paragraph markers
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def extract_words(self, text: str) -> List[str]:
        """
        Extract words from text.
        
        Args:
            text: Text to process
        
        Returns:
            List of words
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.lower().split()
        
        return words
    
    def get_text_stats(self, text: str) -> dict:
        """
        Get statistics about text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary of statistics
        """
        words = self.extract_words(text)
        sentences = self.split_into_sentences(text)
        paragraphs = self.split_into_paragraphs(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / max(len(words), 1)
        }


def main():
    """Test the preprocessor."""
    preprocessor = TextPreprocessor()
    
    test_text = """
    This is a test text with some URLs: https://example.com and 
    emails: test@example.com. It has multiple   spaces and
    special characters like @#$%.
    
    This is a second paragraph with more content!!!
    """
    
    print("Original:")
    print(test_text)
    print("\nCleaned:")
    print(preprocessor.clean_text(test_text))
    print("\nStats:")
    print(preprocessor.get_text_stats(test_text))


if __name__ == '__main__':
    main()