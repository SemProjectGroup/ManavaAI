"""
Advanced Text Humanizer V2 - Aggressively makes AI text sound human
Specifically designed to counter AI detection patterns.
"""

import re
import random
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextHumanizer:
    """
    Transforms AI-generated text to sound naturally human.
    Targets specific patterns that AI detectors look for.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the humanizer."""
        logger.info("Initializing Advanced Text Humanizer V2...")
        
        # === AI PHRASES TO REMOVE/REPLACE ===
        # These MUST be replaced as detectors specifically look for them
        self.ai_phrase_replacements = {
            # Conclusions - CRITICAL to replace
            "in conclusion": ["so yeah", "at the end of the day", "honestly", "all things considered", "looking at all this"],
            "to summarize": ["so basically", "long story short", "the point is"],
            "in summary": ["basically", "to put it simply", "the main thing is"],
            "to sum up": ["so yeah", "basically", "the thing is"],
            "all in all": ["when you think about it", "honestly", "at the end of the day"],
            
            # "One of the most" - CRITICAL
            "one of the most": ["probably the most", "really", "a super", "a really", "definitely a"],
            "some of the most": ["really", "some super", "some really"],
            
            # Importance phrases - CRITICAL
            "it is important to note": ["here's the thing -", "something to keep in mind is", "you should know that"],
            "it's important to note": ["the thing is,", "just so you know,", "here's something -"],
            "it is worth noting": ["interestingly,", "fun fact -", "what's cool is"],
            "it's worth noting": ["something interesting is", "by the way,", "here's a cool thing -"],
            "it is essential to": ["you really gotta", "you need to", "it's key to"],
            "it is crucial to": ["you really should", "it's super important to", "you've gotta"],
            "it is vital to": ["you need to", "it's really important to", "you've gotta"],
            "it is important to understand": ["the thing to get is", "what matters here is", "here's what's important -"],
            
            # Plays a role phrases - CRITICAL
            "plays a crucial role": ["is super important", "really matters", "is a big deal", "makes a huge difference"],
            "plays a vital role": ["is really important", "matters a ton", "is essential", "is key"],
            "plays a significant role": ["is pretty important", "matters quite a bit", "is a big factor"],
            "plays an important role": ["matters", "is important", "helps a lot", "makes a difference"],
            "played a crucial role": ["was super important", "really mattered", "was a big deal"],
            "played a vital role": ["was really important", "mattered a ton", "was essential"],
            
            # Formal transitions - CRITICAL
            "furthermore": ["also", "plus", "and", "on top of that", "another thing is"],
            "moreover": ["plus", "and also", "not just that but", "and"],
            "additionally": ["also", "and", "plus", "oh and"],
            "consequently": ["so", "because of that", "that's why", "and so"],
            "nevertheless": ["but still", "even so", "still though", "but hey"],
            "nonetheless": ["but still", "even so", "that said", "still"],
            "therefore": ["so", "that's why", "which is why", "and that's why"],
            "thus": ["so", "and so", "that's how", "which means"],
            "hence": ["so", "that's why", "which is why"],
            "accordingly": ["so", "that's why", "because of this"],
            "subsequently": ["then", "after that", "later", "and then"],
            "henceforth": ["from now on", "after this", "going forward"],
            
            # Structure phrases
            "firstly": ["first", "first off", "so first", "to start"],
            "secondly": ["second", "then", "next", "also"],
            "thirdly": ["third", "and then", "next up", "also"],
            "finally": ["and finally", "last thing", "lastly", "and"],
            "lastly": ["and finally", "last thing", "oh and one more thing"],
            "in addition": ["also", "plus", "and", "on top of that"],
            "as a result": ["so", "because of this", "that's why", "and so"],
            "for instance": ["like", "for example", "such as", "say"],
            "for example": ["like", "such as", "say", "think of"],
            
            # AI-specific phrases
            "in today's world": ["nowadays", "these days", "now", "today"],
            "in today's society": ["nowadays", "these days", "in our world"],
            "in today's digital age": ["with all the tech nowadays", "these days", "now"],
            "in the modern era": ["nowadays", "today", "these days", "now"],
            "of the modern era": ["of today", "of our time", "nowadays"],
            "in the digital age": ["with technology nowadays", "these days", "now"],
            "in this day and age": ["nowadays", "these days", "now"],
            "in recent years": ["lately", "recently", "these days"],
            "over the years": ["over time", "as time went on", "through the years"],
            "throughout history": ["for ages", "for a long time", "historically"],
            
            # Cannot be phrases
            "cannot be overstated": ["is huge", "is really important", "can't be ignored"],
            "cannot be understated": ["is really significant", "is a big deal", "matters a lot"],
            
            # Listing phrases
            "a wide range of": ["lots of", "all kinds of", "many", "tons of"],
            "a variety of": ["different", "various", "all sorts of", "many kinds of"],
            "a myriad of": ["tons of", "so many", "lots of", "a bunch of"],
            "a plethora of": ["tons of", "loads of", "so many", "lots of"],
            "a multitude of": ["lots of", "many", "tons of", "a bunch of"],
            "there are several": ["there's a few", "there are some", "you'll find some"],
            "there are many": ["there are lots of", "there's tons of", "you'll find many"],
            "there are numerous": ["there are tons of", "there are so many", "there's lots of"],
            "there are various": ["there are different", "there's all kinds of", "you'll find various"],
            
            # Authority phrases
            "research has shown": ["studies show", "research shows", "apparently", "it turns out"],
            "studies have shown": ["research shows", "studies show", "it seems like"],
            "research suggests": ["it looks like", "studies hint", "research points to"],
            "studies suggest": ["it seems", "research hints", "studies point to"],
            "evidence suggests": ["it looks like", "it seems", "things point to"],
            "experts agree": ["most people agree", "experts say", "people generally think"],
            "it has been proven": ["we know", "it's been shown", "it's clear"],
            "it is well established": ["we know", "it's known", "it's pretty clear"],
            "it is widely accepted": ["most people think", "it's generally agreed", "people accept that"],
            "it is commonly known": ["everyone knows", "it's known", "people know"],
            
            # Other AI phrases
            "on the other hand": ["but then again", "but", "although", "on the flip side"],
            "in contrast": ["but", "unlike this", "on the other hand"],
            "as opposed to": ["unlike", "instead of", "rather than"],
            "it can be argued": ["you could say", "some think", "arguably"],
            "one might argue": ["you could say", "some might say", "maybe"],
            "in order to": ["to", "so that", "for"],
            "due to the fact that": ["because", "since", "as"],
            "the fact that": ["that", "how"],
            "this demonstrates": ["this shows", "you can see", "it's clear"],
            "this illustrates": ["this shows", "you can see", "this is a good example of"],
            "this highlights": ["this shows", "you can see", "this points out"],
            "this underscores": ["this shows", "this really shows", "this proves"],
            "serves as a testament": ["shows", "proves", "is proof"],
            "stands as a testament": ["shows", "proves", "is proof"],
            "is widely regarded": ["people think of it", "is seen", "is considered"],
            "has emerged as": ["became", "has become", "turned into"],
            "evolved significantly": ["changed a lot", "really changed", "grew"],
            "the significance of": ["how important", "why", "the importance of"],
            "the importance of": ["how important", "why", "how much"],
        }
        
        # Contractions - MUST apply these
        self.contractions = [
            (r"\b[Ii]t is\b", "it's"),
            (r"\b[Ii]t has\b", "it's"),
            (r"\b[Tt]hey are\b", "they're"),
            (r"\b[Tt]hey have\b", "they've"),
            (r"\b[Tt]hey will\b", "they'll"),
            (r"\b[Ww]e are\b", "we're"),
            (r"\b[Ww]e have\b", "we've"),
            (r"\b[Ww]e will\b", "we'll"),
            (r"\b[Yy]ou are\b", "you're"),
            (r"\b[Yy]ou have\b", "you've"),
            (r"\b[Yy]ou will\b", "you'll"),
            (r"\b[Hh]e is\b", "he's"),
            (r"\b[Ss]he is\b", "she's"),
            (r"\b[Tt]here is\b", "there's"),
            (r"\b[Tt]here are\b", "there are"),  # Keep this one natural
            (r"\b[Tt]hat is\b", "that's"),
            (r"\b[Ww]hat is\b", "what's"),
            (r"\b[Hh]ere is\b", "here's"),
            (r"\b[Ww]ho is\b", "who's"),
            (r"\b[Hh]ow is\b", "how's"),
            (r"\bcannot\b", "can't"),
            (r"\bcan not\b", "can't"),
            (r"\bwill not\b", "won't"),
            (r"\bdo not\b", "don't"),
            (r"\bdoes not\b", "doesn't"),
            (r"\bdid not\b", "didn't"),
            (r"\bwould not\b", "wouldn't"),
            (r"\bcould not\b", "couldn't"),
            (r"\bshould not\b", "shouldn't"),
            (r"\bhave not\b", "haven't"),
            (r"\bhas not\b", "hasn't"),
            (r"\bhad not\b", "hadn't"),
            (r"\bis not\b", "isn't"),
            (r"\bare not\b", "aren't"),
            (r"\bwas not\b", "wasn't"),
            (r"\bwere not\b", "weren't"),
            (r"\bI am\b", "I'm"),
            (r"\bI have\b", "I've"),
            (r"\bI will\b", "I'll"),
            (r"\bI would\b", "I'd"),
            (r"\blet us\b", "let's"),
        ]
        
        # Formal words to replace
        self.formal_words = {
            "utilize": ["use", "work with"],
            "implement": ["use", "put in place", "set up"],
            "facilitate": ["help", "make easier", "help with"],
            "comprehensive": ["complete", "full", "thorough"],
            "numerous": ["many", "lots of", "a bunch of"],
            "sufficient": ["enough"],
            "approximately": ["about", "around", "roughly"],
            "demonstrate": ["show", "prove"],
            "illustrate": ["show", "explain"],
            "indicate": ["show", "suggest", "point to"],
            "significant": ["big", "major", "important"],
            "substantial": ["big", "large", "major"],
            "considerable": ["quite a bit of", "lots of", "a good amount of"],
            "obtain": ["get"],
            "acquire": ["get", "pick up"],
            "possess": ["have", "got"],
            "require": ["need"],
            "attempt": ["try"],
            "commence": ["start", "begin"],
            "terminate": ["end", "stop"],
            "purchase": ["buy", "get"],
            "inquire": ["ask"],
            "assist": ["help"],
            "endeavor": ["try", "effort"],
            "frequently": ["often", "a lot"],
            "occasionally": ["sometimes", "now and then"],
            "immediately": ["right away", "now"],
            "previously": ["before", "earlier"],
            "currently": ["now", "right now"],
            "primarily": ["mainly", "mostly"],
            "predominantly": ["mostly", "mainly"],
            "essentially": ["basically", "really"],
            "fundamentally": ["basically", "at its core"],
            "extremely": ["really", "super", "very"],
            "exceptionally": ["really", "super"],
            "remarkably": ["really", "surprisingly"],
            "particularly": ["especially", "really"],
            "specifically": ["especially"],
            "undoubtedly": ["definitely", "for sure"],
            "certainly": ["definitely", "for sure"],
            "indispensable": ["essential", "must-have"],
            "paramount": ["super important", "crucial"],
            "multifaceted": ["complex", "many-sided"],
            "intricate": ["complex", "detailed"],
            "robust": ["strong", "solid"],
            "optimal": ["best", "ideal"],
            "diverse": ["different", "varied"],
            "regarding": ["about"],
            "concerning": ["about"],
            "pertaining": ["about", "related to"],
            "whereby": ["where", "by which"],
            "wherein": ["where", "in which"],
            "thereof": ["of it", "of that"],
            "henceforth": ["from now on"],
            "notwithstanding": ["despite", "even though"],
            "aforementioned": ["mentioned", "that"],
            "subsequently": ["then", "later", "after"],
            "prior": ["before", "earlier"],
            "hence": ["so", "therefore"],
            "thus": ["so", "this way"],
            "moreover": ["also", "plus"],
            "furthermore": ["also", "plus", "and"],
            "nevertheless": ["but", "still", "however"],
            "nonetheless": ["still", "but", "even so"],
            "accordingly": ["so"],
            "consequently": ["so", "as a result"],
            "whereas": ["while", "but"],
            "albeit": ["although", "even though"],
        }
        
        # Casual phrases to insert
        self.casual_starters = [
            "Honestly, ", "I think ", "You know, ", "Basically, ",
            "The thing is, ", "Here's the deal - ", "Look, ",
            "So basically, ", "What's cool is ", "Fun fact - ",
        ]
        
        self.casual_connectors = [
            " - and honestly ", " - which is pretty cool", " you know?",
            ", right?", " - pretty neat actually", ", if you think about it",
        ]
        
        logger.info("✅ Advanced Text Humanizer V2 initialized!")
    
    def _replace_ai_phrases(self, text: str) -> Tuple[str, int]:
        """Replace AI phrases with human alternatives. Returns (text, count)."""
        result = text
        count = 0
        
        # Sort by length (longer phrases first)
        sorted_phrases = sorted(self.ai_phrase_replacements.keys(), key=len, reverse=True)
        
        for phrase in sorted_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            matches = pattern.findall(result)
            
            if matches:
                replacement = random.choice(self.ai_phrase_replacements[phrase])
                
                def replace_with_case(match):
                    original = match.group(0)
                    if original[0].isupper():
                        return replacement[0].upper() + replacement[1:]
                    return replacement
                
                result = pattern.sub(replace_with_case, result)
                count += len(matches)
        
        return result, count
    
    def _apply_contractions(self, text: str) -> str:
        """Apply contractions throughout the text."""
        result = text
        
        for pattern, replacement in self.contractions:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _replace_formal_words(self, text: str, intensity: str = 'medium') -> str:
        """Replace formal words with casual alternatives."""
        result = text
        
        prob = {'light': 0.5, 'medium': 0.75, 'aggressive': 0.95}.get(intensity, 0.75)
        
        for word, replacements in self.formal_words.items():
            if random.random() < prob:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                replacement = random.choice(replacements)
                
                def replace_with_case(match):
                    if match.group(0)[0].isupper():
                        return replacement[0].upper() + replacement[1:]
                    return replacement
                
                result = pattern.sub(replace_with_case, result)
        
        return result
    
    def _add_casual_elements(self, text: str, intensity: str = 'medium') -> str:
        """Add casual phrases and sentence starters."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result_sentences = []
        
        starter_prob = {'light': 0.08, 'medium': 0.15, 'aggressive': 0.25}.get(intensity, 0.15)
        connector_prob = {'light': 0.05, 'medium': 0.10, 'aggressive': 0.18}.get(intensity, 0.10)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            modified = sentence
            
            # Add casual starter to some sentences (not the first one)
            if i > 0 and i % 3 == 0 and random.random() < starter_prob:
                starter = random.choice(self.casual_starters)
                if modified[0].isupper():
                    modified = starter + modified[0].lower() + modified[1:]
                else:
                    modified = starter + modified
            
            # Add casual connector to end of some sentences
            if random.random() < connector_prob and not modified.rstrip().endswith('?'):
                connector = random.choice(self.casual_connectors)
                modified = modified.rstrip('.') + connector + '.'
            
            result_sentences.append(modified)
        
        return ' '.join(result_sentences)
    
    def _vary_sentence_structure(self, text: str) -> str:
        """Vary sentence structure for more natural flow."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            words = sentence.split()
            
            # Occasionally split very long sentences
            if len(words) > 28 and ', ' in sentence and random.random() < 0.5:
                parts = sentence.split(', ', 1)
                if len(parts[0].split()) > 6 and len(parts[1].split()) > 6:
                    sentence = parts[0] + '. ' + parts[1][0].upper() + parts[1][1:]
            
            # Occasionally start with "And" or "But" (informal style)
            if i > 0 and len(result) > 0 and random.random() < 0.12:
                starters = ["And ", "But ", "So ", "Plus, "]
                if sentence[0].isupper() and not sentence.startswith(tuple(starters)):
                    sentence = random.choice(starters) + sentence[0].lower() + sentence[1:]
            
            result.append(sentence)
        
        return ' '.join(result)
    
    def _add_personal_touches(self, text: str, intensity: str = 'medium') -> str:
        """Add personal pronouns and opinions."""
        if intensity == 'light':
            return text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        personal_phrases = [
            ("I think ", 0.08),
            ("In my opinion, ", 0.05),
            ("I'd say ", 0.06),
            ("If you ask me, ", 0.04),
        ]
        
        if intensity == 'aggressive':
            personal_phrases = [(p, prob * 2) for p, prob in personal_phrases]
        
        for i, sentence in enumerate(sentences):
            modified = sentence
            
            # Add personal touch to some sentences
            if i > 0 and i % 4 == 0:
                for phrase, prob in personal_phrases:
                    if random.random() < prob:
                        if modified[0].isupper():
                            modified = phrase + modified[0].lower() + modified[1:]
                        break
            
            result.append(modified)
        
        return ' '.join(result)
    
    def _fix_grammar(self, text: str) -> str:
        """Fix grammar issues from transformations."""
        result = text
        
        # Fix double spaces
        result = re.sub(r'\s+', ' ', result)
        
        # Fix capitalization after sentence endings
        result = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), result)
        
        # Fix "a" vs "an"
        result = re.sub(r'\ba ([aeiouAEIOU])', r'an \1', result)
        
        # Fix spacing around punctuation
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        result = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', result)
        
        # Fix multiple punctuation
        result = re.sub(r'\.{2,}', '.', result)
        result = re.sub(r'\?{2,}', '?', result)
        result = re.sub(r'!{2,}', '!', result)
        result = re.sub(r'-{3,}', ' - ', result)
        
        # Fix "I" capitalization
        result = re.sub(r'\bi\b', 'I', result)
        result = re.sub(r"\bi'm\b", "I'm", result)
        result = re.sub(r"\bi've\b", "I've", result)
        result = re.sub(r"\bi'll\b", "I'll", result)
        result = re.sub(r"\bi'd\b", "I'd", result)
        
        # Fix sentence start capitalization
        sentences = re.split(r'(?<=[.!?])\s+', result)
        fixed_sentences = []
        for s in sentences:
            if s:
                fixed_sentences.append(s[0].upper() + s[1:] if len(s) > 1 else s.upper())
        result = ' '.join(fixed_sentences)
        
        # Clean up any remaining issues
        result = result.replace(' .', '.')
        result = result.replace(' ,', ',')
        result = result.replace('  ', ' ')
        
        return result.strip()
    
    def humanize(
        self,
        text: str,
        style: str = 'casual',
        intensity: str = 'medium'
    ) -> str:
        """
        Humanize AI-generated text.
        
        Args:
            text: Input text to humanize
            style: 'casual', 'formal', or 'academic'
            intensity: 'light', 'medium', or 'aggressive'
            
        Returns:
            Humanized text
        """
        if not text or len(text.strip()) < 20:
            return text
        
        result = text.strip()
        
        # Step 1: Replace ALL AI phrases (most important!)
        result, phrase_count = self._replace_ai_phrases(result)
        logger.info(f"   Replaced {phrase_count} AI phrases")
        
        # Step 2: Apply contractions (very important for human feel)
        result = self._apply_contractions(result)
        
        # Step 3: Replace formal words
        result = self._replace_formal_words(result, intensity)
        
        # Step 4: Vary sentence structure
        if intensity != 'light':
            result = self._vary_sentence_structure(result)
        
        # Step 5: Add casual elements
        if style == 'casual':
            result = self._add_casual_elements(result, intensity)
        
        # Step 6: Add personal touches
        if style == 'casual' and intensity in ['medium', 'aggressive']:
            result = self._add_personal_touches(result, intensity)
        
        # Step 7: Fix any grammar issues
        result = self._fix_grammar(result)
        
        return result
    
    def humanize_aggressive(self, text: str) -> str:
        """Shortcut for maximum humanization."""
        return self.humanize(text, style='casual', intensity='aggressive')


def main():
    """Test the humanizer."""
    humanizer = TextHumanizer()
    
    test_text = """Cats are one of the most popular domestic animals in the world. They have lived alongside humans for thousands of years and are known for their independence, intelligence, and graceful behavior. Unlike many other pets, cats can take care of themselves to a large extent, which makes them suitable for people with busy lifestyles.

One of the most interesting traits of cats is their behavior. They communicate using body language, sounds, and facial expressions. A cat may purr when it is happy, hiss when it feels threatened, or rub against people to show affection.

In conclusion, cats are fascinating animals that combine independence with affection. Their unique behavior, physical abilities, and companionship make them special pets."""
    
    print("=" * 60)
    print("ORIGINAL:")
    print("=" * 60)
    print(test_text)
    
    print("\n" + "=" * 60)
    print("HUMANIZED (aggressive):")
    print("=" * 60)
    result = humanizer.humanize(test_text, style='casual', intensity='aggressive')
    print(result)
    
    # Count improvements
    print("\n" + "=" * 60)
    print("CHANGES MADE:")
    print("=" * 60)
    print(f"  'in conclusion' removed: {'in conclusion' not in result.lower()}")
    print(f"  'one of the most' removed: {'one of the most' not in result.lower()}")
    print(
    f"  Contractions added: "
    f"{'it\'s' in result.lower() or 'they\'re' in result.lower()}"
)


if __name__ == '__main__':
    main()