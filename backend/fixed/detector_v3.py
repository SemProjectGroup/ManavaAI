"""
ManavAI Ultimate Detector V3 - FIXED LABELS
"""

import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))


class ModernAIDetector:
    
    def __init__(self):
        print("Initializing Ultimate AI Detector V3...")
        
        # === AI INDICATOR PHRASES ===
        self.strong_ai_phrases = [
            "it's important to note", "it's worth noting", "it is important to note",
            "it is worth noting", "it's worth mentioning", "it is essential to",
            "it is crucial to", "it is vital to", "it is important to understand",
            "first and foremost", "in this article", "in this essay",
            "in this discussion", "let's dive in", "let's explore", "let's delve",
            "in conclusion", "to summarize", "in summary", "to sum up",
            "all in all", "in essence", "at the end of the day",
            "it can be argued", "one might argue", "some might say",
            "it could be said", "it's safe to say", "needless to say",
            "it goes without saying",
            "cannot be overstated", "cannot be understated",
            "plays a crucial role", "plays a vital role",
            "plays a significant role", "plays an important role",
            "played a crucial role", "played a vital role",
            "serves as a testament", "stands as a testament",
            "furthermore", "moreover", "additionally", "consequently",
            "nevertheless", "nonetheless", "henceforth", "thereby",
            "hence", "accordingly", "subsequently", "thus", "therefore",
            "in today's world", "in today's society", "in today's digital age",
            "in the modern era", "of the modern era", "in the digital age",
            "in this day and age", "in recent years", "over the years",
            "over time", "throughout history", "since the dawn of",
            "has evolved significantly", "evolved significantly",
            "has emerged as", "emerged as one of", "is widely regarded",
            "is often considered", "are widely recognized",
            "firstly", "secondly", "thirdly", "finally", "lastly",
            "in addition", "as a result", "for instance", "for example",
            "this transition", "this evolution", "this transformation",
            "the evolution of", "the emergence of", "the development of",
            "the importance of", "the significance of", "the impact of",
            "the role of", "reflects a", "reflecting a", "demonstrates a",
            "illustrates a", "underscores the", "highlights the",
            "it is essential", "it is important", "it is necessary",
            "a wide range of", "a broad range of", "a variety of",
            "a myriad of", "a plethora of", "a multitude of",
            "there are several", "there are many", "there are numerous",
            "foster growth", "foster innovation", "drive innovation",
            "promote inclusivity", "enhance productivity", "boost efficiency",
            "maximize potential", "unlock potential", "leverage technology",
            "navigate challenges", "address concerns", "tackle issues",
            "informed citizens", "active participation", "contribute positively",
            "on the other hand", "on one hand", "by contrast", "in contrast",
            "compared to", "as opposed to", "while some may", "although some",
            "despite the fact", "regardless of", "irrespective of",
            "research has shown", "studies have shown", "research suggests",
            "studies suggest", "evidence suggests", "experts agree",
            "it has been proven", "it is well established", "it is widely accepted",
            "one of the most", "some of the most", "many of the",
            "the majority of", "a significant number", "a growing number",
            "an important", "an essential", "an indispensable",
            "this highlights", "this demonstrates", "this illustrates",
            "this underscores", "this emphasizes", "this suggests",
            "this indicates", "this shows", "this reveals",
            "as such", "with that said", "that being said", "having said that",
            "by doing so", "in doing so", "therefore it is",
        ]
        
        self.ai_indicator_words = {
            'comprehensive', 'facilitate', 'utilize', 'implement', 'leverage',
            'optimize', 'enhance', 'streamline', 'paradigm', 'synergy',
            'multifaceted', 'holistic', 'robust', 'scalable', 'innovative',
            'transformative', 'unprecedented', 'realm', 'landscape', 'foster',
            'delve', 'intricate', 'pivotal', 'crucial', 'imperative',
            'encompass', 'myriad', 'plethora', 'aforementioned', 'herein',
            'whereby', 'thereof', 'therein', 'wherein', 'notwithstanding',
            'pertaining', 'respective', 'subsequent', 'prior',
            'discourse', 'interdisciplinary', 'empirical', 'theoretical',
            'foundational', 'contemporary', 'assessments', 'refinement',
            'variability', 'proliferation', 'resilience', 'mitigation',
            'adaptation', 'sustainable', 'integration', 'prominence',
            'indispensable', 'societal', 'cohesion', 'stability',
            'significant', 'substantial', 'considerable', 'remarkable',
            'notable', 'profound', 'extensive', 'thorough',
            'invaluable', 'paramount', 'quintessential',
            'exemplary', 'exceptional', 'outstanding', 'extraordinary',
            'progressive', 'essential', 'vital', 'critical',
            'emphasize', 'underscore', 'highlight', 'illustrate', 'demonstrate',
            'exemplify', 'embody', 'entail', 'necessitate',
            'enable', 'empower', 'cultivate', 'nurture', 'emerged',
            'evolved', 'expanded', 'established', 'consolidated', 'recognized',
            'promotes', 'ensures', 'provides', 'contributes', 'influences',
            'essentially', 'fundamentally', 'inherently', 'intrinsically',
            'particularly', 'specifically', 'notably', 'importantly',
            'significantly', 'ultimately', 'primarily', 'predominantly',
            'largely', 'increasingly', 'rapidly', 'effectively', 'actively',
        }
        
        self.contractions = {
            "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
            "he's", "she's", "it's", "we're", "we've", "we'll", "we'd",
            "they're", "they've", "they'll", "they'd", "that's", "there's",
            "here's", "what's", "who's", "how's", "where's", "when's", "why's",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
            "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
            "hadn't", "let's", "gonna", "wanna", "gotta", "kinda", "sorta",
            "dunno", "ain't", "y'all", "could've", "would've", "should've",
        }
        
        self.casual_expressions = {
            'lol', 'lmao', 'rofl', 'haha', 'hahaha', 'omg', 'wtf',
            'smh', 'tbh', 'ngl', 'imo', 'imho', 'idk', 'ikr', 'irl',
            'btw', 'fyi', 'brb', 'gtg', 'thx', 'pls', 'plz',
            'lowkey', 'highkey', 'fr', 'bet', 'lit', 'fire', 'slay',
            'vibe', 'vibes', 'mood', 'salty', 'shook', 'tea',
            'sis', 'bro', 'dude', 'fam', 'squad', 'goals',
            'yeah', 'yep', 'yup', 'nope', 'nah', 'meh', 'ugh', 'argh',
            'hmm', 'umm', 'um', 'uh', 'ah', 'oh', 'ooh', 'wow',
            'yay', 'woohoo', 'oops', 'whoops', 'yikes', 'geez', 'gosh',
            'damn', 'crap', 'cmon', 'ok', 'okay', 'whatevs',
            'totes', 'adorbs', 'probs', 'obvi', 'legit', 'perf',
            'basically', 'literally', 'honestly', 'obviously',
            'apparently', 'totally', 'absolutely', 'anyway', 'anyways',
        }
        
        self.personal_phrases = [
            "i remember", "i think", "i believe", "i feel", "i guess",
            "i suppose", "i mean", "i know", "i hope", "i wish",
            "i tried", "i went", "i saw", "i got", "i made",
            "i was like", "i said", "i told", "i asked", "i thought",
            "my mom", "my dad", "my friend", "my boss", "my dog", "my cat",
            "my wife", "my husband", "my boyfriend", "my girlfriend",
            "my brother", "my sister", "my family", "my coworker",
            "yesterday", "last week", "last month", "last year", "last night",
            "this morning", "the other day", "a few days ago", "when i was",
            "growing up", "back in", "one time", "this one time",
            "not gonna lie", "to be honest", "can't believe", "no way",
            "so basically", "long story short", "funny story",
        ]
        
        self.ai_sentence_starters = [
            "it is", "it's important", "there are", "there is", "this is",
            "these are", "one of", "the importance", "the significance",
            "the impact", "the role", "the concept", "the idea",
            "in order to", "in addition", "in fact", "in general",
            "as a result", "as such", "as mentioned", "as discussed",
            "furthermore", "moreover", "however", "therefore", "thus",
            "additionally", "consequently", "nevertheless", "nonetheless",
            "firstly", "secondly", "thirdly", "finally", "lastly",
            "overall", "ultimately", "essentially", "education is",
            "education plays", "education helps", "education provides",
            "an educated", "countries with", "through formal",
        ]
        
        self._load_transformer()
        print("✅ Ultimate AI Detector V3 initialized!")
    
    def _load_transformer(self):
        """Load pre-trained transformer model."""
        self.transformer_available = False
        self.model = None
        self.tokenizer = None
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            models_to_try = [
                "Hello-SimpleAI/chatgpt-detector-roberta",
                "roberta-base-openai-detector",
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"   Loading {model_name}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.to(self.device)
                    self.model.eval()
                    self.transformer_available = True
                    self.model_name = model_name
                    print(f"   ✅ Loaded {model_name}")
                    break
                except Exception as e:
                    print(f"   ⚠️ Could not load {model_name}: {e}")
                    continue
            
            if not self.transformer_available:
                print("   ⚠️ No transformer model available.")
                
        except ImportError:
            print("   ⚠️ Transformers library not available.")
    
    def _get_transformer_score(self, text: str) -> float:
        """Get AI probability from transformer model - CORRECT LABELS."""
        if not self.transformer_available:
            return 0.5
        
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # CORRECT: Label 0 = Human, Label 1 = AI
                ai_prob = probs[0][1].item()
            
            return ai_prob
            
        except Exception as e:
            return 0.5
    
    def _analyze_text(self, text: str) -> Dict:
        """Comprehensive text analysis."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        clean_words = [re.sub(r'[^\w]', '', w) for w in words]
        clean_words = [w for w in clean_words if w]
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        sentence_count = max(len(sentences), 1)
        
        analysis = {
            'word_count': word_count,
            'sentence_count': sentence_count,
        }
        
        # Count AI phrase matches
        ai_phrase_matches = []
        for phrase in self.strong_ai_phrases:
            if phrase in text_lower:
                ai_phrase_matches.append(phrase)
        
        analysis['ai_phrases_found'] = ai_phrase_matches
        analysis['ai_phrase_count'] = len(ai_phrase_matches)
        
        # Count AI word matches
        ai_word_count = sum(1 for w in clean_words if w in self.ai_indicator_words)
        analysis['ai_word_count'] = ai_word_count
        analysis['ai_word_ratio'] = ai_word_count / max(word_count, 1)
        
        # Count contractions
        contraction_count = 0
        for word in words:
            clean = word.lower().strip('.,!?;:"\'')
            if clean in self.contractions:
                contraction_count += 1
        
        analysis['contraction_count'] = contraction_count
        analysis['contraction_ratio'] = contraction_count / max(word_count, 1)
        
        # Count casual expressions
        casual_count = sum(1 for w in clean_words if w in self.casual_expressions)
        analysis['casual_count'] = casual_count
        analysis['casual_ratio'] = casual_count / max(word_count, 1)
        
        # Count personal phrases
        personal_count = sum(1 for phrase in self.personal_phrases if phrase in text_lower)
        analysis['personal_phrase_count'] = personal_count
        
        # Sentence structure
        sentence_lengths = [len(s.split()) for s in sentences]
        
        if sentence_lengths:
            analysis['avg_sentence_length'] = np.mean(sentence_lengths)
            analysis['sentence_length_std'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            analysis['burstiness'] = analysis['sentence_length_std'] / max(analysis['avg_sentence_length'], 1)
        else:
            analysis['avg_sentence_length'] = 0
            analysis['sentence_length_std'] = 0
            analysis['burstiness'] = 0
        
        # Check sentence starters
        ai_starter_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for starter in self.ai_sentence_starters:
                if sentence_lower.startswith(starter):
                    ai_starter_count += 1
                    break
        
        analysis['ai_starter_ratio'] = ai_starter_count / sentence_count
        
        # Punctuation
        analysis['question_marks'] = text.count('?')
        analysis['exclamation_marks'] = text.count('!')
        
        # First person usage
        first_person_words = {'i', 'me', 'my', 'mine', 'myself'}
        first_person_count = sum(1 for w in clean_words if w in first_person_words)
        analysis['first_person_ratio'] = first_person_count / max(word_count, 1)
        
        # Essay structure
        has_intro = any(p in text_lower[:300] for p in ["is one of", "has emerged", "is widely", "is a"])
        has_conclusion = any(p in text_lower[-400:] for p in ["in conclusion", "to conclude", "in summary", "therefore"])
        analysis['has_essay_structure'] = has_intro and has_conclusion
        
        return analysis
    
    def _calculate_ai_score(self, analysis: Dict) -> Tuple[float, List[str]]:
        """Calculate AI probability based on linguistic analysis."""
        
        score = 0.50
        reasons = []
        
        # === AI SIGNALS ===
        
        ai_phrase_count = analysis.get('ai_phrase_count', 0)
        if ai_phrase_count >= 10:
            score += 0.35
            reasons.append(f"Many AI phrases ({ai_phrase_count})")
        elif ai_phrase_count >= 7:
            score += 0.30
            reasons.append(f"Several AI phrases ({ai_phrase_count})")
        elif ai_phrase_count >= 5:
            score += 0.25
            reasons.append(f"Multiple AI phrases ({ai_phrase_count})")
        elif ai_phrase_count >= 3:
            score += 0.18
            reasons.append(f"AI phrases detected ({ai_phrase_count})")
        elif ai_phrase_count >= 2:
            score += 0.10
            reasons.append(f"Some AI phrases ({ai_phrase_count})")
        elif ai_phrase_count >= 1:
            score += 0.05
        
        ai_word_count = analysis.get('ai_word_count', 0)
        ai_word_ratio = analysis.get('ai_word_ratio', 0)
        if ai_word_count >= 12 or ai_word_ratio > 0.05:
            score += 0.18
            reasons.append(f"Heavy AI vocabulary ({ai_word_count} words)")
        elif ai_word_count >= 8 or ai_word_ratio > 0.035:
            score += 0.12
            reasons.append(f"High AI vocabulary ({ai_word_count} words)")
        elif ai_word_count >= 5 or ai_word_ratio > 0.02:
            score += 0.07
            reasons.append(f"AI vocabulary detected")
        elif ai_word_count >= 3:
            score += 0.03
        
        ai_starter_ratio = analysis.get('ai_starter_ratio', 0)
        if ai_starter_ratio > 0.4:
            score += 0.10
            reasons.append("Formal sentence structure")
        elif ai_starter_ratio > 0.25:
            score += 0.05
        
        if analysis.get('has_essay_structure', False):
            score += 0.07
            reasons.append("Formal essay structure")
        
        burstiness = analysis.get('burstiness', 0.5)
        if burstiness < 0.25:
            score += 0.08
            reasons.append("Very uniform writing style")
        elif burstiness < 0.35:
            score += 0.04
        
        word_count = analysis.get('word_count', 0)
        contraction_count = analysis.get('contraction_count', 0)
        if word_count > 150 and contraction_count == 0:
            score += 0.08
            reasons.append("No contractions (formal)")
        elif word_count > 80 and contraction_count == 0:
            score += 0.04
        
        first_person_ratio = analysis.get('first_person_ratio', 0)
        if word_count > 100 and first_person_ratio < 0.005:
            score += 0.05
            reasons.append("Impersonal writing")
        
        # === HUMAN SIGNALS ===
        
        casual_count = analysis.get('casual_count', 0)
        if casual_count >= 5:
            score -= 0.35
            reasons.append(f"Casual language ({casual_count})")
        elif casual_count >= 3:
            score -= 0.25
            reasons.append(f"Informal expressions")
        elif casual_count >= 1:
            score -= 0.12
        
        if contraction_count >= 8:
            score -= 0.22
            reasons.append(f"Heavy contraction use")
        elif contraction_count >= 5:
            score -= 0.16
            reasons.append(f"Natural contractions")
        elif contraction_count >= 3:
            score -= 0.10
        elif contraction_count >= 1:
            score -= 0.05
        
        personal_count = analysis.get('personal_phrase_count', 0)
        if personal_count >= 4:
            score -= 0.22
            reasons.append(f"Personal narrative")
        elif personal_count >= 2:
            score -= 0.14
            reasons.append(f"Personal references")
        elif personal_count >= 1:
            score -= 0.07
        
        if first_person_ratio > 0.05:
            score -= 0.12
            reasons.append("First-person narrative")
        elif first_person_ratio > 0.03:
            score -= 0.07
        
        questions = analysis.get('question_marks', 0)
        exclamations = analysis.get('exclamation_marks', 0)
        
        if questions >= 3:
            score -= 0.08
            reasons.append("Multiple questions")
        elif questions >= 1:
            score -= 0.03
        
        if exclamations >= 3:
            score -= 0.12
            reasons.append("Expressive punctuation")
        elif exclamations >= 1:
            score -= 0.05
        
        if burstiness > 0.6:
            score -= 0.10
            reasons.append("Varied writing style")
        elif burstiness > 0.5:
            score -= 0.05
        
        final_score = max(0.05, min(0.95, score))
        
        return final_score, reasons
    
    def detect(self, text: str, detailed: bool = False) -> Dict:
        """Detect if text is AI-generated."""
        text = text.strip()
        word_count = len(text.split())
        
        if word_count < 15:
            return {
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'verdict': 'uncertain',
                'confidence': 'low',
                'message': 'Text too short for reliable detection'
            }
        
        analysis = self._analyze_text(text)
        linguistic_score, reasons = self._calculate_ai_score(analysis)
        
        if self.transformer_available:
            transformer_score = self._get_transformer_score(text)
        else:
            transformer_score = 0.5
        
        # === SMART WEIGHTING ===
        ai_phrase_count = analysis.get('ai_phrase_count', 0)
        ai_word_count = analysis.get('ai_word_count', 0)
        casual_count = analysis.get('casual_count', 0)
        contraction_count = analysis.get('contraction_count', 0)
        personal_count = analysis.get('personal_phrase_count', 0)
        
        human_signals = casual_count + contraction_count + personal_count
        ai_signals = ai_phrase_count + (ai_word_count // 3)
        
        # Trust linguistic more when signals are clear
        if ai_signals >= 8 and human_signals == 0:
            weight_linguistic = 0.92
        elif human_signals >= 5 and ai_phrase_count <= 1:
            weight_linguistic = 0.92
        elif ai_signals >= 5 and human_signals == 0:
            weight_linguistic = 0.85
        elif human_signals >= 3 and ai_phrase_count <= 2:
            weight_linguistic = 0.85
        elif ai_signals >= 3 and human_signals <= 1:
            weight_linguistic = 0.75
        elif human_signals >= 2:
            weight_linguistic = 0.75
        else:
            weight_linguistic = 0.55
        
        weight_transformer = 1 - weight_linguistic
        ai_probability = weight_linguistic * linguistic_score + weight_transformer * transformer_score
        
        # Floor/ceiling for very clear cases
        if ai_signals >= 6 and human_signals == 0 and linguistic_score > 0.75:
            ai_probability = max(ai_probability, 0.80)
        
        if human_signals >= 4 and ai_phrase_count <= 1 and linguistic_score < 0.30:
            ai_probability = min(ai_probability, 0.20)
        
        human_probability = 1 - ai_probability
        
        # Verdict
        if ai_probability >= 0.80:
            verdict = 'ai_generated'
            confidence = 'high'
        elif ai_probability >= 0.65:
            verdict = 'likely_ai'
            confidence = 'medium'
        elif ai_probability >= 0.40:
            verdict = 'mixed'
            confidence = 'low'
        elif ai_probability >= 0.20:
            verdict = 'likely_human'
            confidence = 'medium'
        else:
            verdict = 'human_written'
            confidence = 'high'
        
        result = {
            'ai_probability': round(ai_probability, 4),
            'human_probability': round(human_probability, 4),
            'ai_percentage': round(ai_probability * 100, 1),
            'human_percentage': round(human_probability * 100, 1),
            'verdict': verdict,
            'confidence': confidence,
            'word_count': word_count,
        }
        
        if detailed:
            result['analysis'] = analysis
            result['linguistic_score'] = round(linguistic_score, 4)
            result['detection_reasons'] = reasons
            if self.transformer_available:
                result['transformer_score'] = round(transformer_score, 4)
                result['weight_linguistic'] = round(weight_linguistic, 2)
            result['ai_phrases_found'] = analysis.get('ai_phrases_found', [])
        
        return result
    
    def get_detailed_report(self, text: str) -> str:
        """Generate detailed report."""
        result = self.detect(text, detailed=True)
        
        lines = [
            "",
            "=" * 65,
            "              AI CONTENT DETECTION REPORT",
            "=" * 65,
            "",
            "📊 VERDICT",
            f"   Result:            {result['verdict'].replace('_', ' ').upper()}",
            f"   AI Probability:    {result['ai_percentage']}%",
            f"   Human Probability: {result['human_percentage']}%",
            f"   Confidence:        {result['confidence'].title()}",
            "",
            f"📝 TEXT STATS",
            f"   Words:             {result['word_count']}",
        ]
        
        if 'analysis' in result:
            a = result['analysis']
            lines.extend([
                f"   Sentences:         {a.get('sentence_count', 0)}",
                f"   Avg sent. length:  {a.get('avg_sentence_length', 0):.1f} words",
                "",
                "🔴 AI SIGNALS DETECTED",
                f"   AI phrases:        {a.get('ai_phrase_count', 0)}",
                f"   AI vocabulary:     {a.get('ai_word_count', 0)} words",
                f"   Formal starters:   {a.get('ai_starter_ratio', 0)*100:.0f}%",
            ])
            
            if result.get('ai_phrases_found'):
                lines.append(f"   Phrases found:")
                for phrase in result['ai_phrases_found'][:8]:
                    lines.append(f"      • \"{phrase}\"")
            
            lines.extend([
                "",
                "🟢 HUMAN SIGNALS DETECTED",
                f"   Contractions:      {a.get('contraction_count', 0)}",
                f"   Casual words:      {a.get('casual_count', 0)}",
                f"   Personal refs:     {a.get('personal_phrase_count', 0)}",
                f"   Questions:         {a.get('question_marks', 0)}",
                f"   Exclamations:      {a.get('exclamation_marks', 0)}",
                f"   Writing variance:  {a.get('burstiness', 0):.2f}",
            ])
        
        if result.get('detection_reasons'):
            lines.extend([
                "",
                "💡 KEY FACTORS",
            ])
            for reason in result['detection_reasons']:
                lines.append(f"   • {reason}")
        
        if 'transformer_score' in result:
            lines.extend([
                "",
                "🤖 MODEL SCORES",
                f"   Linguistic:        {result['linguistic_score']*100:.1f}%",
                f"   Neural model:      {result['transformer_score']*100:.1f}%",
                f"   Linguistic weight: {result.get('weight_linguistic', 0)*100:.0f}%",
                f"   Combined:          {result['ai_percentage']}%",
            ])
        
        lines.extend([
            "",
            "=" * 65,
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    detector = ModernAIDetector()
    
    test_text = """Education is one of the most important pillars of human civilization and plays a vital role in the overall development of individuals and societies. In conclusion, education is an indispensable element of human development."""
    
    print(detector.get_detailed_report(test_text))