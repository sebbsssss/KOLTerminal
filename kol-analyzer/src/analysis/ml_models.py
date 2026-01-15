"""
ML Models Manager - Lazy loading and caching for ML models.

Provides centralized access to:
- Twitter RoBERTa sentiment model
- All-MiniLM-L6-v2 sentence embeddings
- Detoxify toxicity detection
- SpaCy NLP pipeline
- Zero-shot classification
"""

import os
from typing import Optional, List, Dict, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Global model instances (lazy loaded)
_sentiment_model = None
_sentiment_tokenizer = None
_embedding_model = None
_detoxify_model = None
_spacy_nlp = None
_zero_shot_classifier = None

# Flag to track if models are available
_models_available = {
    'sentiment': None,
    'embeddings': None,
    'detoxify': None,
    'spacy': None,
    'zero_shot': None
}


def _check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def get_sentiment_model():
    """
    Get the Twitter RoBERTa sentiment model.
    Model: cardiffnlp/twitter-roberta-base-sentiment-latest

    Returns:
        Tuple of (model, tokenizer) or (None, None) if not available
    """
    global _sentiment_model, _sentiment_tokenizer, _models_available

    if _models_available['sentiment'] is False:
        return None, None

    if _sentiment_model is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            logger.info(f"Loading sentiment model: {model_name}")

            _sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _sentiment_model.eval()

            _models_available['sentiment'] = True
            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
            _models_available['sentiment'] = False
            return None, None

    return _sentiment_model, _sentiment_tokenizer


def get_embedding_model():
    """
    Get the sentence embedding model.
    Model: sentence-transformers/all-MiniLM-L6-v2

    Returns:
        SentenceTransformer model or None if not available
    """
    global _embedding_model, _models_available

    if _models_available['embeddings'] is False:
        return None

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")

            _embedding_model = SentenceTransformer(model_name)

            _models_available['embeddings'] = True
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            _models_available['embeddings'] = False
            return None

    return _embedding_model


def get_detoxify_model():
    """
    Get the Detoxify toxicity detection model.

    Returns:
        Detoxify model or None if not available
    """
    global _detoxify_model, _models_available

    if _models_available['detoxify'] is False:
        return None

    if _detoxify_model is None:
        try:
            from detoxify import Detoxify

            logger.info("Loading Detoxify model")

            _detoxify_model = Detoxify('original')

            _models_available['detoxify'] = True
            logger.info("Detoxify model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load Detoxify model: {e}")
            _models_available['detoxify'] = False
            return None

    return _detoxify_model


def get_spacy_nlp():
    """
    Get the SpaCy NLP pipeline.
    Model: en_core_web_sm

    Returns:
        SpaCy Language model or None if not available
    """
    global _spacy_nlp, _models_available

    if _models_available['spacy'] is False:
        return None

    if _spacy_nlp is None:
        try:
            import spacy

            logger.info("Loading SpaCy model: en_core_web_sm")

            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Try to download if not available
                logger.info("SpaCy model not found, attempting download...")
                from spacy.cli import download
                download("en_core_web_sm")
                _spacy_nlp = spacy.load("en_core_web_sm")

            _models_available['spacy'] = True
            logger.info("SpaCy model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load SpaCy model: {e}")
            _models_available['spacy'] = False
            return None

    return _spacy_nlp


def get_zero_shot_classifier():
    """
    Get the zero-shot classification pipeline.
    Model: facebook/bart-large-mnli

    Returns:
        Transformers pipeline or None if not available
    """
    global _zero_shot_classifier, _models_available

    if _models_available['zero_shot'] is False:
        return None

    if _zero_shot_classifier is None:
        try:
            from transformers import pipeline

            logger.info("Loading zero-shot classification pipeline")

            _zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )

            _models_available['zero_shot'] = True
            logger.info("Zero-shot classifier loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load zero-shot classifier: {e}")
            _models_available['zero_shot'] = False
            return None

    return _zero_shot_classifier


# Utility functions for common operations

def analyze_sentiment(text: str) -> Optional[Dict[str, float]]:
    """
    Analyze sentiment of text using Twitter RoBERTa.

    Args:
        text: Text to analyze

    Returns:
        Dict with 'positive', 'negative', 'neutral' scores or None
    """
    model, tokenizer = get_sentiment_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch

        # Truncate text to model max length
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]

        # Model outputs: negative, neutral, positive
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2])
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return None


def analyze_sentiment_batch(texts: List[str]) -> List[Optional[Dict[str, float]]]:
    """
    Analyze sentiment of multiple texts.

    Args:
        texts: List of texts to analyze

    Returns:
        List of sentiment dicts
    """
    model, tokenizer = get_sentiment_model()
    if model is None or tokenizer is None:
        return [None] * len(texts)

    try:
        import torch

        results = []
        # Process in batches of 32
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)

            for score in scores:
                results.append({
                    'negative': float(score[0]),
                    'neutral': float(score[1]),
                    'positive': float(score[2])
                })

        return results
    except Exception as e:
        logger.error(f"Batch sentiment analysis error: {e}")
        return [None] * len(texts)


def get_embeddings(texts: List[str]) -> Optional[Any]:
    """
    Get sentence embeddings for texts.

    Args:
        texts: List of texts to embed

    Returns:
        Numpy array of embeddings or None
    """
    model = get_embedding_model()
    if model is None:
        return None

    try:
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def compute_similarity(text1: str, text2: str) -> Optional[float]:
    """
    Compute cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1) or None
    """
    embeddings = get_embeddings([text1, text2])
    if embeddings is None:
        return None

    try:
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        return float(similarity)
    except Exception as e:
        logger.error(f"Similarity computation error: {e}")
        return None


def analyze_toxicity(text: str) -> Optional[Dict[str, float]]:
    """
    Analyze toxicity of text using Detoxify.

    Args:
        text: Text to analyze

    Returns:
        Dict with toxicity scores or None
    """
    model = get_detoxify_model()
    if model is None:
        return None

    try:
        results = model.predict(text)
        return {k: float(v) for k, v in results.items()}
    except Exception as e:
        logger.error(f"Toxicity analysis error: {e}")
        return None


def analyze_toxicity_batch(texts: List[str]) -> List[Optional[Dict[str, float]]]:
    """
    Analyze toxicity of multiple texts.

    Args:
        texts: List of texts to analyze

    Returns:
        List of toxicity dicts
    """
    model = get_detoxify_model()
    if model is None:
        return [None] * len(texts)

    try:
        results = model.predict(texts)
        # Convert from dict of lists to list of dicts
        batch_results = []
        for i in range(len(texts)):
            batch_results.append({
                k: float(v[i]) if hasattr(v, '__iter__') else float(v)
                for k, v in results.items()
            })
        return batch_results
    except Exception as e:
        logger.error(f"Batch toxicity analysis error: {e}")
        return [None] * len(texts)


def extract_entities(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Extract named entities using SpaCy.

    Args:
        text: Text to analyze

    Returns:
        List of entity dicts with 'text', 'label' or None
    """
    nlp = get_spacy_nlp()
    if nlp is None:
        return None

    try:
        doc = nlp(text)
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return None


def analyze_text_nlp(text: str) -> Optional[Dict[str, Any]]:
    """
    Comprehensive NLP analysis using SpaCy.

    Args:
        text: Text to analyze

    Returns:
        Dict with various NLP metrics or None
    """
    nlp = get_spacy_nlp()
    if nlp is None:
        return None

    try:
        doc = nlp(text)

        # Count POS tags
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        # Get entities
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

        # Sentence-level analysis
        sentences = list(doc.sents)
        avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))

        # Noun chunks (key phrases)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        return {
            'pos_counts': pos_counts,
            'entities': entities,
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'noun_chunks': noun_chunks[:20],  # Limit to top 20
            'token_count': len(doc)
        }
    except Exception as e:
        logger.error(f"NLP analysis error: {e}")
        return None


def classify_zero_shot(text: str, labels: List[str]) -> Optional[Dict[str, float]]:
    """
    Zero-shot classification of text.

    Args:
        text: Text to classify
        labels: List of candidate labels

    Returns:
        Dict mapping labels to scores or None
    """
    classifier = get_zero_shot_classifier()
    if classifier is None:
        return None

    try:
        result = classifier(text, labels)
        return dict(zip(result['labels'], result['scores']))
    except Exception as e:
        logger.error(f"Zero-shot classification error: {e}")
        return None


def classify_zero_shot_batch(
    texts: List[str],
    labels: List[str]
) -> List[Optional[Dict[str, float]]]:
    """
    Zero-shot classification of multiple texts.

    Args:
        texts: List of texts to classify
        labels: List of candidate labels

    Returns:
        List of classification dicts
    """
    classifier = get_zero_shot_classifier()
    if classifier is None:
        return [None] * len(texts)

    try:
        results = []
        for text in texts:
            result = classifier(text, labels)
            results.append(dict(zip(result['labels'], result['scores'])))
        return results
    except Exception as e:
        logger.error(f"Batch zero-shot classification error: {e}")
        return [None] * len(texts)


def is_model_available(model_name: str) -> bool:
    """
    Check if a specific model is available.

    Args:
        model_name: One of 'sentiment', 'embeddings', 'detoxify', 'spacy', 'zero_shot'

    Returns:
        True if model is available
    """
    if model_name not in _models_available:
        return False

    # If we haven't tried loading yet, try now
    if _models_available[model_name] is None:
        if model_name == 'sentiment':
            get_sentiment_model()
        elif model_name == 'embeddings':
            get_embedding_model()
        elif model_name == 'detoxify':
            get_detoxify_model()
        elif model_name == 'spacy':
            get_spacy_nlp()
        elif model_name == 'zero_shot':
            get_zero_shot_classifier()

    return _models_available.get(model_name, False)


def get_available_models() -> Dict[str, bool]:
    """
    Get availability status of all models.

    Returns:
        Dict mapping model names to availability status
    """
    return {k: v if v is not None else False for k, v in _models_available.items()}


def preload_models(models: Optional[List[str]] = None):
    """
    Preload specified models (or all if not specified).

    Args:
        models: List of model names to preload, or None for all
    """
    if models is None:
        models = ['sentiment', 'embeddings', 'detoxify', 'spacy', 'zero_shot']

    for model_name in models:
        logger.info(f"Preloading model: {model_name}")
        is_model_available(model_name)
