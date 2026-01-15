#!/usr/bin/env python3
"""
Test script to verify ML model integration in the KOL analyzer.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ml_models_manager():
    """Test the ML models manager."""
    print("=" * 60)
    print("Testing ML Models Manager")
    print("=" * 60)

    try:
        from analysis.ml_models import (
            get_available_models,
            is_model_available,
            analyze_sentiment,
            get_embeddings,
            analyze_toxicity,
            analyze_text_nlp,
            classify_zero_shot
        )

        print("\n1. Checking model availability...")
        models = get_available_models()
        for name, available in models.items():
            status = "✓ Available" if available else "✗ Not loaded yet"
            print(f"   {name}: {status}")

        print("\n2. Testing sentiment analysis...")
        test_text = "Bitcoin is going to the moon! This is amazing!"
        result = analyze_sentiment(test_text)
        if result:
            print(f"   Text: {test_text[:50]}...")
            print(f"   Sentiment: {result}")
        else:
            print("   Sentiment model not available (will use fallback)")

        print("\n3. Testing embeddings...")
        texts = ["I love crypto", "I hate crypto"]
        embeddings = get_embeddings(texts)
        if embeddings is not None:
            print(f"   Generated embeddings for {len(texts)} texts")
            print(f"   Embedding shape: {embeddings.shape}")
        else:
            print("   Embedding model not available (will use fallback)")

        print("\n4. Testing toxicity analysis...")
        toxic_text = "You're an idiot if you buy this scam"
        result = analyze_toxicity(toxic_text)
        if result:
            print(f"   Text: {toxic_text}")
            print(f"   Toxicity scores: {result}")
        else:
            print("   Detoxify model not available (will use fallback)")

        print("\n5. Testing SpaCy NLP...")
        nlp_text = "Elon Musk announced Bitcoin support for Tesla in 2021."
        result = analyze_text_nlp(nlp_text)
        if result:
            print(f"   Text: {nlp_text}")
            print(f"   Entities: {result.get('entities', [])}")
            print(f"   Sentence count: {result.get('sentence_count', 0)}")
        else:
            print("   SpaCy model not available (will use fallback)")

        print("\n6. Testing zero-shot classification...")
        zs_text = "Check out this new token, guaranteed 100x returns!"
        labels = ["promotional content", "educational content", "personal opinion"]
        result = classify_zero_shot(zs_text, labels)
        if result:
            print(f"   Text: {zs_text}")
            print(f"   Classification: {result}")
        else:
            print("   Zero-shot classifier not available (will use fallback)")

        print("\n" + "=" * 60)
        print("ML Models Manager Test Complete")
        print("=" * 60)

    except Exception as e:
        print(f"Error testing ML models manager: {e}")
        import traceback
        traceback.print_exc()


def test_enhanced_analyzers():
    """Test the enhanced analyzers."""
    print("\n" + "=" * 60)
    print("Testing Enhanced Analyzers")
    print("=" * 60)

    # Sample tweets for testing
    sample_tweets = [
        {
            "id": "1",
            "text": "Bitcoin to $100k! This is guaranteed, trust me. 🚀🚀🚀",
            "timestamp": "2024-01-01T12:00:00Z",
            "likes": 100,
            "retweets": 50
        },
        {
            "id": "2",
            "text": "I've been holding BTC since 2017. Diamond hands forever! 💎🙌",
            "timestamp": "2024-01-02T12:00:00Z",
            "likes": 200,
            "retweets": 80
        },
        {
            "id": "3",
            "text": "Just sold all my Bitcoin. This market is dead. Time to move on.",
            "timestamp": "2024-01-10T12:00:00Z",
            "likes": 50,
            "retweets": 20
        },
        {
            "id": "4",
            "text": "If you're not buying the dip, you're an idiot. NGMI.",
            "timestamp": "2024-01-15T12:00:00Z",
            "likes": 300,
            "retweets": 100
        },
        {
            "id": "5",
            "text": "NFA but you should definitely buy $SOL right now. Last chance!",
            "timestamp": "2024-01-20T12:00:00Z",
            "likes": 150,
            "retweets": 60
        }
    ]

    try:
        # Test AssholeAnalyzer
        print("\n1. Testing AssholeAnalyzer (with Detoxify)...")
        from analysis.asshole_analyzer import AssholeAnalyzer
        analyzer = AssholeAnalyzer(use_ml=True)
        result = analyzer.analyze(sample_tweets, "test_user")
        print(f"   Asshole Score: {result.asshole_score:.1f}")
        print(f"   Toxicity Level: {result.toxicity_level} {result.toxicity_emoji}")
        print(f"   ML Available: {result.ml_available}")
        if result.ml_available:
            print(f"   ML Toxicity: {result.ml_toxicity_score:.1f}")

        # Test ConsistencyTracker
        print("\n2. Testing ConsistencyTracker (with RoBERTa sentiment)...")
        from analysis.consistency_tracker import ConsistencyTracker
        tracker = ConsistencyTracker(use_ml=True)
        result = tracker.analyze(sample_tweets)
        print(f"   Consistency Score: {result.consistency_score:.1f}")
        print(f"   Flip Count: {result.flip_count}")
        print(f"   Topics Tracked: {result.topics_tracked}")

        # Test ContradictionAnalyzer
        print("\n3. Testing ContradictionAnalyzer (with embeddings)...")
        from analysis.contradiction_analyzer import ContradictionAnalyzer
        analyzer = ContradictionAnalyzer(use_ml=True)
        result = analyzer.analyze(sample_tweets, "test_user")
        print(f"   BS Score: {result.bs_score:.1f}")
        print(f"   Contradiction Count: {result.contradiction_count}")
        print(f"   ML Available: {result.ml_available}")
        if result.ml_available:
            print(f"   Semantic Contradictions: {result.semantic_contradictions}")

        # Test LinguisticAnalyzer
        print("\n4. Testing LinguisticAnalyzer (with SpaCy)...")
        from analysis.linguistic_analyzer import LinguisticAnalyzer
        analyzer = LinguisticAnalyzer(use_ml=True)
        result = analyzer.analyze(sample_tweets)
        print(f"   Authenticity Score: {result.authenticity_score:.1f}")
        print(f"   Manipulation Score: {result.manipulation_score:.1f}")
        print(f"   ML Available: {result.ml_available}")
        if result.ml_available:
            print(f"   Key Topics: {result.key_topics[:5]}")

        # Test ArchetypeClassifier
        print("\n5. Testing ArchetypeClassifier (with zero-shot)...")
        from analysis.archetype_classifier import ArchetypeClassifier
        classifier = ArchetypeClassifier(use_ml=True)
        tweet_texts = [t['text'] for t in sample_tweets]
        result = classifier.classify(
            engagement_score=70,
            consistency_score=60,
            dissonance_score=50,
            baiting_score=40,
            privilege_score=55,
            prediction_score=50,
            transparency_score=60,
            follower_quality_score=70,
            follower_count=10000,
            tweet_count=len(sample_tweets),
            tweet_texts=tweet_texts
        )
        print(f"   Primary Archetype: {result.primary_archetype.value}")
        print(f"   Confidence: {result.confidence:.1f}")
        print(f"   ML Available: {result.ml_available}")
        if result.ml_available:
            print(f"   Content Themes: {result.content_themes}")

        print("\n" + "=" * 60)
        print("Enhanced Analyzers Test Complete")
        print("=" * 60)

    except Exception as e:
        print(f"Error testing enhanced analyzers: {e}")
        import traceback
        traceback.print_exc()


def test_credibility_engine():
    """Test the full credibility engine."""
    print("\n" + "=" * 60)
    print("Testing Full Credibility Engine")
    print("=" * 60)

    sample_tweets = [
        {
            "id": str(i),
            "text": f"Test tweet {i} with some crypto content about $BTC and market analysis",
            "timestamp": f"2024-01-{i+1:02d}T12:00:00Z",
            "likes": 100 + i * 10,
            "retweets": 50 + i * 5
        }
        for i in range(10)
    ]

    try:
        from analysis.credibility_engine import CredibilityEngine

        print("\n1. Testing with ML enabled...")
        engine = CredibilityEngine(use_ml=True)
        result = engine.analyze(
            tweets=sample_tweets,
            follower_count=10000,
            username="test_user",
            account_age_days=365
        )
        print(f"   Overall Score: {result.overall_score:.1f}")
        print(f"   Grade: {result.grade}")
        print(f"   Archetype: {result.archetype}")

        print("\n2. Testing with ML disabled (fallback)...")
        engine = CredibilityEngine(use_ml=False)
        result = engine.analyze(
            tweets=sample_tweets,
            follower_count=10000,
            username="test_user",
            account_age_days=365
        )
        print(f"   Overall Score: {result.overall_score:.1f}")
        print(f"   Grade: {result.grade}")
        print(f"   Archetype: {result.archetype}")

        print("\n" + "=" * 60)
        print("Credibility Engine Test Complete")
        print("=" * 60)

    except Exception as e:
        print(f"Error testing credibility engine: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KOL Analyzer ML Integration Test")
    print("=" * 60)

    test_ml_models_manager()
    test_enhanced_analyzers()
    test_credibility_engine()

    print("\n" + "=" * 60)
    print("All Tests Complete!")
    print("=" * 60)
