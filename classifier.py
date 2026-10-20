"""
classifier.py

Confidence-aware selective email classifier for the
Personalized AI Email Triage System.

The classifier:

1. Extracts interpretable email features.
2. Calculates the personalized relevance score.
3. Estimates routing confidence.
4. Applies important-email safeguards.
5. Routes the email to:
       PRIORITY
       REVIEW
       SPAM
6. Generates a human-readable explanation.

IMPORTANT:
REVIEW acts as the safe/default abstention state when the system
does not have enough evidence for a confident Priority or Spam decision.
"""

from config import (
    PRIORITY,
    REVIEW,
    SPAM,
    RELEVANCE_WEIGHTS,
    SPAM_MIN_SPAM_SCORE,
    SPAM_MIN_CONFIDENCE,
    PRIORITY_MIN_RELEVANCE,
    PRIORITY_MIN_CONFIDENCE,
    SAFEGUARD_SENDER_IMPORTANCE,
    SAFEGUARD_PERSONAL_RELEVANCE,
)

from feature_extraction import extract_features


# ==========================================================================
# 1. RELEVANCE SCORE
# ==========================================================================

def calculate_relevance(features):
    """
    Calculate personalized relevance using the configured weighted model.

    R(email) =
        w1 * personal_relevance
        + w2 * sender_importance
        + w3 * semantic_relevance
        + w4 * urgency
        + w5 * action_required
        + w6 * historical_preference
    """

    relevance = 0.0

    for feature_name, weight in RELEVANCE_WEIGHTS.items():
        relevance += features.get(feature_name, 0.0) * weight

    return max(0.0, min(1.0, relevance))


# ==========================================================================
# 2. PRIORITY CONFIDENCE
# ==========================================================================

def calculate_priority_confidence(features, relevance_score):
    """
    Estimate confidence that an email belongs in Priority.

    This is an initial transparent confidence approximation.

    Later, this function can be replaced by calibrated probabilities
    from a trained machine-learning model.
    """


    confidence = (
        0.15 * relevance_score
        + 0.10 * features["personal_relevance"]
        + 0.25 * features["sender_importance"]
        + 0.10 * features["semantic_relevance"]
        + 0.20 * features["urgency"]
        + 0.20 * features["action_required"]
    )

    return max(0.0, min(1.0, confidence))

             




# ==========================================================================
# 3. SPAM CONFIDENCE
# ==========================================================================

def calculate_spam_confidence(features):
    """
    Estimate confidence in a Spam decision.

    For the first prototype, spam confidence is closely tied to the
    strength of spam evidence.

    This is intentionally simple and will later be replaced/calibrated
    using labelled data.
    """

    spam = features["spam_score"]

    # Strong spam evidence produces higher confidence.
    if spam >= 0.90:
        confidence = 0.95

    elif spam >= 0.75:
        confidence = 0.80

    elif spam >= 0.50:
        confidence = 0.65

    else:
        confidence = 0.50

    return confidence


# ==========================================================================
# 4. IMPORTANT-EMAIL SAFEGUARD
# ==========================================================================

def spam_safeguard_triggered(features):
    """
    Return True when strong personal/sender evidence should prevent
    direct routing to Spam.

    Such messages are sent to Review instead.
    """

    sender_is_important = (
        features["sender_importance"]
        >= SAFEGUARD_SENDER_IMPORTANCE
    )

    personally_relevant = (
        features["personal_relevance"]
        >= SAFEGUARD_PERSONAL_RELEVANCE
    )

    return sender_is_important or personally_relevant


# ==========================================================================
# 5. EXPLANATION GENERATOR
# ==========================================================================

def generate_explanation(
    category,
    features,
    relevance_score,
    priority_confidence,
    spam_confidence,
    safeguard=False,
):
    """
    Produce a human-readable explanation of the routing decision.
    """

    reasons = []

    if category == PRIORITY:

        if features["personal_relevance"] >= 0.50:
            reasons.append(
                "The email strongly matches the user's interests "
                "or important topics."
            )

        if features["sender_importance"] >= 0.60:
            reasons.append(
                "The sender or sender domain is considered important."
            )

        if features["urgency"] >= 0.50:
            reasons.append(
                "Time-sensitive or deadline-related language was detected."
            )

        if features["action_required"] >= 0.50:
            reasons.append(
                "The message appears to require user action."
            )

        if not reasons:
            reasons.append(
                "The combined relevance evidence is sufficiently strong."
            )

    elif category == SPAM:

        reasons.append(
            "The message has very strong spam/junk indicators."
        )

        if features["personal_relevance"] < 0.50:
            reasons.append(
                "The message has low personal relevance."
            )

        if features["sender_importance"] < 0.60:
            reasons.append(
                "The sender is not identified as important."
            )

    else:

                if safeguard and features.get("ai_spam_probability", 0.0) >= 0.50:
                 reasons.append(
                "Spam-like characteristics were detected, but important "
                "sender or personal-relevance evidence prevented direct "
                "routing to Spam."
            )

                elif relevance_score < PRIORITY_MIN_RELEVANCE:
                 reasons.append(
                "The email does not have enough relevance evidence "
                "for Priority."
            )

                elif priority_confidence < PRIORITY_MIN_CONFIDENCE:
                 reasons.append(
                "The system is not sufficiently confident to place "
                "the email in Priority."
            )

                else:
                 reasons.append(
                "The available evidence is uncertain, so Review is "
                "used asdef spam_safeguard_triggered the safe destination."
            )

    return " ".join(reasons)


# ==========================================================================
# 6. ROUTING
# ==========================================================================

def route_email(features):
    """
    Route one email using confidence-aware selective triage.

    Decision order:

    1. Check high-confidence Spam.
    2. Apply important-email safeguards.
    3. Check high-confidence Priority.
    4. Otherwise abstain to Review.
    """

    relevance_score = calculate_relevance(features)

    priority_confidence = calculate_priority_confidence(
        features,
        relevance_score
    )

    # DistilBERT spam probability
    ai_spam_probability = features["ai_spam_probability"]

     # Retain the rule-based spam score for comparison
    rule_spam_score = features["spam_score"]
     # Use the trained model as the main spam signal
    spam_confidence = ai_spam_probability

    safeguard = spam_safeguard_triggered(features)

    # If AI and rule-based evidence strongly disagree,
    # send the email for human review.
    conflicting_evidence = (
        ai_spam_probability >= 0.90
        and rule_spam_score <= 0.20
    )

    # ------------------------------------------------------------------
        # SPAM / REVIEW / PRIORITY DECISION
        
    
    # MiniLM-first, personalized AI routing.
    semantic = features.get("semantic_priority", 0.0)
    personal = features.get("personal_relevance", 0.0)
    urgency = features.get("urgency", 0.0)
    action = features.get("action_required", 0.0)
    sender = features.get("sender_importance", 0.0)

    # MiniLM is the primary importance signal.
    strong_ai_priority = semantic >= 0.60

    contextual_ai_priority = (
        semantic >= 0.50
        and personal >= 0.25
        and (urgency >= 0.40 or action >= 0.40)
    )

    personalized_priority = sender >= 0.90

    if (
        ai_spam_probability >= 0.90
        and not safeguard
        and not conflicting_evidence
    ):
        category = SPAM

    elif ai_spam_probability >= 0.50 or conflicting_evidence:
        category = REVIEW

    elif (
        ai_spam_probability < 0.20
        and (
            strong_ai_priority
            or contextual_ai_priority
            or personalized_priority
        )
    ):
        category = PRIORITY

    else:
        category = REVIEW

    
   

    # ------------------------------------------------------------------
    # REVIEW / ABSTENTION
    # ------------------------------------------------------------------

    

    explanation = generate_explanation(
        category=category,
        features=features,
        relevance_score=relevance_score,
        priority_confidence=priority_confidence,
        spam_confidence=spam_confidence,
        safeguard=safeguard,
    )

    return {
        "category": category,
        "relevance_score": relevance_score,
        "priority_confidence": priority_confidence,
        "spam_score": features["spam_score"],
        "spam_confidence": spam_confidence,
        "safeguard_triggered": safeguard,
        "features": features,
        "explanation": explanation,
    }


# ==========================================================================
# 7. COMPLETE CLASSIFICATION FUNCTION
# ==========================================================================

def classify_email(email, profile_id):
    """
    Complete classification pipeline for one email.

    Email
        ->
    Feature extraction
        ->
    Personalized relevance
        ->
    Confidence estimation
        ->
    Safety rules
        ->
    Priority / Review / Spam
    """

    features = extract_features(
        email=email,
        profile_id=profile_id
    )

    result = route_email(features)

    result["profile_id"] = profile_id

    return result


# ==========================================================================
# 8. DISPLAY FUNCTION
# ==========================================================================

def display_result(email, result):
    """Print a readable classification result."""

    print("=" * 70)

    print(f"FROM:    {email.get('sender', '')}")
    print(f"SUBJECT: {email.get('subject', '')}")

    print("-" * 70)

    print(
        f"CATEGORY:            "
        f"{result['category'].upper()}"
    )

    print(
        f"Relevance score:     "
        f"{result['relevance_score']:.2f}"
    )

    print(
        f"Priority confidence: "
        f"{result['priority_confidence']:.2f}"
    )

    print(
        f"Spam score:          "
        f"{result['spam_score']:.2f}"
    )

    print(
        f"Spam confidence:     "
        f"{result['spam_confidence']:.2f}"
    )

    print(
        f"Safeguard triggered: "
        f"{result['safeguard_triggered']}"
    )

    print("\nFEATURES")

    for name, value in result["features"].items():

        if name != "spam_score":
            print(
                f"  {name:25s}: "
                f"{value:.2f}"
            )

    print("\nWHY?")

    print(result["explanation"])

    print("=" * 70)


# ==========================================================================
# 9. MANUAL TEST
# ==========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Test 1: Important academic email
    # ------------------------------------------------------------------

    academic_email = {
        "sender": "professor@college.edu",
        "subject": "Assignment Deadline",
        "body":
            "Submit the machine learning assignment "
            "before tomorrow 5 PM."
    }

    print("\nTEST 1 - Academic email\n")

    result = classify_email(
        academic_email,
        "ananya_ai"
    )

    display_result(
        academic_email,
        result
    )


    # ------------------------------------------------------------------
    # Test 2: Potential spam
    # ------------------------------------------------------------------

    spam_email = {
        "sender": "winner@randomprize.com",
        "subject": "CONGRATULATIONS YOU WON",
        "body":
            "You have won a huge prize! "
            "Click immediately and provide your bank details. "
            "ACT NOW!!!"
    }

    print("\nTEST 2 - Suspicious email\n")

    result = classify_email(
        spam_email,
        "ananya_ai"
    )

    display_result(
        spam_email,
        result
    )


    # ------------------------------------------------------------------
    # Test 3: Potentially useful but uncertain email
    # ------------------------------------------------------------------

    workshop_email = {
        "sender": "events@ieee.org",
        "subject": "Generative AI Workshop",
        "body":
            "IEEE workshop on Generative AI. "
            "Registration closes Friday."
    }

    print("\nTEST 3 - Personalized workshop email\n")

    print("\nANANYA:\n")

    result = classify_email(
        workshop_email,
        "ananya_ai"
    )

    display_result(
        workshop_email,
        result
    )

    print("\nROHAN:\n")

    result = classify_email(
        workshop_email,
        "rohan_mech"
    )

    display_result(
        workshop_email,
        result
    )