"""
config.py
---------

Central configuration for the Personalized, Adaptive and
Uncertainty-Aware AI Email Triage System.

This is the ONE place for:
    - paths
    - labels
    - relevance weights
    - routing thresholds
    - safeguards
    - asymmetric error costs
    - adaptive-learning parameters
    - confidence/calibration settings
    - experimental settings
    - evaluation metrics

IMPORTANT
---------
The initial weights and thresholds below are expert-defined starting values.
They are NOT claimed to be optimal.

During experimental evaluation:
    1. Train the model using the training set.
    2. Tune thresholds/calibration using the validation set.
    3. Freeze all parameters.
    4. Evaluate the final system on the test set.

The final README/research paper should report:
    - initial values
    - tuning procedure
    - final selected values
    - experimental evidence supporting those values

No other module should contain unexplained "magic numbers".
"""

from pathlib import Path


# ==========================================================================
# 1. PATHS
# ==========================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DB_PATH = DATA_DIR / "email_triage.db"
SAMPLE_EMAILS_CSV = DATA_DIR / "sample_emails.csv"
DEMO_PROFILES_JSON = DATA_DIR / "demo_profiles.json"


# ==========================================================================
# 2. EMAIL TRIAGE CATEGORIES
# ==========================================================================

PRIORITY = "priority"
REVIEW = "review"
SPAM = "spam"

CATEGORIES = (
    PRIORITY,
    REVIEW,
    SPAM,
)


# ==========================================================================
# 3. RELEVANCE MODEL
# ==========================================================================
#
# Overall relevance:
#
# R(email) =
#       w1 * PersonalRelevance
#     + w2 * SenderImportance
#     + w3 * SemanticRelevance
#     + w4 * Urgency
#     + w5 * ActionRequired
#     + w6 * HistoricalPreference
#
# All component scores are normalized to 0..1.
#
# The weights sum to 1.0 so that the final relevance score also remains
# interpretable in the range 0..1.
#
# IMPORTANT:
#
# SpamScore and ClassificationConfidence are deliberately NOT included
# in the relevance equation.
#
# Relevance answers:
#     "How useful/important is this email to THIS user?"
#
# SpamScore answers:
#     "How likely is this email to be spam/junk/scam?"
#
# Confidence answers:
#     "How certain is the system about its routing decision?"
#
# Keeping these concepts separate is central to the proposed architecture.
# ==========================================================================

RELEVANCE_WEIGHTS = {
    "personal_relevance": 0.25,
    "sender_importance": 0.20,
    "semantic_relevance": 0.15,
    "urgency": 0.15,
    "action_required": 0.15,
    "historical_preference": 0.10,
}


# ==========================================================================
# 4. ROUTING THRESHOLDS
# ==========================================================================
#
# The system implements CONFIDENCE-AWARE SELECTIVE TRIAGE.
#
# REVIEW is not simply an ordinary third prediction class.
#
# It also acts as an ABSTENTION / SAFETY state.
#
# If there is insufficient evidence for a safe Priority or Spam decision,
# the system deliberately routes the email to REVIEW.
#
# These are INITIAL expert-defined values.
# They will later be tuned using the validation dataset.
# ==========================================================================


# --------------------------------------------------------------------------
# Spam routing
# --------------------------------------------------------------------------
#
# Spam is deliberately difficult to reach.
#
# BOTH spam probability and spam confidence must be very high.
#

SPAM_MIN_SPAM_SCORE = 0.90
SPAM_MIN_CONFIDENCE = 0.90


# --------------------------------------------------------------------------
# Priority routing
# --------------------------------------------------------------------------
#
# Priority requires:
#     1. high relevance
#     2. sufficient confidence
#

PRIORITY_MIN_RELEVANCE = 0.65
PRIORITY_MIN_CONFIDENCE = 0.65


# ==========================================================================
# 5. IMPORTANT-EMAIL SAFEGUARDS
# ==========================================================================
#
# Strong evidence of personal importance blocks direct Spam routing.
#
# Example:
#
# A suspicious-looking email from an important university domain should
# preferably go to REVIEW rather than directly to SPAM.
#
# This helps minimize the most costly error:
#
#               IMPORTANT EMAIL -> SPAM
#
# ==========================================================================

SAFEGUARD_SENDER_IMPORTANCE = 0.60
SAFEGUARD_PERSONAL_RELEVANCE = 0.50


# ==========================================================================
# 6. ASYMMETRIC ERROR COSTS
# ==========================================================================
#
# Different mistakes have different real-world consequences.
#
# An important email hidden in Spam is much worse than an important email
# temporarily appearing in Review.
#
# Conceptually:
#
# Cost(T) =
#
#       10 * ImportantToSpam
#     +  1 * ImportantToReview
#     +  3 * JunkToPriority
#     +  1 * OtherErrors
#
# During Phase 5/9, candidate threshold combinations should be evaluated
# against this cost function.
#
# Final thresholds should be selected on the VALIDATION dataset.
# ==========================================================================

ERROR_COSTS = {
    "important_sent_to_spam": 10.0,
    "important_sent_to_review": 1.0,
    "junk_sent_to_priority": 3.0,
    "other": 1.0,
}


# ==========================================================================
# 7. ADAPTIVE LEARNING / USER FEEDBACK
# ==========================================================================
#
# User feedback gradually changes sender/topic preferences.
#
# A single click should NOT significantly change future routing.
# Repeated consistent behaviour provides stronger evidence.
#
# Example:
#
#   AI workshop       Review -> Priority
#   ML conference     Review -> Priority
#   IEEE AI event     Review -> Priority
#
# After repeated behaviour, similar AI-related messages may receive a
# modest relevance increase.
#
# Preference adjustment is capped so that historical behaviour cannot
# completely dominate the other relevance components.
# ==========================================================================

FEEDBACK_STEP = 0.05

MAX_PREFERENCE_ADJUSTMENT = 0.30

# Require repeated evidence before a learned preference influences
# future classification.
MIN_FEEDBACK_EVENTS = 3


# ==========================================================================
# 8. CONFIDENCE CALIBRATION
# ==========================================================================
#
# Raw model probability should not automatically be interpreted as reliable
# confidence.
#
# The prototype may initially use model probabilities.
#
# The research version should compare calibrated and uncalibrated
# confidence estimates.
#
# Supported methods:
#
#     none
#     isotonic
#     sigmoid
#
# ==========================================================================

CONFIDENCE_CALIBRATION = "isotonic"

SUPPORTED_CALIBRATION_METHODS = (
    "none",
    "isotonic",
    "sigmoid",
)


# ==========================================================================
# 9. FEATURE EXTRACTION
# ==========================================================================

# Shortest brand/domain stem checked for look-alike domains.
#
# Example:
#     "ieee"
#
MIN_DOMAIN_STEM_LENGTH = 4


# Number of characters to inspect before a credential-related phrase
# when looking for negation.
#
# Example:
#
#     "We will NEVER ask for your password."
#
NEGATION_WINDOW_CHARS = 40


# ==========================================================================
# 10. DEMONSTRATION PROFILES
# ==========================================================================
#
# Each demonstration profile should have a corresponding:
#
#     label_<profile_id>
#
# column in the demonstration dataset.
# ==========================================================================

DEMO_PROFILE_IDS = (
    "ananya_ai",
    "rohan_mech",
)


# ==========================================================================
# 11. EXPERIMENTAL REPRODUCIBILITY
# ==========================================================================
#
# Dataset:
#
#     70% Training
#     15% Validation
#     15% Test
#
# Training:
#     model learning
#
# Validation:
#     threshold selection
#     confidence calibration
#     hyperparameter tuning
#
# Test:
#     final unbiased evaluation only
#
# A fixed random seed allows experiments to be reproduced.
# ==========================================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================================================
# 12. EXPERIMENT MODES
# ==========================================================================
#
# These modes support systematic comparison and ablation experiments.
#
# Baseline 1:
#     conventional binary Important / Spam classification
#
# Baseline 2:
#     ordinary three-class classification
#
# Proposed stages:
#     personalization
#     uncertainty-aware selective triage
#     adaptive learning
#
# ==========================================================================

EXPERIMENT_BINARY_BASELINE = "binary_baseline"

EXPERIMENT_THREE_CLASS_BASELINE = "three_class_baseline"

EXPERIMENT_PERSONALIZED = "personalized"

EXPERIMENT_UNCERTAINTY_AWARE = "personalized_uncertainty"

EXPERIMENT_ADAPTIVE = "personalized_uncertainty_adaptive"


EXPERIMENT_MODES = (
    EXPERIMENT_BINARY_BASELINE,
    EXPERIMENT_THREE_CLASS_BASELINE,
    EXPERIMENT_PERSONALIZED,
    EXPERIMENT_UNCERTAINTY_AWARE,
    EXPERIMENT_ADAPTIVE,
)


# Complete proposed architecture.
DEFAULT_EXPERIMENT_MODE = EXPERIMENT_ADAPTIVE


# ==========================================================================
# 13. EVALUATION METRICS
# ==========================================================================
#
# analytics.py should eventually calculate these metrics.
#
# Particularly important:
#
#     Important-email recall
#     Priority precision
#     Spam precision
#     Critical Important -> Spam error
#     Review / abstention rate
#     Inbox clutter reduction
#     Calibration error
#     Personalization gain
#
# ==========================================================================

EVALUATION_METRICS = (
    "priority_precision",
    "important_email_recall",
    "spam_precision",
    "review_rate",
    "critical_priority_to_spam_rate",
    "inbox_clutter_reduction",
    "classification_accuracy",
    "classification_confidence",
    "calibration_error",
    "personalization_gain",
)


# ==========================================================================
# 14. CONFIGURATION VALIDATION
# ==========================================================================

def validate_config():
    """
    Validate configuration consistency.

    Raises
    ------
    ValueError
        If weights, thresholds, ratios or other settings are invalid.
    """

    # ----------------------------------------------------------------------
    # Relevance weights
    # ----------------------------------------------------------------------

    total_weight = sum(RELEVANCE_WEIGHTS.values())

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "RELEVANCE_WEIGHTS must sum to 1.0, "
            f"got {total_weight}"
        )

    for component, weight in RELEVANCE_WEIGHTS.items():

        if not 0.0 <= weight <= 1.0:

            raise ValueError(
                f"Relevance weight '{component}' must be "
                f"between 0 and 1, got {weight}"
            )

    # ----------------------------------------------------------------------
    # Routing thresholds
    # ----------------------------------------------------------------------

    threshold_names = (
        "SPAM_MIN_SPAM_SCORE",
        "SPAM_MIN_CONFIDENCE",
        "PRIORITY_MIN_RELEVANCE",
        "PRIORITY_MIN_CONFIDENCE",
        "SAFEGUARD_SENDER_IMPORTANCE",
        "SAFEGUARD_PERSONAL_RELEVANCE",
    )

    for name in threshold_names:

        value = globals()[name]

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"{name} must be between 0 and 1, "
                f"got {value}"
            )

    # ----------------------------------------------------------------------
    # Error costs
    # ----------------------------------------------------------------------

    required_error_costs = (
        "important_sent_to_spam",
        "important_sent_to_review",
        "junk_sent_to_priority",
        "other",
    )

    for error_type in required_error_costs:

        if error_type not in ERROR_COSTS:

            raise ValueError(
                f"Missing required ERROR_COSTS entry: "
                f"{error_type}"
            )

    for error_type, cost in ERROR_COSTS.items():

        if cost < 0:

            raise ValueError(
                f"Error cost '{error_type}' "
                f"cannot be negative."
            )

    # ----------------------------------------------------------------------
    # Feedback configuration
    # ----------------------------------------------------------------------

    if FEEDBACK_STEP <= 0:

        raise ValueError(
            "FEEDBACK_STEP must be greater than 0."
        )

    if FEEDBACK_STEP > 1.0:

        raise ValueError(
            "FEEDBACK_STEP must not exceed 1.0."
        )

    if MAX_PREFERENCE_ADJUSTMENT < FEEDBACK_STEP:

        raise ValueError(
            "MAX_PREFERENCE_ADJUSTMENT must be "
            "greater than or equal to FEEDBACK_STEP."
        )

    if MAX_PREFERENCE_ADJUSTMENT > 1.0:

        raise ValueError(
            "MAX_PREFERENCE_ADJUSTMENT must not exceed 1.0."
        )

    if not isinstance(MIN_FEEDBACK_EVENTS, int):

        raise ValueError(
            "MIN_FEEDBACK_EVENTS must be an integer."
        )

    if MIN_FEEDBACK_EVENTS < 1:

        raise ValueError(
            "MIN_FEEDBACK_EVENTS must be at least 1."
        )

    # ----------------------------------------------------------------------
    # Dataset split
    # ----------------------------------------------------------------------

    split_total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(split_total - 1.0) > 1e-9:

        raise ValueError(
            "TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO "
            f"must equal 1.0, got {split_total}"
        )

    for name in (
        "TRAIN_RATIO",
        "VALIDATION_RATIO",
        "TEST_RATIO",
    ):

        value = globals()[name]

        if not 0.0 < value < 1.0:

            raise ValueError(
                f"{name} must be between 0 and 1, "
                f"got {value}"
            )

    # ----------------------------------------------------------------------
    # Confidence calibration
    # ----------------------------------------------------------------------

    if CONFIDENCE_CALIBRATION not in SUPPORTED_CALIBRATION_METHODS:

        raise ValueError(
            "Unsupported CONFIDENCE_CALIBRATION: "
            f"{CONFIDENCE_CALIBRATION}. "
            f"Expected one of "
            f"{SUPPORTED_CALIBRATION_METHODS}"
        )

    # ----------------------------------------------------------------------
    # Experiment mode
    # ----------------------------------------------------------------------

    if DEFAULT_EXPERIMENT_MODE not in EXPERIMENT_MODES:

        raise ValueError(
            "DEFAULT_EXPERIMENT_MODE must be one of "
            f"{EXPERIMENT_MODES}"
        )

    # ----------------------------------------------------------------------
    # Demonstration profiles
    # ----------------------------------------------------------------------

    if not DEMO_PROFILE_IDS:

        raise ValueError(
            "At least one demonstration profile is required."
        )

    if len(set(DEMO_PROFILE_IDS)) != len(DEMO_PROFILE_IDS):

        raise ValueError(
            "DEMO_PROFILE_IDS contains duplicate profile IDs."
        )

    # ----------------------------------------------------------------------
    # Evaluation metrics
    # ----------------------------------------------------------------------

    if not EVALUATION_METRICS:

        raise ValueError(
            "At least one evaluation metric must be configured."
        )


# ==========================================================================
# 15. AUTOMATIC CONFIGURATION CHECK
# ==========================================================================

# Validate configuration immediately when this module is imported.
#
# This allows configuration errors to be detected early rather than
# producing difficult-to-understand errors later in the application.

validate_config()