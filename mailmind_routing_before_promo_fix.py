"""Personalized post-model routing for MailMind AI.

DistilBERT and MiniLM remain in the existing classifier. This module applies
explicit user preferences and conservative spam safeguards to its output.
"""
import re
from email.utils import parseaddr

SECURITY_PHRASES = (
    'security alert', 'suspicious sign-in', 'suspicious login',
    'password reset', 'password changed', 'new sign-in',
    'two-factor authentication', 'oauth application',
    'third-party oauth', 'account recovery', 'unauthorized access',
    'application has been added', 'authorized to access your account',
)
SUSPICIOUS_PHRASES = (
    'send your password', 'share your password', 'send your otp',
    'share your otp', 'send your verification code',
    'gift card payment', 'transfer funds immediately',
    'verify your account by sending',
)


def _address(sender):
    return parseaddr(str(sender or ''))[1].strip().lower()


def _domain(address):
    return address.rsplit('@', 1)[1] if '@' in address else ''


def _explicit_sender(sender, profile):
    address = _address(sender)
    if not address:
        return False
    return any(_address(entry) == address for entry in profile.get('important_senders', []) if _address(entry))


def _explicit_domain(sender, profile):
    domain = _domain(_address(sender))
    if not domain:
        return False
    for entry in profile.get('important_domains', []):
        candidate = str(entry).lower().strip().lstrip('@')
        # Ignore topics mistakenly entered as domains, e.g. 'deadline'.
        if '.' in candidate and (domain == candidate or domain.endswith('.' + candidate)):
            return True
    return False


def _topic_matches(subject, body, phrases):
    text = f'{subject} {body}'.lower()
    return [str(p) for p in phrases if str(p).strip() and re.search(r'(?<!\w)' + re.escape(str(p).strip().lower()) + r'(?!\w)', text)]


def choose_category(email, profile, model_result):
    """Return (category, explanation) without changing AI feature values.

    Explicit senders receive priority only when spam evidence is low and
    obvious suspicious wording is absent. A matching From header is not proof
    of authentication (SPF/DKIM/DMARC are not available here).
    """
    f = model_result.get('features', {})
    ai_spam = float(f.get('ai_spam_probability', 0.5))
    rule_spam = float(f.get('spam_score', 0))
    subject, body = str(email.get('subject') or ''), str(email.get('body') or '')
    text = (subject + '\n' + body).lower()
    sender_match = _explicit_sender(email.get('sender'), profile)
    domain_match = _explicit_domain(email.get('sender'), profile)
    trusted_preference = sender_match or domain_match
    suspicious = any(phrase in text for phrase in SUSPICIOUS_PHRASES)
    security = any(phrase in text for phrase in SECURITY_PHRASES)
    low_matches = _topic_matches(subject, body, profile.get('low_priority_topics', []))
    positive_matches = _topic_matches(subject, body, profile.get('interests', []) + profile.get('important_topics', []))
    baseline = str(model_result.get('category', 'review')).upper()

    # User-defined exclusions override AI classification.
    # Exact sender/domain matches and whole-phrase topic matches.
    excluded = profile.get("excluded", [])
    sender_address = _address(email.get("sender"))
    sender_domain = _domain(sender_address)

    for item in excluded:
        term = str(item).strip().lower().lstrip("@")
        if not term:
            continue

        if "@" in term:
            matched = sender_address == term
        elif "." in term and " " not in term:
            matched = sender_domain == term or sender_domain.endswith("." + term)
        else:
            matched = bool(re.search(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                text
            ))

        if matched:
            return "SPAM", f"Matched your Exclude preference: {item}"

    # Spam is conservative: do not hide a user-important message on one signal.
    if ai_spam >= 0.70:
        if trusted_preference:
            return 'REVIEW', 'High spam probability conflicts with an important-sender preference; verify the sender independently.'
        return 'SPAM', f'DistilBERT predicts high spam probability ({ai_spam:.0%}); topic relevance does not override spam evidence.'
    if ai_spam >= 0.50 or suspicious:
        return 'REVIEW', 'Suspicious content or substantial spam evidence: review before trusting links or requests.'
    if trusted_preference:
        if ai_spam < 0.20 and rule_spam < 0.50:
            reason = 'explicitly important sender' if sender_match else 'explicitly important domain'
            suffix = ' Account-security content detected; verify independently before acting.' if security else ' Sender address alone does not prove authenticity.'
            return 'PRIORITY', f'Priority because this matches your {reason} and spam evidence is low.' + suffix
        return 'REVIEW', 'Important sender preference matched, but spam evidence warrants manual review.'
    # Personalized unwanted-promotion detection.
    # Require multiple marketing signals and low semantic relevance.
    promotional_phrases = (
        "unsubscribe", "shop now", "limited time offer",
        "exclusive offer", "special offer", "recommended for you",
        "recommendations", "discover new", "explore more",
        "trending", "sale ends", "discount", "newsletter",
        "handpicked for you", "this is peak you",
    )

    promotional_sender_terms = (
        "recommendations@", "newsletter@", "offers@",
        "promotions@", "marketing@", "updates.",
    )

    marketing_hits = sum(
        phrase in text for phrase in promotional_phrases
    )

    marketing_sender = any(
        term in sender_address for term in promotional_sender_terms
    )

    semantic_relevance = float(
        f.get("semantic_relevance", 0.0)
    )

    if (
        ai_spam < 0.50
        and not trusted_preference
        and not positive_matches
        and not security
        and (
            (marketing_sender and semantic_relevance < 0.60)
            or (marketing_hits >= 3 and semantic_relevance < 0.35)
        )
    ):
        return (
            "SPAM",
            "Unwanted promotional email: marketing signals "
            "and low personalized relevance."
        )

    if low_matches and not positive_matches:
        return 'REVIEW', f'Lower relevance to your profile (matched: {", ".join(low_matches[:3])}). Low relevance is not proof of spam.'
    if baseline == 'PRIORITY':
        return 'PRIORITY', str(model_result.get('explanation', 'Priority according to personalized classifier.'))
    if baseline == 'SPAM' and ai_spam < 0.70:
        return 'REVIEW', 'Spam evidence is not strong enough to hide this message automatically.'
    return 'REVIEW', str(model_result.get('explanation', 'Insufficient evidence for priority or spam.'))
