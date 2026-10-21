from mailmind_routing import choose_category

profile = {"interests": ["machine learning", "python"],
           "important_topics": ["deadline"],
           "important_senders": [], "important_domains": [],
           "low_priority_topics": [], "excluded": []}

promo = {"sender": "Store <deals@shop.com>",
         "subject": "Sale ends tonight, 50% discount",
         "body": "Shop now! Limited time offer. Unsubscribe anytime.",
         "labels": ["CATEGORY_PROMOTIONS"], "list_unsubscribe": True}
r1 = {"category": "REVIEW",
      "features": {"ai_spam_probability": 0.05, "spam_score": 0.05,
                   "semantic_relevance": 0.2}}

useful = {"sender": "professor@college.edu",
          "subject": "Assignment deadline",
          "body": "Submit the machine learning assignment tomorrow.",
          "labels": ["INBOX"], "list_unsubscribe": False}
r2 = {"category": "REVIEW",
      "features": {"ai_spam_probability": 0.02, "spam_score": 0.05,
                   "semantic_relevance": 0.7}}

print("promo  ->", choose_category(promo, profile, r1))
print("useful ->", choose_category(useful, profile, r2))

promo2 = {"sender": "Coursehub <hello@coursehub.com>",
          "subject": "New machine learning and Python courses picked for you",
          "body": "Hi! Boost your career with our latest machine learning and Python courses. Unsubscribe here.",
          "labels": ["CATEGORY_PROMOTIONS"], "list_unsubscribe": True}
r3 = {"category": "REVIEW",
      "features": {"ai_spam_probability": 0.05, "spam_score": 0.05,
                   "semantic_relevance": 0.55}}
print("promo2 ->", choose_category(promo2, profile, r3))
