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

personal = {"sender": "Rahul <rahul@gmail.com>", "subject": "Project review tomorrow",
            "body": "Please confirm and submit your part before tomorrow 5 PM.",
            "labels": ["INBOX", "CATEGORY_PERSONAL"], "list_unsubscribe": False}
r4 = {"category": "REVIEW", "features": {"ai_spam_probability": 0.03, "spam_score": 0.05,
      "semantic_relevance": 0.3, "urgency": 0.43, "action_required": 0.65}}
order = {"sender": "Swiggy <noreply@swiggy.in>", "subject": "Your order has been delivered",
         "body": "Thanks for ordering. Your order was delivered.",
         "labels": ["INBOX", "CATEGORY_UPDATES"], "list_unsubscribe": False}
r5 = {"category": "REVIEW", "features": {"ai_spam_probability": 0.02, "spam_score": 0.05,
      "semantic_relevance": 0.1, "urgency": 0.1, "action_required": 0.1}}
print("personal ->", choose_category(personal, profile, r4))
print("order    ->", choose_category(order, profile, r5))
