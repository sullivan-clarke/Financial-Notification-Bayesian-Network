import math
import re
from collections import defaultdict

# =============================================================================
# TRAINING DATA
# =============================================================================

TRAINING_DATA = [

    #HIGH PRIORITY DATA 
    ("Stock XYZs price has dropped, buying opportunity detected!",   "High Priority"),
    ("Stock XYZs price has climbed, continue to hold!",    "High Priority"),
    ("Your credit card bill is due in 24 hours",      "High Priority"),
    ("We have detected security breaches in your accounts",           "High Priority"),
    ("Unknown login attempt detected in your investment account",   "High Priority"),
    ("Transaction to invest in company XYZ completed",   "High Priority"),
    ("Transaction to invest in company XYZ failed",   "High Priority"),
    ("Large withdrawal detected from your account", "High Priority"),
    ("Your account password was changed successfully", "High Priority"),
    ("Suspicious activity detected on your debit card", "High Priority"),
    ("Payment failed due to insufficient funds", "High Priority"),
    ("Margin call: additional funds required immediately", "High Priority"),
    ("Your account has been locked due to multiple failed login attempts", "High Priority"),
    ("Wire transfer of $5000 initiated", "High Priority"),
    ("Unrecognized device logged into your account", "High Priority"),
    ("Your loan payment is overdue", "High Priority"),
    ("Immediate action required: verify your identity", "High Priority"),

    #MEDIUM PRIORITY DATA
    ("Your credit card balance is $1000, due in 2 weeks", "Medium Priority"),
    ("Check your credit score today",   "Medium Priority"),
    ("Company XYZ recently released their earnings statement",     "Medium Priority"),
    ("Company XYZ is going public soon",           "Medium Priority"),
    ("Todays market news",         "Medium Priority"),
    ("Your monthly bank statement is now available", "Medium Priority"),
    ("Reminder: upcoming bill due next week", "Medium Priority"),
    ("Stock ABC reached your watchlist price", "Medium Priority"),
    ("Dividend payment has been issued", "Medium Priority"),
    ("Portfolio performance summary for this month", "Medium Priority"),
    ("Your credit score has been updated", "Medium Priority"),
    ("Market volatility is increasing today", "Medium Priority"),
    ("New analyst rating for company XYZ", "Medium Priority"),
    ("ETF in your portfolio has rebalanced", "Medium Priority"),
    ("Weekly investment newsletter", "Medium Priority"),

    #LOW PRIORITY DATA
    ("Open a new high-yield savings account and earn up to a $500 bonus!",         "Low Priority"),
    ("New credit card deals",              "Low Priority"),
    ("The best stock sources to follow in 2026",      "Low Priority"),
    ("Take our investment survey for the chance to win a giftcard",          "Low Priority"),
    ("We have recently updated our terms of service",      "Low Priority"),
    ("Earn rewards with our new credit card offer", "Low Priority"),
    ("Top 10 stocks to watch this year", "Low Priority"),
    ("Upgrade your account for premium benefits", "Low Priority"),
    ("Join our webinar on financial planning", "Low Priority"),
    ("Limited time offer on personal loans", "Low Priority"),
    ("Discover new investment opportunities today", "Low Priority"),
    ("Refer a friend and earn bonuses", "Low Priority"),
    ("Check out our latest blog post on saving money", "Low Priority"),
    ("Exclusive offer just for you", "Low Priority"),
    ("Start investing with zero fees today", "Low Priority"),
]

# =============================================================================
# ALERTS TO CLASSIFY
# =============================================================================

ALERTS_TO_CLASSIFY = [
    # --- HIGH PRIORITY ---
    "Unrecognized transaction detected on your account",  # high
    "Your account has been temporarily locked",  # high
    "Immediate action required: suspicious login attempt",  # high
    "Payment declined due to insufficient funds",  # high
    "Your password was changed from a new device",  # high
    "Wire transfer request pending approval",  # high
    "Security alert: unusual activity detected",  # high
    "Error, your transaction has failed",  # high


    # --- MEDIUM PRIORITY ---
    "Reminder: your credit card bill is due soon",  # medium
    "Stock ABC has reached your target price",  # medium
    "Monthly account statement is ready to view",  # medium
    "Your portfolio value has changed today",  # medium
    "Dividend payout has been processed",  # medium
    "Market update: stocks trending downward",  # medium
    "New analyst report available for company XYZ",  # medium
    "New company rating for company XYZ", # medium 


    # --- LOW PRIORITY ---
    "Apply now for our premium rewards credit card",  # low
    "Special promotion: earn cashback on all purchases",  # low
    "Join our free investing webinar this weekend",  # low
    "Limited time offer on personal loans",  # low
    "Check out our latest financial tips blog",  # low
    "Refer a friend and earn rewards",  # low
    "Upgrade your account for exclusive benefits",  # low
    "Earn up to $1000 when opening a new checking account with us", # low
    "Check out our newest credit card offers", # low



    # --- EDGE CASES (good for testing model quality) ---
    "Your payment is scheduled for tomorrow",  # medium / high
    "Stock XYZ is experiencing unusual volatility",  # medium / high
    "Account verification recommended",  # medium / high
    "New login from a recognized device",  # medium
    "Investment opportunity available now",  # low / medium
    "Your balance is lower than usual",  # medium
    "New investment information over stock XYS", # high / medium 
]

# =============================================================================
# NAIVE BAYES — core logic
# =============================================================================
 
def tokenize(text):
    """Lowercase, strip punctuation, split on whitespace, drop 1-char tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return [w for w in text.split() if len(w) > 1]
 
 
def train(data):
    """
    Build the model from training data.
 
    Returns a dict with everything needed to classify new text:
      - class_counts   : how many training examples per label
      - word_counts    : word frequencies per label
      - word_totals    : total words seen per label
      - vocab_size     : number of unique words across all labels
      - labels         : list of unique labels
      - total_docs     : total number of training examples
    """
    class_counts = defaultdict(int)
    word_counts  = defaultdict(lambda: defaultdict(int))
    vocab        = set()
 
    for text, label in data:
        class_counts[label] += 1
        for word in tokenize(text):
            word_counts[label][word] += 1
            vocab.add(word)
 
    word_totals = {label: sum(word_counts[label].values()) for label in class_counts}
 
    return {
        "class_counts" : dict(class_counts),
        "word_counts"  : {k: dict(v) for k, v in word_counts.items()},
        "word_totals"  : word_totals,
        "vocab_size"   : len(vocab),
        "labels"       : list(class_counts.keys()),
        "total_docs"   : len(data),
    }
 
 
def classify(model, text):
    """
    Score each label using Bayes' theorem in log-space:
 
        log P(label | words)  ∝  log P(label)  +  Σ log P(word | label)
 
    Laplace smoothing (+1) prevents zero probabilities for unseen words.
 
    Returns:
      - predicted  : the winning label
      - probs      : dict of label -> human-readable probability (0–1)
      - log_scores : raw log-probability scores per label
    """
    cc  = model["class_counts"]
    wc  = model["word_counts"]
    wt  = model["word_totals"]
    V   = model["vocab_size"]
    N   = model["total_docs"]
    words = tokenize(text)
 
    log_scores = {}
    for label in model["labels"]:
        # Prior: how common is this label in training data?
        log_score = math.log(cc[label] / N)
 
        # Likelihood: how likely is each word given this label?
        for word in words:
            count = wc[label].get(word, 0) + 1   # +1 = Laplace smoothing
            denom = wt[label] + V                 # normalise over vocab
            log_score += math.log(count / denom)
 
        log_scores[label] = log_score
 
    # Convert log-scores to probabilities for display
    max_score = max(log_scores.values())
    exp_scores = {l: math.exp(s - max_score) for l, s in log_scores.items()}
    total_exp  = sum(exp_scores.values())
    probs      = {l: v / total_exp for l, v in exp_scores.items()}
 
    predicted = max(log_scores, key=log_scores.get)
    return predicted, probs, log_scores
 
 
def print_result(text, model):
    predicted, probs, _ = classify(model, text)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
 
    print(f'\n  Input : "{text}"')
    print(f"  Result: {predicted}")
    print("  Probabilities:")
    for label, prob in sorted_probs:
        bar   = "█" * int(prob * 30)
        arrow = " <-- predicted" if label == predicted else ""
        print(f"    {label:<12} {prob*100:5.1f}%  {bar}{arrow}")
 
 
def print_model_summary(model):
    print("\n" + "=" * 60)
    print("  MODEL SUMMARY")
    print("=" * 60)
    print(f"  Training examples : {model['total_docs']}")
    print(f"  Vocabulary size   : {model['vocab_size']} unique words")
    print(f"  Categories        : {', '.join(model['labels'])}")
    for label in model["labels"]:
        count = model["class_counts"][label]
        pct   = count / model["total_docs"] * 100
        print(f"    {label:<12} {count} examples  ({pct:.0f}%)")
    print("=" * 60)
 
 
def print_classification_summary(results):
    """
    Print a summary of how many alerts were classified into each category.
 
    Expects `results` to be a list of (text, predicted_label) tuples,
    which is built up in the main block below.
    """
    counts = defaultdict(int)
    for _, label in results:
        counts[label] += 1
 
    total = len(results)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
 
    print("\n" + "=" * 60)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total alerts classified: {total}\n")
    for label, count in sorted_counts:
        bar = "█" * int((count / total) * 30)
        pct = count / total * 100
        print(f"    {label:<12} {count:>3} alert(s)  {pct:5.1f}%  {bar}")
    print("=" * 60)
 
 
# =============================================================================
# MAIN
# =============================================================================
 
if __name__ == "__main__":
    # 1. Train
    model = train(TRAINING_DATA)
    print_model_summary(model)
 
    # 2. Classify each alert, print results, and collect into a list
    print("\n  CLASSIFICATIONS")
    print("=" * 60)
    results = []
    for alert in ALERTS_TO_CLASSIFY:
        print_result(alert, model)
        predicted, _, _ = classify(model, alert)
        results.append((alert, predicted))
 
    # 3. Print the summary of how many fell into each category
    print_classification_summary(results)
 
    # 4. Interactive mode 
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE  (type an alert, or 'quit' to exit)")
    print("=" * 60)
    while True:
        try:
            user_input = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Exiting.")
            break
        if user_input:
            print_result(user_input, model)