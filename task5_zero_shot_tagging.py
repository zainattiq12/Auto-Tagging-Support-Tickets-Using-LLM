from transformers import pipeline
import pandas as pd

# Load support tickets dataset
df = pd.read_csv("support_tickets.csv")

# Zero-shot classification model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Candidate tags
candidate_labels = [
    "Login Issue",
    "Password Reset",
    "Payment Problem",
    "Refund Request",
    "Technical Bug",
    "Account Access",
    "Order Issue",
    "Account Suspension"
]

results = []

for ticket in df["ticket"]:

    prediction = classifier(
        ticket,
        candidate_labels,
        multi_label=True
    )

    top3_tags = prediction["labels"][:3]
    top3_scores = prediction["scores"][:3]

    results.append({
        "ticket": ticket,
        "tag1": top3_tags[0],
        "tag2": top3_tags[1],
        "tag3": top3_tags[2],
        "score1": round(top3_scores[0], 4),
        "score2": round(top3_scores[1], 4),
        "score3": round(top3_scores[2], 4)
    })

output = pd.DataFrame(results)

print(output)

output.to_csv(
    "ticket_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")