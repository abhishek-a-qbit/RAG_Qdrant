from data_driven_question_generator import generate_data_driven_questions

print("Testing question generator...")
questions = generate_data_driven_questions("6sense for startups", 3)
print(f"Generated {len(questions)} questions:")
for i, q in enumerate(questions):
    print(f"Question {i+1}: {q.get('question', 'No question')}")
    print(f"Topic: {q.get('topic', 'No topic')}")
    print(f"Confidence: {q.get('confidence', 0)}")
