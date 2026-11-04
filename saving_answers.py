import set

partial_answers = dict()




def create_partial_author(answer):
    question_topic = "gender"  # string e.g. "Is author male or female?"
    binary_answer = 0  # 0/1 e.g. 0 for male, 1 for female

    partial_answers[question_topic] = binary_answer

def save_answers(answers):
    for answer in answers:
        partial_answers[create_partial_author(answer)] = answer