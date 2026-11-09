partial_answers = dict()




def create_partial_author(answer, question_topic):
    binary_answer = 0  # 0/1 e.g. 0 for male, 1 for female

    partial_answers[question_topic] = binary_answer

def save_answers(answer, question_topic):
    partial_answers[create_partial_author(answer, question_topic)] = answer
    
    
def retrieve_partial_answers():
    return partial_answers    