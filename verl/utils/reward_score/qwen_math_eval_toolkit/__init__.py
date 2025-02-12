import re
from .grader import math_equal
from .parser import extract_answer


def compute_score(solution_str, ground_truth) -> float:
    # breakpoint()
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extracted_answer = extract_answer(model_output, data_name="math")  #TODO: check the data_name, hard code here for now

    if math_equal(prediction=extracted_answer, reference=ground_truth, timeout=False):
        box_match = 1.0
    else:
        box_match = -0.5
        
    if "boxed" not in model_output:
        box_match = -1.0

    return box_match
