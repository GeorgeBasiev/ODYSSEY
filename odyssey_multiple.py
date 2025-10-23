import sys
import json
import os

from odyssey_single import process_single


if __name__=="__main__":
    with open(sys.argv[1], "r") as f:
        inputs = json.load(f)
    
    outputs_dict = inputs
    for index, input in enumerate(inputs):
        input_dict = dict(input)
        answer = process_single(input_dict)

        outputs_dict[index]["answer"] = answer
    
    os.makedirs("outputs")
    output_filename = f"outputs/{os.path.basename(sys.argv[1][:sys.argv[1].index(".")])}_with_answers.json"
    with open(output_filename, "w") as f:
        json.dump(outputs_dict, f, indent=4)
    
    print("Successfully processed {sys.argv[1]} and saved results to {output_filename}!")

