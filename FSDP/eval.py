import json
from pydantic import BaseModel
from inspect import signature
import ast
import argparse

class FunctionCall(BaseModel):
    name: str
    arguments: dict


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pred_file" , type=str)
    args.add_argument("--error_file" , type=str)
    return args.parse_args() 

if __name__ == "__main__":
    args = get_args()

    with open(args.pred_file , "r") as f:
        data = f.read()

    lis = data.split("-------------------")
    lis = lis[0:-1]
    print(len(lis))
    s = 0
    c = 0 
    error_count = 0
    with open(args.error_file , "w") as f:
        for j,i in enumerate(lis):
            original = i.split("Original: ")[1].split("\n")[0]
            function_call_ast = ast.parse(original)
            function_name = function_call_ast.body[0].value.func.id
            arguments = {}
            for arg in function_call_ast.body[0].value.keywords:
                arguments[arg.arg] = arg.value.s
            
            function_call = FunctionCall(
            name=function_name,
            arguments=arguments
            )
            
            json_string = function_call.json(indent=4)
            #print(json_string)
            try:
                pred = i.split("<functioncall> ")[1].split("\n")[0]
                pred = pred.replace("'", '')
                pred_json = json.loads(pred)

                if pred_json == json.loads(json_string):
                    s = s + 1
                else:
                    error_count = error_count + 1
                    if pred_json["name"] == json.loads(json_string)["name"]:
                        print("name same")
                        print("original" , json_string)
                        print("pred" , pred_json)

                    f.write("original" + json_string + "\n")
                    f.write("pred" + str(pred_json) + "\n")
                    f.write("---------------------------------------------------------\n")
            except:
                f.write("original" + json_string + "\n")
                f.write("pred " +  'ERROR' + "\n")
                f.write("---------------------------------------------------------\n")
            c = c + 1

    print("total count" , c)
    print("error count" , error_count)
    print("accuracy"  , s/c * 100)