"""
Model utils
"""
import os
import json

def clean_text(inp):
    def clean(txt):
        return "\n".join(
            line.replace("•", "").replace("-", "").replace("*", "").replace("#", " ")
            for line in txt.split("\n")
        )

    if isinstance(inp, list):
        return_out = ",".join([clean(string) for string in inp])
    elif isinstance(inp, str):
        return_out = clean(inp)

    return return_out

def read_json_data(filepath):
    if not isinstance(filepath, list):
        filepath = [filepath]

    return_json_content_dict = dict()
    for path in filepath:
        if os.path.isfile(path) and os.path.splitext(path)[-1].lower()==".json":
            with open(path) as json_in:
                json_data = json.load(json_in)
        else:
            json_data = None
        return_json_content_dict[path] = json_data
    return return_json_content_dict
