import pandas as pd
import numpy as np
import re
import json
import pickle

dataset = pd.read_json("data/Entity Recognition in Resumes.json", lines=True)
cv_text = np.array(dataset.content)


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


all_labels = []
for ind, annotation in enumerate(dataset.annotation):
    _ = [
        all_labels.append(entity_lst["label"])
        for entity_lst in annotation
        if entity_lst["label"] not in all_labels and len(entity_lst["label"]) > 0
    ]

dataset_reformatted = pd.DataFrame(
    columns=[
        "documentNum",
        "documentText",
        "documentAnnotation",
        "sentenceNum",
        "sentence",
        "labelsDict",
        "containsLabel",
        "wordNum",
        "word",
        "labelName",
    ]
)
k = 0

data_annotated = []
for df_index in range(len(dataset)):
    this_df = dataset.iloc[df_index]
    this_df_content = this_df["content"]
    this_df_annotation = this_df["annotation"]
    this_df_annotation_sorted = sorted(
        this_df_annotation, key=lambda label_dict: label_dict["points"][0]["start"]
    )
    is_begining = True
    is_last = False
    final_text_list = []
    final_out_list = []
    prev_end = None
    # i=0; label_dict = this_df_annotation_sorted[i]
    for i, label_dict in enumerate(this_df_annotation_sorted):
        if len(label_dict["label"]) > 0:
            label = label_dict["label"][0]
            start = label_dict["points"][0]["start"]
            end = label_dict["points"][0]["end"]
            text = label_dict["points"][0]["text"]
            if i == len(this_df_annotation_sorted) - 1:
                is_last = True

            if is_begining:
                # Additional text i.e. Os
                extra_text = clean_text(this_df_content[:start])
                is_begining = False
            else:
                extra_text = clean_text(this_df_content[prev_end + 1 : start])

            prev_end = end
            text_list = [
                txt.replace("\n", "")
                for txt in extra_text.split(" ")
                if txt.strip() != "" and txt.strip() != "" and len(txt.strip()) > 1
            ]
            final_text_list += text_list
            final_out_list += ["O"] * len(text_list)

            text_list = [
                txt.replace("\n", "")
                for txt in clean_text(text).split(" ")
                if txt.strip() != "" and txt.strip() != "" and len(txt.strip()) > 1
            ]
            final_text_list += text_list
            final_out_list += [label.upper() + "-B"] + (
                [label.upper() + "-I"] * (len(text_list) - 1)
            )

            if is_last:
                extra_text = clean_text(this_df_content[end + 1 :])
                text_list = [
                    txt.replace("\n", "")
                    for txt in extra_text.split(" ")
                    if txt.strip() != "" and txt.strip() != "" and len(txt.strip()) > 1
                ]
                final_text_list += text_list
                final_out_list += ["O"] * len(text_list)

    data_annotated.append(tuple(zip(final_text_list, final_out_list)))

with open("data/data_ready_list.pkl", "wb") as out_file:
    pickle.dump(data_annotated, out_file, protocol=3)
