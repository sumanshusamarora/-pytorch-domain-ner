"""
Prepare data for processing from Data Studio annotations
"""
import os
import argparse
import pickle
import json
from utils import clean_text, read_json_data

class CleanPrepareDataset:

    def __init__(self, dir_path, file_extension='.json'):
        """

        :param dir_path: directory containing jsons
        :param file_extension: defaults to .json
        """
        self.dir_path = dir_path
        self.file_extension = file_extension

    def prepare_ner_dataset(self):
        self.filepath_list = self.get_file_list()
        self.json_content_dict = read_json_data(self.filepath_list)
        self.label_dict_list = self.simplify_label_dict(self.json_content_dict)
        self.trianable_dataset = self.get_trainable_data(self.label_dict_list)
        return self.trianable_dataset

    def save(self, filename:str="data_ready_list.pkl", save_dir_path:str=None, pickle_protocol=5):
        if not save_dir_path:
            save_dir_path = self.dir_path

        if not filename.lower().endswith(".pkl"):
            filename = f"{filename}.pkl"


        with open(os.path.join(save_dir_path, filename), "wb") as out_file:
            pickle.dump(self.trianable_dataset, out_file, protocol=pickle_protocol)


    def get_file_list(self):
        """

        :param dir_path:
        :param file_extension:
        :return:
        """
        dir_path = self.dir_path.replace("\\", "/")

        if not self.file_extension.startswith('.'):
            self.file_extension = f".{self.file_extension}"

        if os.path.isdir(dir_path):
            return_list = [os.path.join(os.path.splitext(dir_path)[0], file) for file in os.listdir(dir_path) if
                           os.path.splitext(file)[-1].lower() == self.file_extension]
        else:
            return_list = []
        return return_list

    @staticmethod
    def simplify_label_dict(json_content_dict: dict):
        """

        :param json_content_dict:
        :return:
        """
        label_dict_list = []

        def split_text_start_end_label(data_studio_json):
            """

            :param data_studio_json:
            :return:
            """
            out_dict = dict()
            out_dict["full_text"] = data_studio_json['data']['text']
            out_dict["labels"] = dict()
            for result in data_studio_json['completions'][0]['result']:
                this_value_dict = result['value']
                label = this_value_dict['labels'][0]
                start = this_value_dict['start']
                end = this_value_dict['end']
                text = this_value_dict['text']
                out_dict["labels"][label] = {"start": start, "end": end, "text": text}
            return out_dict

        for annotation_dict in json_content_dict.values():
            label_dict_list.append(split_text_start_end_label(annotation_dict))

        return label_dict_list

    def get_trainable_data(self, label_dict_list):
        data_annotated = []
        for ind, label_dict in enumerate(label_dict_list):
            this_content = label_dict["full_text"]
            this_annotation = label_dict["labels"]
            this_annotation_sorted = sorted(
                this_annotation.items(), key=lambda item: item[1]['start']
            )
            is_begining = True
            is_last = False
            final_text_list = []
            final_out_list = []
            prev_end = None
            # i=0; label_dict = this_annotation_sorted[i]
            for i, label_dict in enumerate(this_annotation_sorted):
                if len(label_dict) > 0:
                    label = label_dict[0]
                    start = label_dict[1]["start"]
                    end = label_dict[1]["end"]
                    text = label_dict[1]["text"]
                    if i == len(this_annotation_sorted) - 1:
                        is_last = True

                    if is_begining:
                        # Additional text i.e. Os
                        extra_text = clean_text(this_content[:start])
                        is_begining = False
                    else:
                        extra_text = clean_text(this_content[prev_end + 1: start])

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
                        extra_text = clean_text(this_content[end + 1:])
                        text_list = [
                            txt.replace("\n", "")
                            for txt in extra_text.split(" ")
                            if txt.strip() != "" and txt.strip() != "" and len(txt.strip()) > 1
                        ]
                        final_text_list += text_list
                        final_out_list += ["O"] * len(text_list)

            data_annotated.append(tuple(zip(final_text_list, final_out_list)))
        return data_annotated






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Input Values")
    parser.add_argument(
        "--data-path",
        dest="DATA_PATH",
        default='data',
        type=str,
        help="Directory path containing json annotation files",
    )

    parser.add_argument(
        "--file-ext",
        dest="FILE_EXTENSION",
        default='.json',
        type=str,
        help="File extension to look for in data path",
    )

    parser.add_argument(
        "--save-dir-path",
        dest="SAVE_DIR_PATH",
        default='data',
        type=str,
        help="Directory path to save dataset pickle file",
    )

    parser.add_argument(
        "--filename",
        dest="filename",
        default='data_ready_list.pkl',
        type=str,
        help="Name of pickle file to save",
    )

    parser.add_argument(
        "--pickle-protocol",
        dest="PICKLE_PROTOCOL",
        default=3,
        type=int,
        help="Pickle Protocol",
    )

    args = parser.parse_args()

    cpd = CleanPrepareDataset(dir_path=args.DATA_PATH, file_extension=args.FILE_EXTENSION)
    dataset = cpd.prepare_ner_dataset()
    cpd.save(filename=args.filename, save_dir_path=args.SAVE_DIR_PATH)
