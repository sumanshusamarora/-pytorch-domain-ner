"""
Inference code
"""
import argparse
import os
import torch
import ast
import dill
from torchnlp.encoders.text import pad_tensor
import mlflow.pytorch
import yaml
from utils import (
    clean_text,
    get_word_proba,
    get_entities_values_joint_probas,
    get_one_value_each_entity,
)
from train_cnn_rnn_crf import (
    get_POS_tags,
    trim_list_of_lists_upto_max_len,
    tokenize_pos_tags,
    pad_and_stack_list_of_list,
    enrich_data,
)

with open("inference_config.yml", "r") as fh:
    infer_config = yaml.safe_load(fh)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, x_padded, x_postag_padded, x_char_padded, x_enriched_features, restrict_if_no_begining=True):
    mask = torch.where(x_padded > 0,
                       torch.Tensor([1]).type(torch.uint8),
                       torch.Tensor([0]).type(torch.uint8),
                       )

    with torch.no_grad():
        out, decoded, crf_loss = model.predict(
            x_padded.to(device),
            x_postag_padded.to(device),
            x_char_padded.to(device),
            x_enriched_features.to(device),
            mask.to(device),
        )

    out_proba, softmax_scores = get_word_proba(emmision_matrix=out,
                                               transition_matrix=model.crf.transitions,
                                               decoded_out=decoded,
                                               o_index=y_ner_encoder.token_to_index['O'])

    result_y = [[y_ner_encoder.index_to_token[word] for word in prediction] for prediction in decoded][:len(X_text_list_as_is)]
    proba_y = [[proba.item() for proba in proba_list] for proba_list in out_proba]

    final_out_list = []
    for j in range(len(X_text_list_as_is)):
        final_out_list.append(
            get_entities_values_joint_probas(result=result_y[j],
                                             sentence=X_text_list_as_is[j],
                                             proba=proba_y[j],
                                             log_score=False,
                                             add_factor=0.5,
                                             restrict_if_no_begining=restrict_if_no_begining))

    final_out_dict = get_one_value_each_entity(final_out_list)
    return final_out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Input Values")
    parser.add_argument(
        "--data-text",
        dest="DATA_TEXT",
        default=infer_config["DATA_TEXT"],
        type=str,
        help="Text",
    )

    parser.add_argument(
        "--experiment-id",
        dest="EXPERIMENT_ID",
        default=infer_config["EXPERIMENT_ID"],
        type=str,
    )

    parser.add_argument(
        "--run-id",
        dest="RUN_ID",
        default=infer_config["RUN_ID"],
        type=str,
        help="MLFLOW Run Id",
    )

    parser.add_argument(
        "--restrict-if-no-beg",
        dest="RESTRICT_IF_NO_BEG",
        default=ast.literal_eval(infer_config["RESTRICT_IF_NO_BEG"]),
        action='store_false',
        help="Do not include prediction, if no beginning tag found",
    )

    args = parser.parse_args()

    experiment = mlflow.get_experiment(args.EXPERIMENT_ID)
    artifacts_uri = experiment.artifact_location
    artifcats_location = f"{artifacts_uri}/{args.RUN_ID}/artifacts/files"
    model_location = f"{artifacts_uri}/{args.RUN_ID}/artifacts/model"
    params_location = f"{artifacts_uri.replace('file:///', '')}/{args.RUN_ID}/params"

    with open(os.path.join(params_location, 'MAX_SENTENCE_LEN'), 'r') as infile:
        max_sentence_len = ast.literal_eval(infile.read())

    with open(os.path.join(params_location, 'MAX_WORD_LENGTH'), 'r') as infile:
        max_word_length = ast.literal_eval(infile.read())

    with open(os.path.join(params_location, 'TEST_INDEX'), 'r') as infile:
        TEST_INDEX = ast.literal_eval(infile.read())

    model = mlflow.pytorch.load_model(model_location).to(device)

    with open(f"mlruns/{args.EXPERIMENT_ID}/{args.RUN_ID}/artifacts/files/x_encoder", "rb") as infile:
        x_encoder = dill.load(infile)

    with open(f"mlruns/{args.EXPERIMENT_ID}/{args.RUN_ID}/artifacts/files/x_char_encoder", "rb") as infile:
        x_char_encoder = dill.load(infile)

    with open(f"mlruns/{args.EXPERIMENT_ID}/{args.RUN_ID}/artifacts/files/y_ner_encoder", "rb") as infile:
        y_ner_encoder = dill.load(infile)

    with open(f"mlruns/{args.EXPERIMENT_ID}/{args.RUN_ID}/artifacts/files/tag_to_index", "rb") as infile:
        tag_to_index = dill.load(infile)

    X_text = clean_text(args.DATA_TEXT)
    X_text_list_as_is = [X_text.split(' ')]
    X_text_list = [[word.lower() for word in lst] for lst in X_text_list_as_is]

    X_tags, tag_to_index_infer = get_POS_tags(X_text_list)
    X_text_list = trim_list_of_lists_upto_max_len(X_text_list, max_sentence_len)
    X_text_list_as_is = trim_list_of_lists_upto_max_len(X_text_list_as_is, max_sentence_len)
    X_tags = trim_list_of_lists_upto_max_len(X_tags, max_sentence_len)

    alnum, numeric, alpha, digit, lower, title, ascii = enrich_data(X_text_list_as_is)

    alnum = pad_and_stack_list_of_list(
        alnum,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )
    numeric = pad_and_stack_list_of_list(
        numeric,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )
    alpha = pad_and_stack_list_of_list(
        alpha,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )
    digit = pad_and_stack_list_of_list(
        digit,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )
    lower = pad_and_stack_list_of_list(
        lower,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )
    title = pad_and_stack_list_of_list(
        title,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )
    ascii = pad_and_stack_list_of_list(
        ascii,
        max_sentence_len=max_sentence_len,
        pad_value=-1,
        tensor_type=torch.FloatTensor,
    )

    x_enriched_features = torch.stack(
        (alnum, numeric, alpha, digit, lower, title, ascii), dim=2
    )

    x_encoded = [x_encoder.encode(text) for text in X_text_list]
    x_padded = [pad_tensor(tensor, max_sentence_len) for tensor in x_encoded]
    x_padded = torch.LongTensor(torch.stack(x_padded))

    x_char_padded = [
        [pad_tensor(x_char_encoder.encode(char[:max_word_length]), max_word_length) for char in word]
        for word in X_text_list_as_is
    ]
    x_char_padded = [
        pad_tensor(torch.stack(lst), max_sentence_len) for lst in x_char_padded
    ]
    x_char_padded = torch.stack(x_char_padded).type(torch.LongTensor)


    x_postag_padded = tokenize_pos_tags(
        X_tags, tag_to_index=tag_to_index, max_sen_len=max_sentence_len
    )

    out_tuple = predict(model, x_padded, x_postag_padded, x_char_padded, x_enriched_features, args.RESTRICT_IF_NO_BEG)
    print("\nOutput:")
    print(out_tuple[0])
