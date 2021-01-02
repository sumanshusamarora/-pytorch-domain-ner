import torch
import mlflow.pytorch
from train_no_binary import (
    load_data,
    get_POS_tags,
    trim_list_of_lists_upto_max_len,
    tokenize_pos_tags,
    pad_and_stack_list_of_list,
    enrich_data,
)
import dill
from torchnlp.encoders.text import pad_tensor
from torchnlp.datasets.dataset import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SENTENCE_LEN = 800
test_index = [
    4,
    205,
    15,
    23,
    217,
    187,
    200,
    98,
    143,
    80,
    213,
    69,
    55,
    181,
    22,
    83,
    202,
    166,
    30,
    53,
    199,
    142,
    89,
    31,
    138,
    19,
    142,
    76,
    122,
    79,
    29,
    182,
    23,
    194,
    23,
    108,
    1,
    54,
    110,
    218,
    196,
    200,
    163,
    140,
]
MAX_WORD_LENGTH = 307
RUN_ID = "83a77c6d613c436e9ad427009b2dacff"

models = mlflow.pytorch.load_model(
    f"file:///home/sam/work/research/ner-domain-specific/mlruns/1/{RUN_ID}/artifacts/ner_model"
)
models = models.to(device)


# X Encoder load
with open(f"mlruns/1/{RUN_ID}/artifacts/x_encoder", "rb") as infile:
    x_encoder = dill.load(infile)

with open(f"artifacts/{RUN_ID}/x_char_encoder", "rb") as infile:
    x_char_encoder = dill.load(infile)

with open(f"artifacts/{RUN_ID}/x_postag_encoder", "rb") as infile:
    x_postag_encoder = dill.load(infile)

with open(f"artifacts/{RUN_ID}/y_ner_encoder", "rb") as infile:
    y_ner_encoder = dill.load(infile)

with open(f"artifacts/{RUN_ID}/tag_to_index", "rb") as infile:
    tag_to_index = dill.load(infile)


X_text_list_as_is, X_text_list, y_ner_list = load_data()
X_tags, tag_to_index = get_POS_tags(X_text_list)
X_text_list = trim_list_of_lists_upto_max_len(X_text_list, MAX_SENTENCE_LEN)
X_text_list_as_is = trim_list_of_lists_upto_max_len(X_text_list_as_is, MAX_SENTENCE_LEN)
y_ner_list = trim_list_of_lists_upto_max_len(y_ner_list, MAX_SENTENCE_LEN)
X_tags = trim_list_of_lists_upto_max_len(X_tags, MAX_SENTENCE_LEN)


alnum, numeric, alpha, digit, lower, title, ascii = enrich_data(X_text_list_as_is)

alnum = pad_and_stack_list_of_list(
    alnum,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)
numeric = pad_and_stack_list_of_list(
    numeric,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)
alpha = pad_and_stack_list_of_list(
    alpha,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)
digit = pad_and_stack_list_of_list(
    digit,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)
lower = pad_and_stack_list_of_list(
    lower,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)
title = pad_and_stack_list_of_list(
    title,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)
ascii = pad_and_stack_list_of_list(
    ascii,
    max_sentence_len=MAX_SENTENCE_LEN,
    pad_value=-1,
    tensor_type=torch.FloatTensor,
)

x_enriched_features = torch.stack(
    (alnum, numeric, alpha, digit, lower, title, ascii), dim=2
)


x_encoded = [x_encoder.encode(text) for text in X_text_list]
x_padded = [pad_tensor(tensor, MAX_SENTENCE_LEN) for tensor in x_encoded]
x_padded = torch.LongTensor(torch.stack(x_padded))

x_char_padded = [
    [pad_tensor(x_char_encoder.encode(char), MAX_WORD_LENGTH) for char in word]
    for word in X_text_list_as_is
]
x_char_padded = [
    pad_tensor(torch.stack(lst), MAX_SENTENCE_LEN) for lst in x_char_padded
]
x_char_padded = torch.stack(x_char_padded)

x_postag_padded = tokenize_pos_tags(
    X_tags, tag_to_index=tag_to_index, max_sen_len=MAX_SENTENCE_LEN
)

y_ner_encoded = [
    [y_ner_encoder.encode(label) for label in label_list] for label_list in y_ner_list
]
y_ner_padded = [pad_tensor(torch.stack(lst), MAX_SENTENCE_LEN) for lst in y_ner_encoded]
y_ner_padded = torch.stack(y_ner_padded)

x_padded = x_padded[test_index]
x_char_padded = x_char_padded[test_index]
x_postag_padded = x_postag_padded[test_index]
y_ner_padded = y_ner_padded[test_index]
x_enriched_features = x_enriched_features[test_index]


dataset_infer = Dataset(
    [
        {
            "x_padded": x_padded[i],
            "x_char_padded": x_char_padded[i],
            "x_postag_padded": x_postag_padded[i],
            "y_ner_padded": y_ner_padded[i],
            "x_enriched_features": x_enriched_features[i],
        }
        for i in range(x_padded.shape[0])
    ]
)


dataloader_infer = DataLoader(dataset=dataset_infer, batch_size=1, shuffle=False)

for i, data_infer in enumerate(dataloader_infer):
    if i == 6:
        mask = torch.where(
            data_infer["x_padded"] > 0,
            torch.Tensor([1]).type(torch.uint8),
            torch.Tensor([0]).type(torch.uint8),
        )
        out, decoded, crf_loss = models(
            data_infer["x_padded"].to(device),
            data_infer["x_postag_padded"].to(device),
            data_infer["x_char_padded"].to(device),
            data_infer["x_enriched_features"].to(device),
            mask.to(device),
            train=False,
        )

        result = [word for word in decoded[0]]

        sentence = [x_encoder.index_to_token[ind] for ind in data_infer["x_padded"][0]]
        result_y = [y_ner_encoder.index_to_token[word] for word in result]
        true_y = [y_ner_encoder.index_to_token[word] for word in result]
        y_true = [
            y_ner_encoder.index_to_token[word] for word in data_infer["y_ner_padded"][0]
        ]
        out_tuple = tuple(zip(sentence, result_y, y_true))
