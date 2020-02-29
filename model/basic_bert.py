from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, \
    BertForMaskedLM, BertForSequenceClassification
import os, torch
from util.logging import args


def get_test_model(output_model_file) -> BertForSequenceClassification:
    # output_model_file = os.path.join('save', "finetuned_pytorch_model.bin")
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    model_state_dict = torch.load(output_model_file)
    test_model = BertForSequenceClassification.from_pretrained(args['bert_model'], num_labels=40,
                                                               state_dict=model_state_dict)
    return test_model.to(device).eval()
