import numpy as np
import torch
import os
from torch import Tensor
# from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from util.logging import args, logger
from util.data_processing import convert_single
from util.data_structure import LabelTextProcessor


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    #     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
          sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()


def set_eval(model, eval_dataloader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    count = 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

        count += 1
        if count >= 4:
            break

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    #     loss = tr_loss/nb_tr_steps if tr_loss else None
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': 0}
    #               'loss': loss}

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


def single_eval(single_text, model: BertForSequenceClassification, tokenizer=None):
    processors = {
        "news_cat_label": LabelTextProcessor
    }
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    eval_feature = convert_single(single_text, tokenizer=tokenizer)

    all_input_ids = torch.tensor([eval_feature.input_ids], dtype=torch.long)
    all_input_mask = torch.tensor([eval_feature.input_mask], dtype=torch.long)
    all_segment_ids = torch.tensor([eval_feature.segment_ids], dtype=torch.long)
    # all_label_ids = torch.tensor([eval_feature.label_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(all_input_ids.to(device), all_input_mask.to(device), all_segment_ids.to(device))

    logits = logits.detach().cpu().numpy()
    return logits[0]


def print_single(logits: list, label_list: list):
    assert logits.__len__() == label_list.__len__()
    pred_ind = np.argmax(logits)
    # noinspection PyTypeChecker
    print('Prediction: {}'.format(label_list[pred_ind]))

    for i in range(logits.__len__()):
        print('{}: {}'.format(label_list[i], logits[i]))
