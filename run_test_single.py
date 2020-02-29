from model import basic_bert
import os
from util import eval
from util.meta import label_list

if __name__ == '__main__':
    sentence = 'Will Smith Joins Diplo And Nicky Jam For The 2018 World Cup\'s Official Song Of course it has a song.'
    model_path = os.path.join('save', "finetuned_pytorch_model.bin")
    model = basic_bert.get_test_model(model_path)
    logits = eval.single_eval(sentence, model)
    eval.print_single(logits, label_list)
