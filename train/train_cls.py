from model.basic_bert import BertForSequenceClassificationExt as BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
import os
from tqdm import trange, tqdm
from util.logging import args, logger
from util.meta import label_list
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler  # , SequentialSampler
from util.data_processing import convert_examples_to_features
from util.data_structure import LabelTextProcessor
from train import opt
from util.eval import set_eval
from torch.utils.tensorboard import SummaryWriter
from general import ROOT
from time import gmtime, strftime

device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def fit(model, train_dataloader, test_dataloader, optimizer, num_train_steps, num_epocs=5):
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    writer_name = os.path.join(ROOT, 'log', time_string)
    print(writer_name)
    writer = SummaryWriter(writer_name)
    n_gpu = 1
    global_step = 0
    output_model_file = os.path.join(ROOT, 'save', "finetuned_pytorch_model_3.bin")
    t_total = num_train_steps
    #     model_state_dict = torch.load(output_model_file)
    #     model = BertForSequenceClassification.from_pretrained(args['bert_model'], num_labels = num_labels, state_dict=model_state_dict)
    #     model.to(device)
    model.train()

    for i_ in trange(int(num_epocs), desc="Epoch"):
        # Load a trained model that you have fine-tuned
        print('hehe')
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                #   scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                # lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % 10 == 0:
                logger.info('Epoch {} Step {} Loss {}'.format(i_, step, tr_loss / nb_tr_steps))
                writer.add_scalar('loss', torch.mean(loss).detach().cpu().numpy(), step)
            if (step + 1) % 100 == 0:
                r = set_eval(model, test_dataloader)
                writer.add_scalar('evalloss', r['eval_loss'], step)
                writer.add_scalar('evalacc', r['eval_accuracy'], step)

        # logger.info('Eval after epoc {}'.format(i_ + 1))
        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)


def train():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=args['do_lower_case'])
    train_examples = None
    num_train_steps = None

    processors = {
        "news_cat_label": LabelTextProcessor
    }
    processor = processors[args['task_name'].lower()](args['data_dir'])
    if args['do_train']:
        train_examples = processor.get_train_examples(args['full_data_dir'], size=args['train_size'])
        num_train_steps = int(
            len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args[
                'num_train_epochs'])
    eval_examples = processor.get_dev_examples(args['data_dir'], size=args['val_size'])

    train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    model = BertForSequenceClassification.from_pretrained(args['bert_model'], num_labels=label_list.__len__())
    model = model.to(device)
    _, optimizer = opt.get_opt(model, num_train_steps)

    fit(model, train_dataloader, eval_dataloader, optimizer, num_train_steps, args['num_train_epochs'])


if __name__ == '__main__':
    train()
