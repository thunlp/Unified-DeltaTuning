import os
import json
import re
import string
import random

import numpy as np

from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyQADataset, MyQAMultiDataset, MyQAPromptDataset, MyQAPromptDataset_intrinsic, MyDataLoader
from .metrics_t5 import METRICS, evaluate

class NLPFulldataGymMultiTaskData(object):

    def __init__(self, logger, args, data_path, tasks, data_split, data_type, is_training, is_test=False):
        self.data_path = data_path
        self.data_type = data_type
        
        self.data = []

        self.task2id = {}
        self.task_num = 0
        
        for task in sorted(tasks):
            task_dir = os.path.join(self.data_path, task)
            logger.info(task_dir)
            files = sorted(os.listdir(task_dir))

            prefixes = []
            for filename in files:
                if not filename.endswith(".tsv"):
                    continue
                prefix = "_".join(filename.split("_")[:-1])
                if prefix not in prefixes:
                    prefixes.append(prefix)

            for prefix in prefixes:
                # fulladata只循环一次
                with open(os.path.join(task_dir, prefix + "_train.tsv"), encoding='utf-8') as fin:
                    lines = fin.readlines()

                train_examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    # assert len(d) == 2, prefix
                    
                    train_examples.append((d[0], d[1:]))

                with open(os.path.join(task_dir, prefix + "_dev.tsv"), encoding='utf-8') as fin:
                    lines = fin.readlines()
                    
                dev_examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    # assert len(d) == 2, prefix
                    dev_examples.append((d[0], d[1:]))

                if is_test:
                    with open(os.path.join(task_dir, prefix + "_test.tsv"), encoding='utf-8') as fin:
                        lines = fin.readlines()
                        
                    test_examples = []
                    for line in lines:
                        d = line.strip().split("\t")
                        # assert len(d) == 2, prefix
                        test_examples.append((d[0], d[1:]))
                else:
                    test_examples = []

                # add auto-encoding prompts
                '''
                task_prompt = []
                prompt_weight_dir = os.path.join(args.inherit_prompt_path, 'singletask-' + task, 'prompt_weight')
                load_flag = False
                for prompt_dir in os.listdir(prompt_weight_dir):
                    if '_' + str(args.select_prefix) + '_' not in prompt_dir:
                        continue
                    if not args.recover_multiple_seeds and 'best' not in prompt_dir:
                        continue
                    if args.recover_multiple_seeds and 'best' in prompt_dir:
                        continue
                    task_prompt.append(torch.load(os.path.join(prompt_weight_dir, prompt_dir)))
                    load_flag = True
                '''
                self.data.append({
                    "task_name": task,
                    "task_prefix": prefix,
                    "train_examples": train_examples,
                    "dev_examples": dev_examples,
                    "test_examples": test_examples,
                    # "task_prompt": task_prompt,
                })
        # logger.info('found prompt in ' + str(c1) + ' tasks')
        # logger.info('did not found prmopt in ' + str(c2) + ' tasks')
        self.data_split = data_split
        self.is_training = is_training
        self.logger = logger
        self.args = args

        self.metric = METRICS
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.load = not args.debug

        self.gen_early_stop = False
        self.extra_id_0 = '<extra_id_0>'

        self.data_evaluate = []

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata
    def concat_metadata(self, meta1, meta2): #将meta1拼接到meta2后面
        if len(meta2)==0:
            return meta1
        else:
            add_item = meta2[-1][-1]
            meta1 = [list(tuple_pair) for tuple_pair in meta1]
            meta1_array = np.array(meta1)
            meta1_array = meta1_array+add_item
            meta1_list = meta1_array.tolist()
            meta1 = [tuple(list_pair) for list_pair in meta1_list]
            meta2 = meta2+meta1
        return meta2
    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        split_identifier = self.args.custom_tasks_splits.split("/")[-1]
        if split_identifier.endswith(".json"):
            split_identifier = split_identifier[:-5]

        preprocessed_path = os.path.join(
            self.data_path,
            self.data_type + "-multi-{}-{}.pth".format(split_identifier, postfix)
        )
        
        if False:
        # if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            preprocessed_data = torch.load(preprocessed_path)
            input_ids = preprocessed_data['input_ids']
            attention_mask = preprocessed_data['attention_mask']
            decoder_input_ids = preprocessed_data['decoder_input_ids']
            decoder_attention_mask = preprocessed_data['decoder_attention_mask']
            metadata = preprocessed_data['metadata']
            
        else:
            self.logger.info("Start tokenizing ... {} tasks".format(len(self.data)))

            inputs_multi = []
            outputs_multi = []
            task_prefix = []
            task_names = []
            
            input_ids_multi = torch.LongTensor([])
            attention_mask_multi = torch.LongTensor([])
            decoder_input_ids_multi = torch.LongTensor([])
            decoder_attention_mask_multi = torch.LongTensor([])
            task_names_multi = []
            metadata_multi = []
            
            input_ids_all = []
            attention_mask_all = []
            decoder_input_ids_all = []
            decoder_attention_mask_all = []
            task_names_all = []
            metadata_all = []
            lines_num = []

            idx = 0
            for task in self.data:
                idx += 1
                if self.args.debug:
                    if idx >= 10:
                        break
                task_name = task["task_name"]
                task_prefix = task["task_prefix"]
                
                # 从singletask tokenize过的文件load
                print(task_name, self.data_split)
                singletask_preprocessed_path = os.path.join(
                    self.data_path,
                    task_name,
                    task_prefix+'_{}-{}.pth'.format(self.data_split, postfix)
                    )
                
                if os.path.exists(singletask_preprocessed_path):
                    self.logger.info("Loading pre-tokenized data from {}".format(singletask_preprocessed_path))
                    preprocessed_data = torch.load(singletask_preprocessed_path)
                    input_ids = preprocessed_data['input_ids']
                    attention_mask = preprocessed_data['attention_mask']
                    decoder_input_ids = preprocessed_data['decoder_input_ids']
                    decoder_attention_mask = preprocessed_data['decoder_attention_mask']
                    metadata = preprocessed_data['metadata']
                else:
                    self.logger.info("Tokenizing {} {} ".format(task_prefix, self.data_split))
                    self.logger.info("Start tokenizing ... {} instances".format(len(task["{}_examples".format(self.data_split)])))

                    inputs = []
                    outputs = []
                    # task_names = []

                    for dp in task["{}_examples".format(self.data_split)]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        # outputs.append(self.extra_id_0 + dp[1]) # is a list
                        output = []
                        for d in dp[1]:
                            output.append(self.extra_id_0+d)
                        outputs.append(output) # is a list
                        # task_names.append(task_name)

                    self.logger.info("Printing 3 examples")
                    for i in range(3):
                        self.logger.info(inputs[i])
                        self.logger.info(outputs[i])

                    # 后续注意处理好metadata
                    outputs, metadata = self.flatten(outputs) # what is metadata?

                    if self.args.do_lowercase:
                        inputs = [input0.lower() for input0 in inputs]
                        outputs = [output0.lower() for output0 in outputs]
                    if self.args.append_another_bos:
                        inputs = ["<s> "+input0 for input0 in inputs]
                        outputs = ["<s> " +output0 for output0 in outputs]
                    
                    self.logger.info("Tokenizing Input ...")
                    tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                                #  pad_to_max_length=True,
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors="pt", 
                                                                max_length=self.args.max_input_length)
                    self.logger.info("Tokenizing Output ...")
                    tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                            #    pad_to_max_length=True,
                                                            padding='max_length',
                                                            truncation=True,
                                                            return_tensors="pt", 
                                                            max_length=self.args.max_output_length)

                    input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
                    decoder_input_ids, decoder_attention_mask = tokenized_output['input_ids'].masked_fill_(tokenized_output['input_ids'] == self.tokenizer.pad_token_id, -100), tokenized_output["attention_mask"]
                    # assert len(input_ids)==len(decoder_input_ids)
                    assert len(decoder_input_ids)==metadata[-1][-1]
                    if self.load:
                        preprocessed_data = {}
                        preprocessed_data['input_ids'] = input_ids
                        preprocessed_data['attention_mask'] = attention_mask
                        preprocessed_data['decoder_input_ids'] = decoder_input_ids
                        preprocessed_data['decoder_attention_mask'] = decoder_attention_mask
                        preprocessed_data['metadata'] = metadata
                        torch.save(preprocessed_data, singletask_preprocessed_path)
                
                
                task_names = [task_name]*len(decoder_input_ids)
                task_names_all.append(task_name)
                input_ids_all.append(input_ids)
                attention_mask_all.append(attention_mask)
                decoder_input_ids_all.append(decoder_input_ids)
                decoder_attention_mask_all.append(decoder_attention_mask)
                metadata_all.append(metadata)
                lines_num.append(len(input_ids))
                # self.data_evaluate.extend(task["{}_examples".format(self.data_split)])
                '''
                input_ids_all.append(input_ids)
                attention_mask_all.append(attention_mask)
                decoder_input_ids_all.append(decoder_input_ids)
                decoder_attention_mask_all.append(decoder_attention_mask)
                metadata_all.append(metadata)
                lines_num = len(input_ids)
                '''
                
                # 数据集train set大的采样率更大
                
                '''
                if self.data_split == "train" or self.data_split == "all":
                    for dp in task["train_examples"]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        output = []
                        for d in dp[1]:
                            output.append(self.extra_id_0+d)
                        outputs.append(output) # is a list                        
                        task_prefix.append(task["task_prefix"])
                        task_names.append(task_name)
                        self.data_evaluate.append(dp)
                if self.data_split == "dev" or self.data_split == "all":
                    for dp in task["dev_examples"]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        output = []
                        for d in dp[1]:
                            output.append(self.extra_id_0+d)
                        outputs.append(output) # is a list
                        task_prefix.append(task["task_prefix"])
                        task_names.append(task_name)
                        self.data_evaluate.append(dp)
                if self.data_split == "test":
                    for dp in task["test_examples"]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        output = []
                        for d in dp[1]:
                            output.append(self.extra_id_0+d)
                        outputs.append(output) # is a list
                        task_prefix.append(task["task_prefix"])
                        task_names.append(task_name)
                        self.data_evaluate.append(dp)
                '''
        # 每个任务train数据为self.args.lines_each_task行，少的重复，多的随机选取
        # dev 100行
        # test原数据不变
        
        # assert self.args.sample_rate == 0.1
        # min_sample_num = min(lines_num)
        if self.data_split == "train":
            for i in range(len(lines_num)):
                # if self.args.train_lines_split and lines_num[i]<=self.args.train_lines_each_task:
                #     n = self.args.train_lines_each_task
                # else:
                #     n = self.args.train_lines_each_task * 2
                n = self.args.train_lines_each_task
                if lines_num[i] >= n:
                    # 不能随机选n行，metadata混乱了                
                    # index = torch.LongTensor(random.sample(range(0, lines_num[i]), self.args.lines_each_task))
                    # # index = torch.randint(0, lines_num[i], (min_sample_num*10,1)).squeeze()
                    # input_ids_multi = torch.cat([input_ids_multi, torch.index_select(input_ids_all[i], dim=0, index=index)])
                    # attention_mask_multi = torch.cat([attention_mask_multi, torch.index_select(attention_mask_all[i], dim=0, index=index)])
                    # decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, torch.index_select(decoder_input_ids_all[i], dim=0, index=index)])
                    # decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, torch.index_select(decoder_attention_mask_all[i], dim=0, index=index)])
                    # task_names_multi.extend([task_names_all[i]] * self.args.lines_each_task)
                    
                    # 选前n行
                    input_ids_multi = torch.cat([input_ids_multi, input_ids_all[i][:n]])
                    attention_mask_multi = torch.cat([attention_mask_multi, attention_mask_all[i][:n]])
                    len_of_decoder_input_ids = metadata_all[i][n-1][-1]
                    decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, decoder_input_ids_all[i][:len_of_decoder_input_ids]])
                    decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, decoder_attention_mask_all[i][:len_of_decoder_input_ids]])
                    # 拼在一起后metadata_all[i]需要更新
                    metadata_multi = self.concat_metadata(metadata_all[i][:n], metadata_multi)
                    # metadata_multi = torch.cat([metadata_multi, metadata_all[i][:n]+metadata_multi[-1][-1]])
                    task_names_multi.extend([task_names_all[i]] * n)
                    assert len(input_ids_multi)==len(attention_mask_multi)==len(metadata_multi)
                    assert len(decoder_input_ids_multi)==len(decoder_attention_mask_multi)==metadata_multi[-1][-1]
                else:
                    # 将原数据拓展至self.args.lines_each_task行, 尤其注意metadata的处理
                    repeat_times = n // lines_num[i]
                    supply_lines = n % lines_num[i]
                    input_ids_multi = torch.cat([input_ids_multi, torch.cat([input_ids_all[i].repeat(repeat_times,1),input_ids_all[i][0:supply_lines,:]])])
                    attention_mask_multi = torch.cat([attention_mask_multi, torch.cat([attention_mask_all[i].repeat(repeat_times,1),attention_mask_all[i][0:supply_lines,:]])])
                    if supply_lines>0:
                        len_of_supply_lines_decoder_input_ids = metadata_all[i][supply_lines-1][-1]
                    else:
                        len_of_supply_lines_decoder_input_ids = 0
                    decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, torch.cat([decoder_input_ids_all[i].repeat(repeat_times,1),decoder_input_ids_all[i][0:len_of_supply_lines_decoder_input_ids,:]])])
                    decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, torch.cat([decoder_attention_mask_all[i].repeat(repeat_times,1),decoder_attention_mask_all[i][0:len_of_supply_lines_decoder_input_ids,:]])])
                    # 拼在一起后metadata_all[i]需要更新
                    for j in range(repeat_times):
                        metadata_multi = self.concat_metadata(metadata_all[i], metadata_multi)
                        # metadata_multi = torch.cat([metadata_multi, metadata_all[i]+metadata_multi[-1][-1]])
                    if supply_lines>0:
                        metadata_multi = self.concat_metadata(metadata_all[i][0:supply_lines], metadata_multi)
                        # metadata_multi = torch.cat([metadata_multi, metadata_all[i][0:supply_lines]+metadata_multi[-1][-1]])
                    task_names_multi.extend([task_names_all[i]] * n)
                    assert len(input_ids_multi)==len(attention_mask_multi)==len(metadata_multi)
                    assert len(decoder_input_ids_multi)==len(decoder_attention_mask_multi)==metadata_multi[-1][-1]
        elif self.data_split == "dev":
            for i in range(len(lines_num)):
                n = self.args.dev_lines_each_task
                if lines_num[i]>=n:
                    input_ids_multi = torch.cat([input_ids_multi, input_ids_all[i][:n]])
                    attention_mask_multi = torch.cat([attention_mask_multi, attention_mask_all[i][:n]])
                    len_of_decoder_input_ids = metadata_all[i][n-1][-1]
                    decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, decoder_input_ids_all[i][:len_of_decoder_input_ids]])
                    decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, decoder_attention_mask_all[i][:len_of_decoder_input_ids]])
                    # 拼在一起后metadata_all[i]需要更新
                    metadata_multi = self.concat_metadata(metadata_all[i][0:n], metadata_multi)
                    # metadata_multi = torch.cat([metadata_multi, metadata_all[i][:99]+metadata_multi[-1][-1]])
                    task_names_multi.extend([task_names_all[i]] * n)
                    self.data_evaluate.extend(self.data[i]["{}_examples".format(self.data_split)][:n])
                    assert len(input_ids_multi)==len(attention_mask_multi)==len(metadata_multi)
                    assert len(decoder_input_ids_multi)==len(decoder_attention_mask_multi)==metadata_multi[-1][-1]
                else:
                    
                    repeat_times = n // lines_num[i]
                    supply_lines = n % lines_num[i]
                    input_ids_multi = torch.cat([input_ids_multi, torch.cat([input_ids_all[i].repeat(repeat_times,1),input_ids_all[i][0:supply_lines,:]])])
                    attention_mask_multi = torch.cat([attention_mask_multi, torch.cat([attention_mask_all[i].repeat(repeat_times,1),attention_mask_all[i][0:supply_lines,:]])])
                    if supply_lines>0:
                        len_of_supply_lines_decoder_input_ids = metadata_all[i][supply_lines-1][-1]
                    else:
                        len_of_supply_lines_decoder_input_ids = 0
                    decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, torch.cat([decoder_input_ids_all[i].repeat(repeat_times,1),decoder_input_ids_all[i][0:len_of_supply_lines_decoder_input_ids,:]])])
                    decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, torch.cat([decoder_attention_mask_all[i].repeat(repeat_times,1),decoder_attention_mask_all[i][0:len_of_supply_lines_decoder_input_ids,:]])])
                    # 拼在一起后metadata_all[i]需要更新
                    for j in range(repeat_times):
                        metadata_multi = self.concat_metadata(metadata_all[i], metadata_multi)
                        # metadata_multi = torch.cat([metadata_multi, metadata_all[i]+metadata_multi[-1][-1]])
                    if supply_lines>0:
                        metadata_multi = self.concat_metadata(metadata_all[i][0:supply_lines], metadata_multi)
                        # metadata_multi = torch.cat([metadata_multi, metadata_all[i][0:supply_lines]+metadata_multi[-1][-1]])
                    task_names_multi.extend([task_names_all[i]] * n)
                    data_evaluate_repeat = self.data[i]["{}_examples".format(self.data_split)] * repeat_times + self.data[i]["{}_examples".format(self.data_split)][0:supply_lines]
                    self.data_evaluate.extend(data_evaluate_repeat)
                    assert len(input_ids_multi)==len(attention_mask_multi)==len(metadata_multi)
                    assert len(decoder_input_ids_multi)==len(decoder_attention_mask_multi)==metadata_multi[-1][-1]
                    '''
                    input_ids_multi = torch.cat([input_ids_multi, input_ids_all[i]])
                    attention_mask_multi = torch.cat([attention_mask_multi, attention_mask_all[i]])
                    decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, decoder_input_ids_all[i]])
                    decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, decoder_attention_mask_all[i]])
                    metadata_multi = self.concat_metadata(metadata_all[i], metadata_multi)
                    # metadata_multi = torch.cat([metadata_multi, metadata_all[i]+metadata_multi[-1][-1]])
                    task_names_multi.extend([task_names_all[i]] * lines_num[i])
                    self.data_evaluate.extend(self.data[i]["{}_examples".format(self.data_split)])
                    assert len(input_ids_multi)==len(attention_mask_multi)==len(metadata_multi)
                    assert len(decoder_input_ids_multi)==len(decoder_attention_mask_multi)==metadata_multi[-1][-1]
                    '''
            self.task_names_multi = task_names_multi
        elif self.data_split == "test":
            for i in range(len(lines_num)):
                input_ids_multi = torch.cat([input_ids_multi, input_ids_all[i]])
                attention_mask_multi = torch.cat([attention_mask_multi, attention_mask_all[i]])
                decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, decoder_input_ids_all[i]])
                decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, decoder_attention_mask_all[i]])
                metadata_multi = self.concat_metadata(metadata_all[i], metadata_multi)
                # metadata_multi = torch.cat([metadata_multi, metadata_all[i]+metadata_multi[-1][-1]])
                task_names_multi.extend([task_names_all[i]] * lines_num[i])
                self.data_evaluate.extend(self.data[i]["{}_examples".format(self.data_split)])
                assert len(input_ids_multi)==len(attention_mask_multi)==len(metadata_multi)
                assert len(decoder_input_ids_multi)==len(decoder_attention_mask_multi)==metadata_multi[-1][-1]
            self.task_names_multi = task_names_multi
        # for i in range(len(lines_num)):
        #     input_ids_multi = torch.cat([input_ids_multi, input_ids_all[i]])
        #     attention_mask_multi = torch.cat([attention_mask_multi, attention_mask_all[i]])
        #     decoder_input_ids_multi = torch.cat([decoder_input_ids_multi, decoder_input_ids_all[i]])
        #     decoder_attention_mask_multi = torch.cat([decoder_attention_mask_multi, decoder_attention_mask_all[i]])
        #     task_names_multi.extend([task_names_all[i]] * lines_num[i])
        #     self.data_evaluate.extend(self.data[i]["{}_examples".format(self.data_split)])
        
        # input_ids_multi = torch.Tensor(input_ids_all)
        # attention_mask_multi = torch.Tensor(attention_mask_all)
        # decoder_input_ids_multi = torch.Tensor(decoder_input_ids_all)
        # decoder_attention_mask_multi = torch.Tensor(decoder_attention_mask_all)
        # task_names_multi = task_names_all
        
        self.dataset = MyQAMultiDataset(input_ids_multi, attention_mask_multi,
                                        decoder_input_ids_multi, decoder_attention_mask_multi,
                                        task_names_multi,
                                        in_metadata=None, out_metadata=metadata_multi,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data from {} tasks".format(len(self.dataset), self.data_type, len(self.data)))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        task_names = self.task_names_multi
        assert len(predictions)==len(self.data_evaluate), (len(predictions), len(self.data_evaluate))
        predictions = [prediction.strip() for prediction in predictions]
        task2id = {}
        for idx, task_name in enumerate(task_names):
            if task_name not in task2id:
                task2id[task_name] = []
            task2id[task_name].append(idx)
        task2score = {}
        for task, ids in task2id.items():
            task2score[task] = evaluate([predictions[x] for x in ids], [self.data_evaluate[x] for x in ids], self.metric[task])
        score_list = []
        for i in range(len(task2score)):
            score_list.append(list(list(task2score.values())[i].values())[0])
        task2score = {}
        for idx, task_name in enumerate(list(task2id.keys())):
            task2score[task_name] = score_list[idx]
        metric = {}
        metric['mean_performance'] = np.mean(np.array(score_list))
        return task2score, metric

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip())==0 else prediction for prediction in predictions]
        prediction_text = [prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(self.args.output_dir, "{}_predictions.txt".format(self.args.prefix))
        with open(save_path, "w", encoding='utf-8') as f:
            f.writelines(prediction_text)
        
        self.logger.info("Saved prediction in {}".format(save_path))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([f1_score(prediction, gt) for gt in groundtruth])
    return f1_score(prediction, groundtruth)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


