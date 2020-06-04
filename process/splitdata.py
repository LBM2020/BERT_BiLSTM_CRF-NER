import os
import numpy as np
from pathlib import Path
from process.dataprocess import processer

def split_data(args,file_name_or_path):

    pro = processer()
    labellist = pro.get_labels()

    file_name_or_path = os.path.join(file_name_or_path,'data.txt')
    if  not os.path.exists(file_name_or_path):
        os.makedirs(file_name_or_path)
    with open(file_name_or_path,'r') as rf:
        lines = rf.readlines()

    train_samples = []
    valid_samples = []
    for label in labellist:
        #***获取每种标签的数据***
        samples = [line for line in lines if line.split('\t')[3].replace('\n','') == label]
        np.random.shuffle(samples)

        #***每种标签数据切分成训练集和验证集***
        valid_size = int(len(samples) * args.valid_size)#会出现小数，所以需要int
        train_sample = samples[valid_size:]
        valid_sample = samples[:valid_size]

        train_samples.extend(train_sample)
        valid_samples.extend(valid_sample)

    #***添加了每种类别的数据以后将数据打乱***
    np.random.shuffle(train_samples)
    np.random.shuffle(valid_samples)

    #***写入文件***
    with open(f'{args.train_file_path}/train.txt','w') as wf:
        for line in train_samples:
            wf.write(line)

    with open(f'{args.train_file_path}/valid.txt','w') as wf:
        for line in valid_samples:
            wf.write(line)

def write_pre_result_to_file(args,predict_label):
    path = Path(f'{args.output_dir}/predict_result')
    if path not exists():
        os.mkdir(path)
    
    with open(os.path.join(path,'predict_result.txt'), 'w') as wf:
        for line in predict_label:
            wf.write(line + '\n')
