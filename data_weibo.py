import os
from tqdm import tqdm
import numpy as np
import pandas as pd

def convert(content):
    content = content.replace("\n", "")
    content = content.replace("\u3000", "")
    content = content.replace(" ", "")
    content = content.replace("\xa0", "")
    content = content.replace("\t", "")

    str2list = list(content)
    if len(str2list) <= 256:
        return content
    else:
        list2str = "".join(content[:256])
        return list2str

def get_data(file_path):
    tmp = pd.read_csv(file_path, encoding = 'gbk')

    with open("all_data_1.txt", 'w', encoding='gbk') as f:
        for i in range(len(tmp['内容标题'])):
            content = tmp['内容标题'][i]

            if tmp['总阅读次数'][i]> 70000:
                label = 2
            elif tmp['总阅读次数'][i]< 30000:
                label = 0
            else:
                label = 1

            f.write(convert(content)+ '\t' + str(label))
            f.write('\n')
    f.close()

# get_data("2022_new.csv")
# get_data("2023_new.csv")


def random_split(data):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0 and i % 10 != 1]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
    test_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 1]
    return train_data, valid_data, test_data

with open('all_data.txt',  'r', encoding='gbk') as f:
    text = f.readlines()

train_data, valid_data, test_data = random_split(text)

with open("train.txt", 'w', encoding='utf-8') as f:
    for line in train_data:
        content = line.split('\t')[0]
        label = line.split('\t')[1]
        f.write(content + '\t' + label)
with open("dev.txt", 'w', encoding='utf-8') as f:
    for line in valid_data:
        content = line.split('\t')[0]
        label = line.split('\t')[1]
        f.write(content + '\t' + label)
with open("test.txt", 'w', encoding='utf-8') as f:
    for line in test_data:
        content = line.split('\t')[0]
        label = line.split('\t')[1]
        f.write(content + '\t' + label)