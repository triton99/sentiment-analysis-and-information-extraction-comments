import pandas as pd
import tqdm
import numpy as np

def prepare_dataset(path):
    if path.split('.')[-1]=='xlsx':
        with open(path, 'rb') as f:
            data = pd.read_excel(f)
    else:
        data = pd.read_csv(path, encoding='utf-8')
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data.rename(columns={'sentence': 'Sentence #', 'tokens': 'Word', 'tag': 'Tag'}, inplace=True)
    data['Sentence #'] = data['Sentence #'].apply(lambda x: f'Sentence: {int(x+1)}')
    
    print("Number of tags: {}".format(len(data.Tag.unique())))

    frequencies = data.Tag.value_counts()
    tags = {}
    for tag, count in zip(frequencies.index, frequencies):
        if tag != "O":
            if tag[2:5] not in tags.keys():
                tags[tag[2:5]] = count
            else:
                tags[tag[2:5]] += count
        continue

    print(sorted(tags.items(), key=lambda x: x[1], reverse=True))

    labels_to_ids = {k: v for v, k in enumerate(data.Tag.unique())}
    ids_to_labels = {v: k for v, k in enumerate(data.Tag.unique())}
    print(labels_to_ids)

    data = data.fillna(method='ffill')
    print(data)
    if path.split('.')[-1]=='csv':
        data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
        data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
    if path.split('.')[-1]=='xlsx':
        data['sentence'] = data.groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(str(v) for v in x))
        data['word_labels'] = data.groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(str(v) for v in x))
    data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)

    return data

def convert_lines(lines, tags, vocab, bpe, labels_to_ids, max_sequence_length=256):
    """
    lines: list các văn bản input
    tags: list các chuỗi tag
    vocab: từ điển dùng để encoding subwords
    bpe: 
    """
    # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
    outputs = np.zeros((len(lines), max_sequence_length), dtype=np.int32) # --> shape (number_lines, max_seq_len)
    outputs_labels = np.zeros((len(lines), max_sequence_length), dtype=np.int32)
    outputs_attention_mask = np.zeros((len(lines), max_sequence_length), dtype=np.int32)
    # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
    cls_id = 0
    eos_id = 2
    pad_id = 1
    
    for idx, row in tqdm.tqdm(enumerate(lines), total=len(lines)): 
        # Mã hóa subwords theo byte pair encoding(bpe)
        subwords = bpe.encode(row)
        subwords = '<s> '+ subwords +' </s>'
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        
        tag_list = ['O'] + tags[idx].split(',') + ['O']
        subword_idx = [subwords.split().index(word) for word in subwords.split() if '@@' in word]
        for i, orig_idx in enumerate(subword_idx):
            tag_list.insert(orig_idx+1, 'X')
        # print(tag_list)
        labels = [labels_to_ids[label] for label in tag_list] 

        # Truncate input nếu độ dài vượt quá max_seq_len
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = eos_id
            labels = labels[:max_sequence_length]
            labels[-1] = -100
        else:
        # Padding nếu độ dài câu chưa bằng max_seq_len
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
            labels = labels + [-100, ]*(max_sequence_length - len(labels))
        
        labels[0] = -100
        # print(len(labels))
        labels[np.where(np.array(input_ids)==eos_id)[0][0]] = -100
        # print(np.where(np.array(input_ids)==eos_id)[0][0])
        # labels[input_ids==eos_id] = -100
        outputs[idx,:] = np.array(input_ids)
        outputs_labels[idx,:] = np.array(labels)
        outputs_attention_mask[idx, np.array(input_ids)!=pad_id] = 1

    return outputs, outputs_labels, outputs_attention_mask


def convert_lines_infer(lines, vocab, bpe, labels_to_ids, max_sequence_length=256):
    """
    lines: list các văn bản input
    vocab: từ điển dùng để encoding subwords
    bpe: 
    """
    # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
    outputs = np.zeros((len(lines), max_sequence_length), dtype=np.int32) # --> shape (number_lines, max_seq_len)
    outputs_labels = np.zeros((len(lines), max_sequence_length), dtype=np.int32)
    outputs_attention_mask = np.zeros((len(lines), max_sequence_length), dtype=np.int32)
    # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
    cls_id = 0
    eos_id = 2
    pad_id = 1
    
    for idx, row in tqdm.tqdm(enumerate(lines), total=len(lines)): 
        # Mã hóa subwords theo byte pair encoding(bpe)
        subwords = bpe.encode(row)
        subwords = '<s> '+ subwords +' </s>'
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        
        subword_idx = [subwords.split().index(word) for word in subwords.split() if '@@' in word]

        # Truncate input nếu độ dài vượt quá max_seq_len
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = eos_id
        else:
        # Padding nếu độ dài câu chưa bằng max_seq_len
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        
        outputs[idx,:] = np.array(input_ids)
        outputs_attention_mask[idx, np.array(input_ids)!=pad_id] = 1

    return outputs, outputs_attention_mask