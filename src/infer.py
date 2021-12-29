import argparse
import preprocess_infer
from preprocess_infer import *
from dataset import *
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import torch.nn.functional as F
from model import *

def infer(df, bpe, vocab, model): 
    labels_to_ids = {'B-DES': 1, 'B-PRI': 3, 'I-DES': 2, 'I-PRI': 4, 'O': 0, 'X': -100}
    # ids_to_labels = {0: 'O', 1: 'B-DES', 2: 'I-DES', 3:'B-PRI', 4:'I-PRI'}

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--bpe-codes', 
    #     default="./PhoBERT_base_transformers/bpe.codes",
    #     required=False,
    #     type=str,
    #     help='path to fastBPE BPE'
    # )
    # args, unknown = parser.parse_known_args()
    # bpe = fastBPE(args)

    # Load the dictionary
    # vocab = Dictionary()
    # vocab.add_from_file("./PhoBERT_base_transformers/dict.txt")

    MAX_LEN = 128
    data = process_df(df)

    X, Y_mask = convert_lines_infer(
        data.text_prep2.values,
        vocab,
        bpe,
        labels_to_ids,
        max_sequence_length=128)

    # checkpoint_path = './checkpoints/Roberta_LSTMCRF_CNN.pth'
    # model = Roberta_LSTMCRF_CNN(config)
    # model.to(device)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    X, Y_mask = torch.tensor(X).to(device), torch.tensor(Y_mask).to(device)


    model.eval()
    with torch.no_grad():
        outputs = model(X, Y_mask)

    w_l = []
    for i, r in enumerate(outputs[1]):
        b = Y_mask[i].cpu().numpy()[::-1]
        w_l.append(r[1:len(b) - np.argmax(b) - 1].cpu().type(torch.int).tolist())


    joint = pd.DataFrame()
    joint['sentence'] = data['text_prep2'].apply(lambda row: row.split())
    joint['word_labels'] = pd.Series(w_l)
    joint['label'] = pd.Series(outputs[0].tolist())
    return joint
