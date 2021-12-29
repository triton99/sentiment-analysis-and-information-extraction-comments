import requests
import pandas as pd
# from bs4 import BeautifulSoup
from annotated_text import annotated_text
import streamlit as st
import json
import re

import argparse
import preprocess_infer
from preprocess_infer import *
from dataset import *
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import torch.nn.functional as F
from model import *
from infer import *

st.set_page_config(page_title='Demo')

if 'cookies' not in st.session_state:
    st.session_state.cookies = {
        "user": "en"
                                }
if 'req_headers' not in st.session_state:
    st.session_state.req_headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "x-requested-with": "XMLHttpRequest",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
if 'map' not in st.session_state:
    st.session_state.map = ["Thiết kế", "Giá cả"]
if 'link' not in st.session_state:
    st.session_state.link = 'https://tiki.vn/ao-thun-nam-ngan-tay-5s-co-tron-tso21003-chat-lieu-thun-mem-mat-ben-mau-p85906869.html'

if 'model' not in st.session_state:
    checkpoint_path = './checkpoints/Roberta_LSTMCRF_CNN.pth'
    model = Roberta_LSTMCRF_CNN(config)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    st.session_state.model = model

if 'vocab' not in st.session_state:
    vocab = Dictionary()
    vocab.add_from_file("./PhoBERT_base_transformers/dict.txt")
    st.session_state.vocab = vocab

if 'bpe' not in st.session_state:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes', 
        default="./PhoBERT_base_transformers/bpe.codes",
        required=False,
        type=str,
        help='path to fastBPE BPE'
    )
    args, unknown = parser.parse_known_args()
    bpe = fastBPE(args)
    st.session_state.bpe = bpe



def update_df():    
    id = int(re.search('(?<=-p)(.\d+)(?=.html)', st.session_state.link1).groups()[0])
    # st.write('Update')
    # st.write(id)
    revs_dict = dict(content = [], star = [])
    # for id in product_id_list:
    response_data = requests.request("GET" ,'https://tiki.vn/api/v2/reviews?include=comments&product_id={}&limit=1000'.format(id), headers=st.session_state.req_headers, verify=False)
    revs = response_data.json()['data']
    # st.write('https://tiki.vn/api/v2/reviews?include=comments&product_id={}&limit=1000'.format(id))
    for r in revs:
        if r['content'] != "":
            revs_dict['content'].append(r['content'])
            revs_dict['star'].append(r['rating'])
    df = pd.DataFrame(revs_dict)
    # st.write(df.shape)
    # def label(x):
    #     if x==1: return 2
    #     elif x in [2, 3]: return 1
    #     elif x in [4, 5]: return 0
    # df['label'] = df['star'].apply(lambda x: label(x))
    # df_des = df.sample(frac=0.5)
    # df_pri = df.drop(df_des.index)

    # joint = pd.read_csv('data_joint.csv', index_col=0).sample(n=50)
    # label_dict = {
    # '(\\b)O(\\b)': '0',
    # '(\\b)(B-DES)(\\b)': '1',
    # '(\\b)(I-DES)(\\b)': '2',
    # '(\\b)(B-PRI)(\\b)': '3',
    # '(\\b)(I-PRI)(\\b)': '4'
    #             }
    # joint[['word_labels']] = joint[['word_labels']].replace(label_dict, regex=True)
    # joint['sentence'] = joint['sentence'].apply(lambda row: row.split())
    # joint['word_labels'] = joint['word_labels'].apply(lambda row: row.split(','))
    # joint['label'] = joint['label'] + 1
    joint = infer(df, st.session_state.bpe, st.session_state.vocab, st.session_state.model)

    mask_des = joint['word_labels'].apply(lambda x: (1 in x) or (2 in x))
    mask_pri = joint['word_labels'].apply(lambda x: (3 in x) or (4 in x))

    df_des = joint[mask_des]
    df_pri = joint[mask_pri]


    # df_des = joint[joint['word_labels'].str.contains(pat='1|2', regex=True)]
    # df_pri = joint[joint['word_labels'].str.contains(pat='3|4', regex=True)]

    # df_des['sentence'] = df_des['sentence'].apply(lambda row: row.split())
    # df_des['word_labels'] = df_des['word_labels'].apply(lambda row: row.split(','))

    # df_pri['sentence'] = df_pri['sentence'].apply(lambda row: row.split())
    # df_pri['word_labels'] = df_pri['word_labels'].apply(lambda row: row.split(','))

    df_dict = {'d': df_des, 'p': df_pri}
    st.session_state.df_dict = df_dict


if 'option' not in st.session_state:
    st.session_state.option = 'd'

if 'df_dict' not in st.session_state:
    # st.write('Init')
    id = int(re.search('(?<=-p)(.\d+)(?=.html)', st.session_state.link).groups()[0])
    # st.write(id)
    revs_dict = dict(content = [], star = [])
    # for id in product_id_list:
    response_data = requests.request("GET" ,'https://tiki.vn/api/v2/reviews?include=comments&product_id={}&limit=1000'.format(id), headers=st.session_state.req_headers, verify=False)
    revs = response_data.json()['data']
    # st.write('https://tiki.vn/api/v2/reviews?include=comments&product_id={}&limit=1000'.format(id))
    for r in revs:
        if r['content'] != "":
            revs_dict['content'].append(r['content'])
            revs_dict['star'].append(r['rating'])
    df = pd.DataFrame(revs_dict)
    # st.write(df.shape)
    # def label(x):
    #     if x==1: return 2
    #     elif x in [2, 3]: return 1
    #     elif x in [4, 5]: return 0
    # df['label'] = df['star'].apply(lambda x: label(x))
    # df_des = df.sample(frac=0.5)
    # df_pri = df.drop(df_des.index)

    joint = infer(df, st.session_state.bpe, st.session_state.vocab, st.session_state.model)
    # joint = pd.read_csv('data_joint.csv', index_col=0).sample(n=50)
    # label_dict = {
    # '(\\b)O(\\b)': '0',
    # '(\\b)(B-DES)(\\b)': '1',
    # '(\\b)(I-DES)(\\b)': '2',
    # '(\\b)(B-PRI)(\\b)': '3',
    # '(\\b)(I-PRI)(\\b)': '4'
    #             }
    # joint[['word_labels']] = joint[['word_labels']].replace(label_dict, regex=True)
    # joint['sentence'] = joint['sentence'].apply(lambda row: row.split())
    # joint['word_labels'] = joint['word_labels'].apply(lambda row: row.split(','))
    # joint['label'] = joint['label'] + 1

    mask_des = joint['word_labels'].apply(lambda x: (1 in x) or (2 in x))
    mask_pri = joint['word_labels'].apply(lambda x: (3 in x) or (4 in x))

    df_des = joint[mask_des]
    df_pri = joint[mask_pri]


    # df_des = joint[joint['word_labels'].str.contains(pat='1|2', regex=True)]
    # df_pri = joint[joint['word_labels'].str.contains(pat='3|4', regex=True)]

    # df_des['sentence'] = df_des['sentence'].apply(lambda row: row.split())
    # df_des['word_labels'] = df_des['word_labels'].apply(lambda row: row.split(','))

    # df_pri['sentence'] = df_pri['sentence'].apply(lambda row: row.split())
    # df_pri['word_labels'] = df_pri['word_labels'].apply(lambda row: row.split(','))

    df_dict = {'d': df_des, 'p': df_pri}
    st.session_state.df_dict = df_dict

st.session_state.link = st.text_input('Nhập đường dẫn sản phẩm:', value='https://tiki.vn/ao-thun-nam-ngan-tay-5s-co-tron-tso21003-chat-lieu-thun-mem-mat-ben-mau-p85906869.html', on_change=update_df, key='link1')
st.session_state.option = ['d', 'p'][st.session_state.map.index(st.selectbox("Chọn khía cạnh:", ("Thiết kế", "Giá cả")))]
# st.write(st.session_state.option)
# st.write(st.session_state.map.index('Thiết kế'))
# cont = ["quần", "đẹp", ",", "giá", "rẻ", ".", "shop", "đóng_gói", "kĩ", "và", "giao", "hàng", "nhanh", ".", "cám_ơn", "shop"]
# l = [1,1,0,3,4,0,0,0,0,0,0,0,0,0,0,0]
# print_a = []
# if st.session_state.option == 'p':
#     for i in range(len(l)):
#         if (l[i] in ['3','4']):
#             print_a.append((cont[i].replace("_", " ")+" ", "", "#faa"))
#         else: print_a.append(cont[i].replace("_", " ")+" ")
# elif st.session_state.option == 'd':
#     for i in range(len(l)):
#         if (l[i] in ['1','2']):
#             print_a.append((cont[i].replace("_", " ")+" ", "", "#faa"))
#         else: print_a.append(cont[i].replace("_", " ")+" ")

# elif st.session_state.option == 'd':
# annotated_text(*print_a)


def tabs(default_tabs = [], default_active_tab=0, option = 'd'):
        if not default_tabs:
            return None
        active_tab = st.radio("", default_tabs, index=default_active_tab, key='tabs')
        df_op = st.session_state.df_dict[option]
        sub_df = df_op[df_op.label==default_tabs.index(active_tab)]
        for _, row in sub_df.iterrows():
            l = row.word_labels
            cont = row.sentence
            print_a = []
            if st.session_state.option == 'p':
                for i in range(min(len(cont),len(l))):
                    if (l[i] in [3,4]):
                        print_a.append((cont[i].replace("_", " ")+" ", "", "#faa"))
                    else: print_a.append(cont[i].replace("_", " ")+" ")
            elif st.session_state.option == 'd':
                for i in range(min(len(cont),len(l))):
                    if (l[i] in [1,2]):
                        print_a.append((cont[i].replace("_", " ")+" ", "", "#faa"))
                    else: print_a.append(cont[i].replace("_", " ")+" ")

            annotated_text(*print_a)
            st.write("\n")

        child = default_tabs.index(active_tab)+1
        st.markdown("""  
            <style type="text/css">
            div[role=radiogroup] > label > div:first-of-type, .stRadio > label {
               display: none;               
            }
            div[role=radiogroup] {
                flex-direction: unset
            }
            div[role=radiogroup] label {             
                border: 1px solid #999;
                background: #EEE;
                padding: 4px 12px;
                border-radius: 4px 4px 0 0;
                position: relative;
                top: 1px;
                }
            div[role=radiogroup] label:nth-child(""" + str(child) + """) {    
                background: #FFF !important;
                border-bottom: 1px solid transparent;
            }            
            </style>
        """,unsafe_allow_html=True)        
        return active_tab

active_tab = tabs(["Tiêu cực", "Trung tính", "Tích cực"], option=st.session_state.option)
# st.write('aaaaaaaaaa')


