# SENTIMENT ANALYSIS AND INFORMATION EXTRACTION FOR PRODUCT COMMENTS

<details> 
<summary>Contents</summary>

  - [1. Overview](#1-overview)
    - [Introduction](#introduction)
    - [Project structure](#neural-network-model)
  - [2. Install the required environment and libraries](#2-install-the-required-environment-and-libraries)
  - [3. Running Demo](#3-running-demo)
  - [4. Evaluation](#4-evaluation)

</details>

## 1. Overview
### Introduction
This is my NLP subject project in VinBigData Institute with my teammates (Duy Tran, Luc Nguyen, Tu Phan).
#### Topic
We automatically collect factual data from Vietnamese e-commerce sites and categorize comments based on emotions, then extract information to explain why customers felt the same way when they made a purchase.

#### Our Dataset
Our dataset focuses mainly on the fashion industry on the e-commerce platform [Tiki](https://tiki.vn/). This dataset contains 3791 comments separated by 3 classes (Positive, Neutral, Negative) and tagged by sequence (Design, Price). 

Link download dataset [here](https://drive.google.com/drive/folders/1A_BGOEztTtxi1zjE7UF_Z7mBhTh_ZABE?usp=sharing).

#### Joint model
The structure of the proposed joint model:

<img src="/images/joint_model.png" width="600" height="300" />


## 2. Install the required libraries
The libraries used in this project are listed in the file [requirements.txt](#requirements.txt):
- `fairseq==0.10.2`
- `numpy==1.21.2`
- `pandas==1.2.4`
- `requests==2.25.1`
- `scikit_learn==1.0.2`
- `st_annotated_text==2.0.0`
- `streamlit==1.3.0`
- `torch==1.10.1`
- `torchcrf==1.1.0`
- `tqdm==4.59.0`
- `transformers==4.15.0`

To install the above libraries, execute the following command:
>`pip install -r requirements.txt`

## 3. Running demo
Download [checkpoint](https://drive.google.com/file/d/1SEJbZJIO3crQibZrf0xoKfQCrs_YJH4A/view?usp=sharing) of our model and put it to `checkpoints` folder.

Clone this repo and go to the home directory:
> `git clone https://github.com/triton99/sentiment-analysis-and-information-extraction-comments.git`

> `cd sentiment-analysis-and-information-extraction-comments`

Install the vncorenlp python wrapper
> `pip3 install vncorenlp -q`

Download VnCoreNLP-1.1.1.jar & its word segmentation component
> `mkdir -p vncorenlp/models/wordsegmenter`
>
> `wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar`
>
> `wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab`
>
> `wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr`
>
> `mv VnCoreNLP-1.1.1.jar vncorenlp/ `
>
> `mv vi-vocab vncorenlp/models/wordsegmenter/`
>
> `mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/`

Download ngrok

> `wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip`
>
> `unzip ngrok-stable-linux-amd64.zip`

Run command system
> 'get_ipython().system_raw('./ngrok authtoken [YourAuthtoken]')'

> `get_ipython().system_raw('./ngrok http 8501 &')`

Get demo link
> `curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Execute the next cell and the go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])'`

Run demo
> `streamlit run src/app.py`

### Demo result
You can access Tiki website, get your favorite product link and paste it to search bar.

<img src="/images/demo.jpeg" width="800" height="350" />


## 4. Evaluation
- F1 score: the F1 score is the harmonic mean of the precision and recall.

|      Model      | F1-Classification | F1-Score Tagging |
| --------------- | ----------------- | ---------------- |
| Joint model     |       0,81        |       0,93       |
| RobertaLSTM_CRF |       ----        |       0,94       |
| RobertaCNN4     |       0.79        |       ----       |
