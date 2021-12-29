from vncorenlp import VnCoreNLP
import re
import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
rdrsegmenter = VnCoreNLP("/content/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

dict_vn = {'dc': "được", "ko": 'không', "r": "rồi", "j": "gì", "lm": "làm", 
          "ah": "à", "k": "không", "kh": "khách_hàng", "wa": "quá", "nx": "nữa",
          "v": "vậy", "đm": "", "vl": "", "cc": "", "sp": "sản phẩm", "onl": "online",
          "bp": "Bộ phận", "cskh": "'chăm_sóc_khách_hàng", "ae": "anh_em", "ns":"nói",
          "dk": "được", "yc": "yêu_cầu", "ib": "nhắn tin", "đc": "được",
          "ktra": "kiểm_tra", "sz": 'size', "vs": "với", "mn": "mọi_người", "z": "vậy",
          "g": "giờ", "qc": "quảng_cáo", 'bj' : 'bây_giờ', "ak": "á", "mink": "mình",
          "trc": "trước", "thik":"thích", "tl" : "trả_lời", "rep" : "trả lời", "tn": "tin_nhắn",
          "khog": "không", "nv": "nhân_viên", "fai": "phải", "hk": "không", "c": "cái",
          "m": "mình", "kq": "kết_quả", "v": "vậy", "vc": "việc", "lh" : "liên_hệ", "đg" : "đường",
          "ng": "người", "sf": "sản_phẩm", "nhug": "nhưng", "ms": "mới", "s": "sao",
          "ạk": "ạ", "oce": "ổn", "cx": 'cũng', "sd": "sử_dụng", "trc": "trước", 
          "kô": "không", "mak": "mà", "đk": 'đăng_kí', "tt": "trung_tâm", "tp": "thành_phố",
          "ntn": "như_thế_nào", "mún": "muốn", "mng": "mọi_người", "c.on": "cảm_ơn", 
          "ox": "ông_xã", "bth": "bình_thường", "dt": "điện_thoại", "vn": "việt_nam",
          "hcm": "hồ_chí_minh", "nhìu": "nhiều", "lf": "là", 'l1': "lần_một", 'l2': "lần_hai",
          "kick": "kích", 'ln': "luôn", ":v": "", ":vvv": "", ":/": "", "bt": "bình_thường",
          "b": "bạn", "mk": "mình", "hok": "không", "phom": "dáng", "m": "mình", 
           "sdt": "số_điện_thoại", ":))": "", ":))))": "", ":(": "", "kt": "kiểm_tra",
          "snghi": "suy_nghĩ", "sl": "số_lượng", "okila": "ổn", "okela": "ổn",
          "lun": "luôn", "nt": "nhắn_tin", "oki": "ổn", "ok": "ổn", "bx": "bà_xã",
          "ch": "chuyển", "mg": "mọi_người", "sg": "sài_gòn", 'nhma': "nhưng_mà",
          "bb": "bạn_bè", "trg": "trong", "bít": "biết", "k0": "không", "=": "bằng",
          "huhuhu": "rất_buồn", "nchung": "nói_chung", "đag": "đang", "bik": "biết",
          "nc": "nói_chung", "in4": "thông_tin", "ch": "chung", "đóa": "đó", "đoá": "đó",
          "ck": "chồng", "bao h": "bao giờ", "sgn": "sài gòn", "e": "em",
          "sđt": "số điện thoại", "sh":"shop", "mí": "mới", "dất": "rất", "hoy":  "thôi",
          "spham": "sản phẩm", "form": "dáng", ":v": "", "nsx": "nhà_sản_xuất", "chx": "chưa",
          'o': "không", "okie": "ổn", "kkk": "vui_vẻ", 'ncc': 'nhà_cung_cấp', "trật": "chật", "lai": "lại",
           "vãi": "vải", "hổ": "hỗ", "gião": "giao", "khaki": "kaki", "hic": "híc", "ps": "nói_thêm", '.': " . "}


uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']


def loaddicchar():
    dic = {}
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
    

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\\p{P}*)([p{L}.]*\\p{L}+)(\\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)


def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)


def text_preprocess(document):
    # xóa html code
    document = remove_html(document)

    #thay url = 'link_spam'
    document = re.sub('http[s]://(?:[a-zA-Z]|[0-9]|[$-_@.,&+]|[*\(\)]|(?:%[0-9a-fA-F][0-9a-fA-F]))', 'link_spam', document, flags=re.MULTILINE)
    
    # chuẩn hóa unicode
    document = convert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = chuan_hoa_dau_cau_tieng_viet(document)
   
    # đưa về lower
    document = document.lower()

    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()

     # tách từ
    document = rdrsegmenter.tokenize(document)

    #Xử lí từ viết tắt
    for token in document[0]:
      #token = token.rstrip()
      if len(token) < 21:
        if token in dict_vn.keys():
          a = document[0].index(token)
          document[0][a] = document[0][a].replace(token, dict_vn[token])
          # string.remove(token)
          # string.append(token_edited)
      else:
        a = document[0].index(token)
        document[0][a] = document[0][a].replace(token, "")
        
    return ' '.join(document[0])

def get_max_str(lst):
    return max(lst, key=len)

def process_df(df):
    df1 = df.copy()
    df1['text_prep'] = df1['content'].apply(text_preprocess)
    df1['text_prep2'] = df1['text_prep'].str.replace('\d{8,}', '')
    df1["token"]= df1["text_prep2"].str.split(" ", expand = False)
    df1["count"]= df1["token"].apply(get_max_str)
    df1['len'] = df1['count'].apply(len)
    df1.drop(['text_prep', 'token', 'count', 'len', 'content'], axis=1, inplace = True)
    return df1

dicchar = loaddicchar()

nguyen_am_to_ids = {}
for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)