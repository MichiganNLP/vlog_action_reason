import pandas as pd
import json
# import srt
# import glob
# import shutil
CROSS_TASK = 0
COIN = 0
YOUCOOK = 0
YouTube = 1

# def save_proc_transcripts():
#     list_srt = glob.glob("data/srt/*.srt")
#     for video in list_srt:
#         video = video.split(".en.srt")[0].split("/")[-1]
#         format_transcript(video)

# def format_transcript(video):
#     f = open("data/srt/" + video + ".en.srt", "r")
#     data = f.read()
#     subs = list(srt.parse(data))
#     dict_sentences = {}
#     for s in subs:
#         text = s.content
#         for sentence in text.split("\n"):
#             if not sentence:
#                 continue
#             if sentence.strip() == "[Music]" or sentence.strip() == "[Applause]":
#                 continue
#             if sentence not in dict_sentences.keys():
#                 dict_sentences[sentence] = []
#             dict_sentences[sentence].append(s.start)
#             dict_sentences[sentence].append(s.end)
#     subs = []
#     i = 1
#     for text in dict_sentences.keys():
#         if len(dict_sentences[text]) == 2:
#             end = max(dict_sentences[text])
#         else:
#             end = sorted(dict_sentences[text])[-2]
#         start = min(dict_sentences[text])
#
#         subs.append(srt.Subtitle(index=i, start=start, end=end, content=text))
#         i += 1
#     if len(subs) != 0:
#         srt_string = srt.compose(subs)
#         # srf_name = os.path.splitext(video_name)[0]
#         with open(os.path.join('data/proc/', video + '.srt'), 'w') as f:
#             f.write(srt_string)


def read_list_urls(channel_id):
    print("read_list_urls")
    list_urls = []
    if CROSS_TASK:
        input_file = "/home/user/Action_Recog/OTHER/CrossTask/crosstask_release/videos.csv"
        list_urls = pd.read_csv(input_file, usecols=[2]).values
        list_urls = [a[0] for a in list_urls]
    elif COIN:
        input_file = "/home/user/Action_Recog/OTHER/COIN.json"
        with open(input_file) as f:
            dict_coin = json.load(f)
        list_urls = [dict_coin["database"][k]["video_url"] for k in dict_coin["database"].keys()]
    elif YOUCOOK:
        input_file = "/home/user/Action_Recog/OTHER/youcookii_annotations_trainval.json"
        with open(input_file) as f:
            dict_youcook = json.load(f)
        list_urls = [dict_youcook["database"][k]["video_url"] for k in dict_youcook["database"].keys()]
    elif YouTube:
        input_file = "data/video_links/" + channel_id + ".csv"
        list_urls = pd.read_csv(input_file).values
        list_urls = [a[0] for a in list_urls]
    return list_urls

def main():
    # remove if need to run for other dataset
    # shutil.rmtree('data/proc')
    # shutil.rmtree('data/srt')
    # shutil.rmtree('data/raw')
    # if not os.path.exists('data/proc'):
    #     os.mkdir('data/proc')
    #     os.mkdir('data/srt')
    #     os.mkdir('data/raw')

    channel_id = ""
    list_urls = read_list_urls(channel_id)



if __name__ == '__main__':
    main()