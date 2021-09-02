import os
import json
import tqdm
import re
import urllib.request
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
nlp = spacy.load('en_core_web_sm')  # or whatever model you have installed
import urllib.request
import ssl
from params import api_key

def save_all_video_urls_in_channel(channel_id):
    print("save_all_video_urls_in_channel")

    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'
    first_url = base_search_url + 'key={}&channelId={}&part=snippet,id&order=date&maxResults=20'.format(api_key, channel_id)

    video_links = []
    url = first_url
    ssl._create_default_https_context = ssl._create_unverified_context
    while True:
        inp = urllib.request.urlopen(url)
        resp = json.load(inp)
        for i in resp['items']:
            if i['id']['kind'] == "youtube#video":
                video_links.append(base_video_url + i['id']['videoId'])
        try:
            next_page_token = resp['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        except:
            break
    df = pd.DataFrame(video_links, columns=["urls"])
    df.to_csv('data/video_links/' + channel_id + '.csv', index=False)
    return video_links


def read_list_urls(channel_id):
    print("read_list_urls")
    input_file = "data/video_links/" + channel_id + ".csv"
    list_urls = pd.read_csv(input_file).values
    list_urls = [a[0] for a in list_urls]
    return list_urls

def save_raw_transcripts(list_urls, channel_id):
    print("save_raw_transcripts")
    dict_transcripts = {}
    video_id = ""
    for url in tqdm.tqdm(list_urls):
        if "https://www.youtube.com/watch?v=" in url:
            video_id = url.split("https://www.youtube.com/watch?v=")[1]
        elif "https://www.youtube.com/embed/" in url:
            video_id = url.split("https://www.youtube.com/embed/")[1]
        else:
            print("Error with unrecognized youtube url")
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            print("Video " + url + " does not have transcript")
        else:
            dict_transcripts[video_id] = transcript

    with open('data/transcripts/transcripts_' + channel_id + '.json', 'w+') as fp:
        json.dump(dict_transcripts, fp)


def sentence_split(channel_id):
    print("sentence_split")
    with open('data/transcripts/transcripts_' + channel_id + '.json') as json_file:
        data = json.load(json_file)

    dict_channel_sentences = {}

    remove_keywords = ["[Laughter]", "[Music]", "[Applause]", "[ __ ]", "[LAUGHTER]", "[MUSIC PLAYING]", "[INAUDIBLE]",
                       "[BLEEP]"]
    for video in tqdm.tqdm(list(data.keys())):
        dict_channel_sentences[video] = []
        content = [c["text"] for c in data[video] if c["text"] not in remove_keywords]
        start = [c["start"] for c in data[video] if c["text"] not in remove_keywords]
        end = [c["start"] + c["duration"] for c in data[video] if c["text"] not in remove_keywords]
        big_sentence = ""
        new = []
        for c, s, e in zip(content, start, end):
            c = c.replace("'s", " is")
            c = c.replace("'re", " are")
            c = c.replace("'ve", " have")
            c = c.replace("'ll", " will")
            c = c.replace("'m", " am")
            c = c.replace("n't", " not")
            c = c.replace(" just ", " ")
            c = c.replace(" only ", " ")
            c = c.replace(" really ", " ")
            c = c.replace("go ahead and", "")
            c = c.replace("went ahead and", "")
            c = c.replace("make sure to", "")
            c = c.replace("make sure", "")
            c = c.replace("yeah", "")
            c = c.replace("uh-huh", "")
            c = c.replace("et/pt", "")
            c = c.replace("[ __ ]", "")
            c = c.replace("[INAUDIBLE]", "")
            c = c.replace("[BLEEP]", "")
            c = c.replace(" im ", " ")
            # c = c.replace(" gonna ", " going ")  # be careful with spaces!!!
            # c = c.replace("gon na", "going")
            # c = c.replace("ca not", "cannot")
            # c = c.replace("wo not", "would not")
            c = c.replace("4am", "4 am")
            c = c.replace(" so I ", " so that is why ")
            c = c.replace(" so they ", " so that is why ")
            c = c.replace(" so we ", " so that is why ")
            c = c.replace(" so you ", " so that is why ")
            c = c.replace(" so he ", " so that is why ")
            c = c.replace(" so she ", " so that is why ")
            c = c.replace(" that way ", " so that is why ")
            c = c.replace(" cuz ", " because ")
            c = c.replace(" I also ", ". I also ")
            c = c.replace(" so next ", ". so next ")
            c = c.replace(" Hobbit ", " habit ")
            c = re.sub(' +', ' ', c)
            big_sentence += c
            big_sentence += " "
            if c.strip() and re.sub('\W+',' ', c.strip()).strip():
                new.append([re.sub('\W+',' ', c.strip()).strip(), s, e])
        big_sentence.strip()
        # transform into sentences
        doc = nlp(big_sentence)
        sentences = [sent.string.strip() for sent in doc.sents]
        sentences = [re.sub('\W+',' ', sent) for sent in sentences]
        sentences = [re.sub(' +', ' ', sent) for sent in sentences]
        sentences = [sent.strip() for sent in sentences if sent.strip()]

        # for s in sentences[:15]:
        #     print(s)
        # print("------------")
        # for x in new[:15]:
        #     [c, s, e] = x
        #     print(c, s, e)
        # break
        #
        ok = 0
        index_transcript = 0
        for sentence in sentences:
            time_sentence = []
            original = sentence
            while index_transcript < len(new):

                [c, s, e] = new[index_transcript]
                if ok:
                    c = new_c
                if c in sentence:
                    time_sentence.append(s)
                    time_sentence.append(e)
                    sentence = sentence[sentence.find(c) + len(c) + 1:]
                    if not sentence:
                        ok = 0
                        index_transcript += 1
                        break
                    index_transcript += 1
                    ok = 0
                elif sentence in c:
                    time_sentence.append(s)
                    time_sentence.append(e)
                    new_c = c[c.find(sentence) + len(sentence) + 1:]
                    ok = 1
                    break
                else:
                    print("sentence: ", sentence)
                    print("c: ", c)
                    index_transcript += 1
                    ok = 0
                    break

            if time_sentence:
                dict_channel_sentences[video].append([original, time_sentence[0], time_sentence[-1]])

    with open('data/sentences_transcripts/' + channel_id + '.json', 'w+') as fp:
        json.dump(dict_channel_sentences, fp)


def save_all_sentences():
    all_dict = {}
    path_to_json = 'data/sentences_transcripts/'
    # json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    # json_files = ["UCM3P_G21gOSVdepXrEFojIg.json"]
    # json_files = ["UC-ga3onzHSJFAGsIebtVeBg.json", "UCM3P_G21gOSVdepXrEFojIg.json", "UC-8yLb1K-DEC6dCYlLOJfiQ.json"]
    # json_files = ["UCM3P_G21gOSVdepXrEFojIg.json"] #rachel
    # json_files = ["UC-ga3onzHSJFAGsIebtVeBg.json"] #lavendaire
    # json_files = ["UC-8yLb1K-DEC6dCYlLOJfiQ.json"] #sadie
    # json_files = ["UCVKFs0cesQUEGtn8gQmhbHw.json"] #britany
    # json_files = ["UCq2E1mIwUKMWzCA4liA_XGQ.json"] #pickup
    json_files = ["UCtk95ovBZbBKYfxIg8RQCSw.json"] #jairwoo
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
        for key in json_text.keys():
            all_dict[key] = json_text[key]

    # with open('data/all_sentence_transcripts.json', 'w+') as fp:
    # with open('data/all_sentence_transcripts_rachel.json', 'w+') as fp:
    # with open('data/all_sentence_transcripts_lavendaire.json', 'w+') as fp:
    # with open('data/all_sentence_transcripts_sadie.json', 'w+') as fp:
    # with open('data/all_sentence_transcripts_britany.json', 'w+') as fp:
    # with open('data/all_sentence_transcripts_pickup.json', 'w+') as fp:
    with open('data/all_sentence_transcripts_jairwoo.json', 'w+') as fp:
        json.dump(all_dict, fp)


def main():
    list_channels = ['UCT9y7nOBdqfWuaZJ_x9mPkA', 'UCM3P_G21gOSVdepXrEFojIg', 'UC-8yLb1K-DEC6dCYlLOJfiQ', 'UCDy89wegrl-5Qv0ZkTtlnPg',
                       'UCK2d_KfjVPwh9gqoczQ9sSw', 'UCVKFs0cesQUEGtn8gQmhbHw', 'UCMfXv2enRXepxG92VoxfrEg', 'UCbQj1aJiioDM8g0tmGmtC_w',
                       'UCJCgaQzY5z5sA-xwkwibygw']
    # list_channels = ['UCM3P_G21gOSVdepXrEFojIg'] # rachel
    # list_channels = ['UC-ga3onzHSJFAGsIebtVeBg'] # lavendaire
    # list_channels = ['UC-8yLb1K-DEC6dCYlLOJfiQ'] # sadie?
    # list_channels = ['UCVKFs0cesQUEGtn8gQmhbHw'] # britany?
    # list_channels = ['UCq2E1mIwUKMWzCA4liA_XGQ'] # pickuplimes?
    list_channels = ['UCtk95ovBZbBKYfxIg8RQCSw'] # jairwoo - man?
    for channel_id in list_channels:
        print(channel_id)
        save_all_video_urls_in_channel(channel_id)
        list_urls = read_list_urls(channel_id)
        save_raw_transcripts(list_urls, channel_id)
        sentence_split(channel_id)
    save_all_sentences()



if __name__ == '__main__':
    main()
