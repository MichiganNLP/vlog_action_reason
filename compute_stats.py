import json
import ast
import time
import datetime
from collections import Counter


def compute_stats(file_in):
    with open(file_in) as json_file:
        data = json.load(json_file)

    list_verbs = []
    list_videos = []
    for key in data.keys():
        verb = ast.literal_eval(key)[1]
        video = ast.literal_eval(key)[0]
        list_verbs.append(verb)
        list_videos.append(video)
    counter = Counter(list_verbs).most_common()
    counter2 = Counter(list_verbs)
    print(counter)
    print("There are " + str(len(counter)) + " verbs")
    print("There are " + str(len(list_videos)) + " videos")
    return counter2

def compute_why_choose_answer(file_in):
    with open(file_in) as json_file:
        data = json.load(json_file)

    list_verbs = []
    list_why = []
    list_conf = []
    for key in data.keys():
        verb = ast.literal_eval(key)[1]
        list_whys = data[key]["why"]
        list_confs = data[key]["confidence"]
        # get majority answers
        if len(list_whys) >= 2:
            list_all_answers_maj = [k for k, v in Counter(list_whys).items() if v >= 2]  # take majority answers
            for ans in list_all_answers_maj:
                list_why.append(ans)
        if len(list_confs) >= 2:
            list_all_answers_maj = [k for k, v in Counter(list_confs).items() if v >= 2]  # take majority answers
            for ans in list_all_answers_maj:
                list_conf.append(ans)
    print(len(data.keys()))
    counter = Counter(list_why).most_common()
    print(counter)
    counter = Counter(list_conf).most_common()
    print(counter)

def compute_nb_words_transcript_nb_hours_video(file_in):
    with open(file_in) as json_file:
        data = json.load(json_file)

    total_s = 0
    total_w = 0
    for key in data.keys():
        video_name = ast.literal_eval(key)[0].split("/")[-1].replace('.mp4?raw=true', '')
        name, time_s, time_e = video_name.split("+")
        x = time.strptime(time_s.split('.')[0], '%H:%M:%S')
        time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
        x = time.strptime(time_e.split('.')[0], '%H:%M:%S')
        time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
        # print(time_s, time_e)
        nb_sec = time_e - time_s
        total_s += nb_sec

        transcript = data[key][0]
        # print(transcript)
        total_w += len(transcript.split())
    print("There are " + str(total_s / 360) + " video hours")
    print("There are " + str(total_w) + " transcript wordds")

def compute_avg_nb_responses(file_in):
    with open(file_in) as json_file:
        data = json.load(json_file)
    s = 0
    leng = 0
    nb_not_found = 0
    for verb in data:
        for d in data[verb]["answers"]:
            if "I cannot find any reason mentioned verbally or shown visually in the video" in d[1]:
                nb_not_found += 1
            s += len(d[1])
            leng += 1
    print(s, leng, s/leng, nb_not_found)



def main():
    # counter = compute_stats(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")
    compute_avg_nb_responses(file_in="data/baselines/dict_web_trial.json")
    # compute_why_choose_answer(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")
    # compute_nb_words_transcript_nb_hours_video(file_in="data/AMT/output/for_spam_detect/final_output/pipeline_trial.json")


if __name__ == '__main__':
    main()