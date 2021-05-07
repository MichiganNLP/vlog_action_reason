import json
import ast
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
def main():
    # counter = compute_stats(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")
    compute_why_choose_answer(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")


if __name__ == '__main__':
    main()