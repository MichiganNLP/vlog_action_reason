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
    print(counter)
    print("There are " + str(len(counter)) + " verbs")
    print("There are " + str(len(list_videos)) + " videos")



def main():
    compute_stats(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")


if __name__ == '__main__':
    main()