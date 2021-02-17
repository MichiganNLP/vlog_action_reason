import json
from collections import Counter
import spacy
import en_core_web_sm
from tqdm import tqdm

nlp = en_core_web_sm.load()

# def get_subject_verb_obj(sentence):
#     print(sentence)
#     tokens = nlp(sentence)
#     svos = findSVOs(tokens)
#     print(svos)
#     print("-------------------------------")

def get_all_verb_dobj(sentence):
    tokens = nlp(sentence)
    list_actions = []
    for t in tokens:
        if t.pos_ == "VERB":
            if t.lemma_ not in ["have", "ve", "do", "can", "will", "could", "would", "have", "share", "welcome", "go", "am",
                                "be", "was", "were", "let"]:
                action = t.lemma_
                for tok in t.children:
                    if tok.dep_ == "dobj":
                        action += (" " + tok.text)
                list_actions.append(action)

    return list_actions

def get_all_verbs(sentence):
    tokens = nlp(sentence)
    list_verbs = []
    for t in tokens:
        if t.pos_ == "VERB":
            if t.lemma_ not in ["ve", "do" "can", "will", "could", "would", "have", "share", "welcome", "go", "am", "be", "was", "were", "let"]:
                list_verbs.append(t.lemma_)

    return list_verbs

def get_verbs_all_videos():
    with open('../data/all_sentence_transcripts_rachel.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)


    dict_verb_dobj_per_video = {}
    dict_verb_dobj = {}
    # for video in tqdm(list(all_sentence_transcripts_rachel.keys())[:30]):
    for video in tqdm(all_sentence_transcripts_rachel.keys()):
        verbs_per_video = []
        actions_per_video = []
        for [part_sentence, time_s, time_e] in list(all_sentence_transcripts_rachel[video]):
            # list_verbs = get_all_verbs(part_sentence)
            list_actions = get_all_verb_dobj(part_sentence)
            for action in list_actions:
                actions_per_video.append(action)
            dict_verb_dobj_per_video[video] = actions_per_video
    #         for verb in list_verbs:
    #             verbs_per_video.append(verb)
    #     dict_verbs_per_video[video] = verbs_per_video
    #
    # with open('data/analyse_verbs/dict_verbs_per_video.json', 'w+') as fp:
    #     json.dump(dict_verbs_per_video, fp)
    with open('analyse_verbs/dict_verb_dobj_per_video.json', 'w+') as fp:
        json.dump(dict_verb_dobj_per_video, fp)

def analyse_verbs():
    with open('analyse_verbs/dict_verbs_per_video.json') as json_file:
        dict_verbs_per_video = json.load(json_file)

    # verb = "clean"
    for verb in ["clean", "read", "write", "play"]:
        before_verb = []
        after_verb = []
        for video in dict_verbs_per_video.keys():
            list_verbs = dict_verbs_per_video[video]
            indices_verb = [i for i, x in enumerate(list_verbs) if x == verb]
            if indices_verb:
                for ind in indices_verb:
                    for v in list_verbs[ind-3:ind]:
                        before_verb.append(v)
                    for v in list_verbs[ind+1:ind+4]:
                        after_verb.append(v)
                    # print("before " + list_verbs[ind] + ": " + str(list_verbs[ind-3:ind]))
                    # print("after " + list_verbs[ind] + ": " + str(list_verbs[ind+1:ind+4]))

        print(verb)
        # print(Counter(before_verb).most_common()[-10:])
        print(Counter(before_verb).most_common()[:15])
        print(Counter(after_verb).most_common()[:15])

def analyse_actions():
    with open('analyse_verbs/dict_verb_dobj_per_video.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    # verb = "clean"
    # for verb in ["clean", "read", "write", "play"]:
    for verb in ["clean"]:
        before_verb = []
        after_verb = []
        for video in dict_verb_dobj_per_video.keys():
            list_verbs = dict_verb_dobj_per_video[video]
            indices_verb = [i for i, x in enumerate(list_verbs) if x == verb]
            if indices_verb:
                for ind in indices_verb:
                    ind_index_before = 1
                    ind_index_after = 1
                    nb_before = 0
                    nb_after = 0
                    while nb_before < 3:
                        action_before = list_verbs[ind - ind_index_before]
                        if len(action_before.split()) >= 2:
                            before_verb.append(action_before)
                        ind_index_before += 1
                        nb_before += 1
                    while nb_after < 3:
                        action_after = list_verbs[ind + ind_index_after]
                        if len(action_after.split()) >= 2:
                            after_verb.append(action_after)
                        ind_index_after += 1
                        nb_after += 1
                    # for v in list_verbs[ind-3:ind]:
                    #     before_verb.append(v)
                    # for v in list_verbs[ind+1:ind+4]:
                    #     after_verb.append(v)
                    # print("before " + list_verbs[ind] + ": " + str(list_verbs[ind-3:ind]))
                    # print("after " + list_verbs[ind] + ": " + str(list_verbs[ind+1:ind+4]))

        print(verb)
        # print(Counter(before_verb).most_common()[-10:])
        print(Counter(before_verb).most_common())
        print(Counter(after_verb).most_common())

def analyse_actions2():
    with open('analyse_verbs/dict_verb_dobj_per_video.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)


        for video in list(dict_verb_dobj_per_video.keys())[:10]:
            list_verbs = dict_verb_dobj_per_video[video]
            list_verb_obj = []
            for verb in list_verbs:
                if len(verb.split()) >= 2:
                    list_verb_obj.append(verb)
            print(list_verb_obj)
            print("-------------")


def main():
    # get_verbs_all_videos()
    # analyse_verbs()
    # analyse_actions()
    analyse_actions2()

if __name__ == '__main__':
    main()