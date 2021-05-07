import itertools
import json
import re
import time
from collections import Counter
import glob
import ast
import pandas as pd
import datetime
import numpy as np
import requests
import spacy
import tqdm
from nltk.collections import OrderedDict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from spellchecker import SpellChecker
from nltk import agreement
from statistics import mean

nlp = spacy.load('en_core_web_sm')

spell = SpellChecker()

from pyinflect import getInflection


def concept_net():
    dict_concept_net = {}
    # got verbs with high concreteness
    list_verbs = ["accept", "anticipate", "arrive", "admire", 'bake', 'bathe', 'boil', 'brush', 'call', 'clap', 'clean',
                  'climb', 'collect', 'comb', 'commute',
                  'cook', 'cover', 'curl', 'cut', 'dance', 'drink', 'drive', 'dust', 'eat', 'exercise', 'feed', 'fold',
                  'flush',
                  'freeze', 'fry', "groom", 'harvest', 'heat', 'jog', 'jump', 'kiss', 'knit', 'knock', 'listen', 'make',
                  'massage', 'meditate', 'mop', 'mow', 'organize', 'paint', 'pay', 'pet', 'plan', 'plant', 'plow',
                  'pour', 'practice', 'quilt', 'read', 'relax', 'rest', 'rinse', 'rub', 'run', 'scratch', 'scream',
                  'screw', 'see', 'shave', 'sing', 'sit', 'skate', 'sleep', 'slice', 'smell', 'smoke', 'soak', 'steer',
                  'stitch', 'study', 'sweat', 'swim', 'swing', 'talk', 'taste', 'teach', 'text', 'throw', 'tie', 'toss',
                  'type', 'walk', 'wash', 'water', 'wear', 'weave', 'weep', 'whisk', 'work', 'write', 'yawn']
    # print(Counter(list_verbs))
    # print(sorted(list_verbs))
    print(len(list_verbs))
    with open('data/verbs-all.json') as json_file:
        verbs = json.load(json_file)
    list_verbs = []
    for verb_l in verbs:
        list_verbs.append(verb_l[0])
    print(len(list_verbs))
    list_no_CN_motivation = []
    for verb in tqdm.tqdm(list_verbs):  # 10000 verbs
        # for verb in tqdm.tqdm(list_verbs):
        try:
            print(verb)
            obj = requests.get(
                'http://api.conceptnet.io/query?start=/c/en/' + verb + '&rel=/r/MotivatedByGoal&limit=100').json()
            print(verb + ": " + str(len(obj['edges'])))
            if len(obj['edges']) == 0:
                list_no_CN_motivation.append(verb)
                continue
            dict_concept_net[verb] = []
            for edge in obj['edges']:
                motivation = edge['@id'][:-2].split('c/en/')[-1]
                dict_concept_net[verb].append(motivation)
        except Exception as e:
            print("error .." + str(e))

    with open('data/dict_concept_net3.json', 'w+') as fp:
        json.dump(dict_concept_net, fp)
    # print(len(list_no_CN_motivation))
    # print(len(list_verbs))
    # print(len(list_verbs) - len(list_no_CN_motivation))

    # #     if "MotivatedByGoal" in edge['@id']:
    # #         print(edge['@id'])
    # #     if "HasLastSubevent" in edge['@id']:
    # #         print(edge['@id'])
    # #     if "HasFirstSubevent" in edge['@id']:
    # #         print(edge['@id'])
    # #     if "Causes" in edge['@id']:
    # #         print(edge['@id'])


def filter_concept_net():
    # verbs with at least 3 causes - 92 verbs
    # with open('data/dict_concept_net2.json') as json_file:
    with open('data/dict_concept_net2.json') as json_file:
        dict_concept_net = json.load(json_file)
    print(len(dict_concept_net.keys()))
    filtered_dict = {}
    for key in dict_concept_net.keys():
        # if len(dict_concept_net[key]) >= 2: #102
        if len(dict_concept_net[key]) >= 1:
            filtered_dict[key] = dict_concept_net[key]
        else:
            print(key, dict_concept_net[key])

    print(len(filtered_dict))
    with open('data/dict_concept_net_filtered.json', 'w+') as fp:
        json.dump(filtered_dict, fp)


def compare_CN_labels_with_transcripts(file_in1, file_in2):
    with open(file_in1) as json_file:
        dict_concept_net = json.load(json_file)
    vbs_few_reasons = dict_concept_net.keys()

    with open(file_in2) as json_file:
        data = json.load(json_file)

    dict_vb_count = {}
    for video in tqdm.tqdm(list(data.keys())):
        for [sentence, time_s, time_e] in list(data[video]):
            for vb in vbs_few_reasons:
                if vb not in dict_vb_count.keys():
                    dict_vb_count[vb] = []
                if vb in sentence:
                    dict_vb_count[vb].append(sentence)
    ordered_d = OrderedDict(sorted(dict_vb_count.items(), key=lambda x: len(x[1])))
    list_vbs_to_filter_out = []
    list_all_vbs_count = []
    for vb in ordered_d.keys():
        if len(dict_vb_count[vb]) < 10:
            list_vbs_to_filter_out.append(vb)
        list_all_vbs_count.append(len(dict_vb_count[vb]))
        # print(vb, str(len(dict_vb_count[vb])))
    print("Total nb of verb appearances in vlogs: " + str(sum(list_all_vbs_count)))
    print("Average nb of appearances per verb: " + str(sum(list_all_vbs_count) / len(list_all_vbs_count)))
    print("Max nb of appearances per verb: " + str(max(list_all_vbs_count)))
    print("Min nb of appearances per verb: " + str(min(list_all_vbs_count)))
    print("Filter out: ")
    print(len(list_vbs_to_filter_out))
    print(list_vbs_to_filter_out)
    return list_vbs_to_filter_out


def cluster_concept_net(list_vbs_to_filter_out, file_in, file_out):
    with open(file_in) as json_file:
        dict_concept_net = json.load(json_file)
    list_verbs = dict_concept_net.keys()
    dict_concept_net_clustered = {}
    model = SentenceTransformer(
        'stsb-roberta-base')  # models: https://www.sbert.net/docs/pretrained_models.html#semantic-textual-similarity

    print("There are initially " + str(len(list_verbs)) + " verbs")
    print("We filter out " + str(len(list_vbs_to_filter_out)) + " verbs.")
    for verb in tqdm.tqdm(list_verbs):
        if verb not in list_vbs_to_filter_out:
            dict_concept_net_clustered[verb] = []
            conceptnet_labels = dict_concept_net[verb]
            sentence_labels = []
            for label in conceptnet_labels:
                misspelled_words = spell.unknown(label.split("_"))  # check and correct spelling
                if misspelled_words:
                    for word in misspelled_words:
                        correction = spell.correction(word)
                        label = label.replace(word, correction)
                sentence_labels.append(" ".join(label.split("_")))
            sentence_embeddings = model.encode(sentence_labels)
            # Perform clustering
            # Normalize the embeddings to unit length
            sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

            # clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2) #, affinity='cosine', linkage='average', distance_threshold=0.4)
            clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average',
                                                       distance_threshold=0.6)

            clustering_model.fit(sentence_embeddings)
            cluster_assignment = clustering_model.labels_

            clustered_sentences = {}
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                if cluster_id not in clustered_sentences:
                    clustered_sentences[cluster_id] = []

                clustered_sentences[cluster_id].append(sentence_labels[sentence_id])
            # for i, cluster in clustered_sentences.items():
            #     print("Cluster ", i + 1)
            #     print(cluster)
            #     print("")
            for i, cluster in clustered_sentences.items():
                dict_concept_net_clustered[verb].append(str(i) + ": " + str(cluster).replace("\n", ""))
            dict_concept_net_clustered[verb] = sorted(dict_concept_net_clustered[verb])

            if len(dict_concept_net_clustered[verb]) < 2:
                print(verb)
                del dict_concept_net_clustered[verb]

    print("There are " + str(len(dict_concept_net_clustered.keys())) + " verbs after filtering for verbs that appear"
                                                                       " < 10 times in all transcripts and have less"
                                                                       " than 2 clusters")

    with open(file_out, 'w+') as fp:
        json.dump(dict_concept_net_clustered, fp)


def stats_concept_net(file_in):
    with open(file_in) as json_file:
        dict_concept_net = json.load(json_file)

    print("Number of verbs after filtering: " + str(len(dict_concept_net.keys())))
    list_nbs_values_per_verb = []
    for key in dict_concept_net.keys():
        # if len(dict_concept_net[key]) == 29:
        #     print(key)
        list_nbs_values_per_verb.append(len(dict_concept_net[key]))
    print("Number of reasons: " + str(sum(list_nbs_values_per_verb)))
    print("Average nb of reasons per verb: " + str(sum(list_nbs_values_per_verb) / len(list_nbs_values_per_verb)))
    print("Max nb of reasons per verb: " + str(max(list_nbs_values_per_verb)))
    print("Min nb of reasons per verb: " + str(min(list_nbs_values_per_verb)))


def save_sentences_per_verb(list_verbs, file_in, file_out):
    with open(file_in) as json_file:
        data = json.load(json_file)

    sentence_iter1, sentence_iter2 = itertools.tee(sentence
                                                   for sentences_time in data.values()
                                                   for sentence, _, _ in sentences_time)
    total_sentences = sum(1 for _ in sentence_iter1)

    doc_iter = nlp.pipe(tqdm.tqdm(sentence_iter2, total=total_sentences), n_process=4)

    # lemmatizer = nlp.vocab.morphology.lemmatizer
    dict_sentences_per_verb = {}
    for video, sentences_time in data.items():
        video_docs = list(itertools.islice(doc_iter, len(sentences_time)))

        for verb in list_verbs:
            if verb not in dict_sentences_per_verb:
                dict_sentences_per_verb[verb] = []

            # verb_lemmatized = lemmatizer(verb.strip(), 'VERB')[0]
            # verb_forms = get_word_forms(verb)['v']
            # for verb_form in verb_forms:

            for index, (sentence, time_s, time_e) in enumerate(sentences_time):
                time_s = str(datetime.timedelta(seconds=time_s))
                time_e = str(datetime.timedelta(seconds=time_e))
                time_s_before, time_e_after = time_s, time_e

                # check video duration to be less than 3 minutes and more than 10 seconds
                x = time.strptime(time_s.split('.')[0], '%H:%M:%S')
                time_s1 = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
                x = time.strptime(time_e.split('.')[0], '%H:%M:%S')
                time_e1 = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
                duration = time_e1 - time_s1
                # print(video, time_e, time_s, duration)
                if duration > 180 or duration < 10:
                    continue

                sentence_before, sentence_after = "", ""
                doc = video_docs[index]
                for token in doc:
                    if token.lemma_ == verb and token.pos_ == "VERB":  # CHECK IF VERB
                        # if verb_form in sentence.split():
                        if index >= 2:
                            sentence_before = sentences_time[index - 2][0] + " " + sentences_time[index - 1][0]
                            time_s_before = str(datetime.timedelta(seconds=sentences_time[index - 2][1]))
                        elif index >= 1:
                            sentence_before = sentences_time[index - 1][0]
                            time_s_before = str(datetime.timedelta(seconds=sentences_time[index - 1][1]))

                        if index + 2 < len(sentences_time):
                            sentence_after = sentences_time[index + 2][0] + " " + sentences_time[index + 1][0]
                            time_e_after = str(datetime.timedelta(seconds=sentences_time[index + 2][2]))
                        elif index + 1 < len(sentences_time):
                            sentence_after = sentences_time[index + 1][0]
                            time_e_after = str(datetime.timedelta(seconds=sentences_time[index + 1][2]))
                        # sentence_after2 = sentences_time[index + 2][0]

                        # sentence = " ".join((sentence_before + " " + sentence + " " + sentence_after).split())
                        # dict_ = {"sentence": sentence, "time_s": time_s, "time_e": time_e, "video": video}
                        # dict_ = {"sentence_before": sentence_before, "sentence": sentence, "sentence_after": sentence_after,
                        #                              "time_s": time_s_before, "time_e": time_e_after, "video": video}
                        dict_ = {"sentence_before": sentence_before, "sentence": sentence,
                                 "sentence_after": sentence_after,
                                 "time_s": time_s_before, "time_e": time_e_after, "video": video,
                                 "verb_pos_sentence": token.idx}  # enlarge video context
                        if dict_ not in dict_sentences_per_verb[verb]:
                            dict_sentences_per_verb[verb].append(dict_)
                        break

    for verb in list_verbs:
        # verb_lemmatized = lemmatizer(verb.strip(), 'VERB')[0]
        print(verb + " " + str(len(dict_sentences_per_verb[verb])))

    # remove keys with no elements
    new_dict_sentences_per_verb = {k: v for k, v in dict_sentences_per_verb.items() if
                                   len(dict_sentences_per_verb[k]) > 0}
    # print("old len(verbs)", str(len(dict_sentences_per_verb.keys())))
    # print("new len(verbs)", str(len(new_dict_sentences_per_verb.keys())))
    with open(file_out, 'w+') as fp:
        json.dump(new_dict_sentences_per_verb, fp)


def regular_expr(list_verbs):
    with open('data/all_sentence_transcripts.json') as json_file:
        data = json.load(json_file)

    dict_regular = {}
    for video in tqdm.tqdm(data.keys()):
        sentences_time = data[video]
        for index, s_t in enumerate(sentences_time):
            sentence = s_t[0]
            for verb in list_verbs:
                if re.search(verb + " .* " + "because", sentence):
                    if verb not in dict_regular.keys():
                        dict_regular[verb] = []
                    dict_regular[verb].append(sentence)

    for verb in list_verbs:
        print(verb + " " + str(len(dict_regular[verb])))

    with open('data/dict_regular.json', 'w+') as fp:
        json.dump(dict_regular, fp)


def change_json_for_web_annotations(list_actions):
    # with open('data/dict_sentences_per_verb_rachel.json') as json_file:
    with open('data/dict_sentences_per_verb_reasons.json') as json_file:
        data = json.load(json_file)

    # with open('data/dict_concept_net.json') as json_file:
    with open('data/dict_concept_net_clustered.json') as json_file:
        dict_concept_net = json.load(json_file)

    web = {}
    nb_examples = 10
    # web["clean"] = {"sentences": [], "reasons": []}
    if not list_actions:
        list_actions = data.keys()
    for action in tqdm.tqdm(list_actions):
        web[action] = {"sentences": [], "reasons": []}
        for dict_ in data[action][:nb_examples]:
            x = time.strptime(dict_["time_s"].split('.')[0], '%H:%M:%S')
            time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            x = time.strptime(dict_["time_e"].split('.')[0], '%H:%M:%S')
            time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_start = str(datetime.timedelta(seconds=time_s))
            time_end = str(datetime.timedelta(seconds=time_e))

            miniclip_name = dict_["video"] + '_' + time_start + '_' + time_end + '.mp4'
            new_dict = {"sentence": dict_["sentence"], "miniclip": miniclip_name}

            # action = "clean"
            web[action]["sentences"].append(new_dict)
            web[action]["reasons"] = dict_concept_net[action]

    with open('../video_annotations/data/reason/dict_web2.json', 'w+') as fp:
        json.dump(web, fp)


def compare_annotations():
    with open('data/annotation_output/output_oana.json') as json_file:
        output_me = json.load(json_file)

    with open('data/annotation_output/output_hanwen.json') as json_file:
        output_hanwen = json.load(json_file)

    output_compare = {}
    for key_value_dict in output_me:
        key = key_value_dict["key"]
        reason = key_value_dict["value"]
        if key not in output_compare:
            output_compare[key] = {"oana": [], "hanwen": []}
        output_compare[key]["oana"].append(reason)
    for key_value_dict in output_hanwen:
        key = key_value_dict["key"]
        reason = key_value_dict["value"]
        if key not in output_compare:
            output_compare[key] = {"oana": [], "hanwen": []}
        output_compare[key]["hanwen"].append(reason)

    for key in output_compare.keys():
        print(key)
        print(output_compare[key]["oana"])
        print(output_compare[key]["hanwen"])
        print("------------------------")

    with open('data/annotation_output/dict_web_annotations_for_agreement.json', 'w+') as fp:
        json.dump(output_compare, fp)


def edit_annotations_for_ouput():
    with open('data/annotation_output/output_oana.json') as json_file:
        output_me = json.load(json_file)

    output = {}
    for key_value_dict in output_me:
        key = key_value_dict["key"]
        reason = key_value_dict["value"]
        if key not in output:
            output[key] = {"oana": [], "hanwen": []}
        output[key]["oana"].append(reason)
        # output[key]["hanwen"].append(reason)

    with open('data/annotation_output/dict_web_annotations_oan.json', 'w+') as fp:
        json.dump(output, fp)


def check_if_key_duplicate():
    with open('data/annotation_output/dict_web_annotations_for_agreement.json') as json_file:
        output = json.load(json_file)
    print(len(output.keys()))
    print(len(set(list(output.keys()))))


def get_SRL_naive(sentence):
    for keyword in ["because", " as ", "therefore", " so that ", "so that is why", " is why "]:
        if keyword in sentence:
            return 1
    return 0


def filter_out_sentence(file_in, list_verbs):
    # filter out sentences that don't provide any info about the cause for an action
    from allennlp.predictors.predictor import Predictor
    from baselines import get_SRL
    SRL_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    lemmatizer = nlp.vocab.morphology.lemmatizer

    with open(file_in) as json_file:
        data = json.load(json_file)
    print(len(data.keys()))

    dict_sentences_per_verb_rachel_no_reasons = {}
    dict_sentences_per_verb_rachel_reasons = {}
    nb_reasons = 0
    nb_no_reasons = 0
    # for verb in data.keys():
    for verb in list_verbs:
        dict_sentences_per_verb_rachel_no_reasons[verb] = []
        dict_sentences_per_verb_rachel_reasons[verb] = []
        list_hyponyms = get_verb_hyponyms(verb)
        for dict in tqdm.tqdm(
                list(data[verb])[:1000]):  # TODO: can limit the number of transcripts per verb - eg. clean has a lot
            # for dict in tqdm.tqdm(data[verb]):
            sentence = dict["sentence"]
            reasons = get_SRL(sentence, SRL_predictor, lemmatizer, list_hyponyms, verb)
            # reasons = get_SRL_naive(sentence)
            if reasons:
                dict_sentences_per_verb_rachel_reasons[verb].append(dict)
                nb_reasons += 1
            else:
                dict_sentences_per_verb_rachel_no_reasons[verb].append(dict)
                nb_no_reasons += 1

    with open('data/dict_sentences_per_verb_reasons.json', 'w+') as fp:
        json.dump(dict_sentences_per_verb_rachel_reasons, fp)
    with open('data/dict_sentences_per_verb_no_reasons.json', 'w+') as fp:
        json.dump(dict_sentences_per_verb_rachel_no_reasons, fp)

    print("len no reason: " + str(nb_no_reasons))
    print("len reason: " + str(nb_reasons))  # 5%


def filter_sentences_by_reason(file_in):
    with open(file_in) as json_file:
        data = json.load(json_file)

    dict_sentences_per_verb_REASONS = {}
    list_verbs = ["clean"]
    list_reasons = ["dirt", "mess", "clutter", "productive", "disgusting", "awful", "guest"]
    for verb in list_verbs:
        dict_sentences_per_verb_REASONS[verb] = {}
        for reason in list_reasons:
            dict_sentences_per_verb_REASONS[verb][reason] = []
            for dict in tqdm.tqdm(
                    data[verb]):  # TODO: can limit the number of transcripts per verb - eg. clean has a lot
                sentence_before = dict["sentence_before"]
                sentence = dict["sentence"]
                sentence_after = dict["sentence_after"]
                big_sentence = sentence_before + " " + sentence + " " + sentence_after
                for word in big_sentence.split():
                    if reason in word:
                        dict_sentences_per_verb_REASONS[verb][reason].append(dict)

    with open('data/dict_sentences_per_verb_REASONS.json', 'w+') as fp:
        json.dump(dict_sentences_per_verb_REASONS, fp)

    for reason in dict_sentences_per_verb_REASONS["clean"].keys():
        print(reason, str(len(dict_sentences_per_verb_REASONS["clean"][reason])))


def filter_sentences_by_casual_markers(file_in, file_out, list_verbs):
    with open(file_in) as json_file:
        data = json.load(json_file)

    list_verbs = data.keys()  # change this if need new pre-defined
    threshold_distance = 30
    threshold_distance_diff = 30
    dict_sentences_per_verb_MARKERS = {}
    list_casual_markers = [" because ", " since ", " so that is why ", " thus ", " therefore "]
    for verb in list_verbs:
        if verb in ["paint", "relax", "drive", "sell", "switch", "shop", "travel", "jump"]:
            threshold_distance = 40
            threshold_distance_diff = 80
        dict_sentences_per_verb_MARKERS[verb] = []
        for dict in tqdm.tqdm(data[verb]):
            sentence_before = dict["sentence_before"]
            sentence = dict["sentence"]
            sentence_after = dict["sentence_after"]
            big_sentence = sentence_before + " " + sentence + " " + sentence_after
            pos_verb = dict["verb_pos_sentence"]
            for marker in list_casual_markers:
                if marker in sentence:
                    pos_marker = sentence.find(marker)
                    if abs(pos_marker - pos_verb) <= threshold_distance:
                        dict_sentences_per_verb_MARKERS[verb].append(dict)
                        break
                if marker in big_sentence:
                    pos_marker = big_sentence.find(marker)
                    if abs(pos_marker - pos_verb) <= threshold_distance + threshold_distance_diff:
                        dict_sentences_per_verb_MARKERS[verb].append(dict)
                        break

    # remove keys with no elements
    new_dict_sentences_per_verb = {k: v for k, v in dict_sentences_per_verb_MARKERS.items() if
                                   len(dict_sentences_per_verb_MARKERS[k]) > 0}
    # print("old len(verbs)", str(len(dict_sentences_per_verb_MARKERS.keys())))
    # print("new len(verbs)", str(len(new_dict_sentences_per_verb.keys())))

    with open(file_out, 'w+') as fp:
        json.dump(new_dict_sentences_per_verb, fp)

    # for verb in new_dict_sentences_per_verb.keys():
    #     print("--------- " + verb + " ---------------")
    #     print(len(new_dict_sentences_per_verb[verb]))


def make_dict_for_annotations(file_in1, file_in2, list_verbs):
    # nb_max_reasons_per_verb = 50
    # nb_max_reasons_per_verb = 5  # trial1
    nb_max_reasons_per_verb = 10  # trial1 - check for other reasons
    with open(file_in1) as json_file:
        data = json.load(json_file)
    with open(file_in2) as json_file:
        dict_concept_net_clustered = json.load(json_file)
    dict_for_annotations = {}
    # list_verbs = data.keys()  #TODO - change if need predefined
    # list_verbs = ["clean", "write", "read"]  # trial1
    for verb in list_verbs:
        if verb not in dict_for_annotations:
            dict_for_annotations[verb] = []
        for value in data[verb][:nb_max_reasons_per_verb]:
            dict_for_annotations[verb].append(value)

    print("Final number verbs: " + str(len(dict_for_annotations.keys())))
    nb_videos = 0
    list_verbs_nb_videos = []
    for verb in dict_for_annotations.keys():
        nb_videos += len(dict_for_annotations[verb])
        # print(verb, len(dict_for_annotations[verb]), len(dict_concept_net_clustered[verb]))
        # print(verb, len(dict_for_annotations[verb]))
        list_verbs_nb_videos.append([verb, len(dict_for_annotations[verb])])

    # sort descending by nb verb
    list_verbs_nb_videos = sorted(list_verbs_nb_videos, key=lambda x: x[1], reverse=True)
    for [verb, nb] in list_verbs_nb_videos:
        print(verb, nb)
    print("Final number videos: " + str(nb_videos))
    print("Final number verbs: " + str(len(list_verbs_nb_videos)))

    with open("data/dict_sentences_per_verb_MARKERS_for_annotation_check_others.json", 'w+') as fp:
        # with open("data/dict_sentences_per_verb_MARKERS_for_annotation_all50.json", 'w+') as fp:
        json.dump(dict_for_annotations, fp)


def get_verb_hyponyms(verb):
    from nltk.corpus import wordnet
    syns = wordnet.synsets(verb, pos='v')
    list_hyponym_names = []
    for syn in syns:
        if syn.lemmas()[0].name() == verb:
            list_hyponyms = syn.hyponyms()
            for hyponym in list_hyponyms:
                list_hyponym_names.append(hyponym.lemmas()[0].name().replace("_", " "))
    print(list_hyponym_names)
    return list_hyponym_names


def get_top_actions():
    with open('data/all_sentence_transcripts_rachel.json') as json_file:
        data = json.load(json_file)

    actions = []
    for video in tqdm.tqdm(list(data.keys())[:30]):
        for [sentence, time_s, time_e] in list(data[video]):
            tokens = nlp(sentence)
            for t in tokens:
                if t.pos_ == "VERB":
                    actions.append(t.lemma_)

    c = Counter(actions)
    print(c.most_common())


def select_actions_for_trial_annotation():
    with open('data/dict_sentences_per_verb_reasons.json') as json_file:
        data = json.load(json_file)

    print(len(data["clean"]))
    print(len(data["read"]))
    print(len(data["write"]))
    print(len(data["sleep"]))
    print(len(data["eat"]))
    print(len(data["travel"]))
    print(len(data["shop"]))
    print(len(data["sew"]))
    print(len(data["run"]))
    print(len(data["listen"]))


# def make_AMT_input(file_in1, file_in2, file_out1, file_out2, list_verbs):
#     list_videos = []
#     list_video_urls = []
#     list_actions = []
#     list_reasons = []
#     list_transcripts = []
#     with open(file_in1) as json_file:
#         dict_sentences_per_verb_MARKERS = json.load(json_file)
#
#     with open(file_in2) as json_file:
#         dict_concept_net = json.load(json_file)
#
#     list_verbs = dict_sentences_per_verb_MARKERS.keys()
#     nb_posts_per_verb = 5
#     trial = 1 #0 before
#     root_video_name = "https://github.com/OanaIgnat/miniclips/blob/main/trial1/" #TODO:change .
#     for verb in list_verbs:
#         reasons = str(
#             dict_concept_net[verb] + ['I cannot find any reason mentioned verbally or shown visually in the video']
#             + ['other (please write them in the provided box)'])
#         # for element in dict_sentences_per_verb_MARKERS[verb][:nb_posts_per_verb]:
#         # for element in dict_sentences_per_verb_MARKERS[verb][nb_posts_per_verb * trial:nb_posts_per_verb * (trial + 1)]:
#         for element in dict_sentences_per_verb_MARKERS[verb]:
#             x = time.strptime(element["time_s"].split('.')[0], '%H:%M:%S')
#             time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
#             time_start = str(datetime.timedelta(seconds=time_s))
#
#             x = time.strptime(element["time_e"].split('.')[0], '%H:%M:%S')
#             time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
#             time_end = str(datetime.timedelta(seconds=time_e))
#
#             video_name = root_video_name + element["video"] + '+' + time_start + '+' + time_end + '.mp4?raw=true'
#             list_video_urls.append(video_name)
#             # list_videos.append(element["video"] + '+' + time_start + '+' + time_end + '.mp4')
#             verb_ing = getInflection(verb, 'VBG')[0]
#             list_actions.append(verb_ing)
#             list_reasons.append(reasons)
#             list_transcripts.append(
#                 element["sentence_before"] + " " + element["sentence"] + " " + element["sentence_after"])
#
#     df_AMT = pd.DataFrame({'video_url': list_video_urls, 'action': list_actions, 'reasons': list_reasons})
#     df_AMT.to_csv(file_out1, index=False)
#
#     dict_content_label = {}
#     for video, action, transcript, reasons in zip(list_video_urls, list_actions, list_transcripts, list_reasons):
#         if str((video, action)) not in dict_content_label.keys():
#             dict_content_label[str((video, action))] = {"transcripts": [], "reasons": []}
#         dict_content_label[str((video, action))]["transcripts"].append(transcript)
#         dict_content_label[str((video, action))]["reasons"].append(reasons)
#
#     with open(file_out2, 'w+') as fp:
#         json.dump(dict_content_label, fp)
#     # df_all_data = pd.DataFrame({'video': list_videos, 'transcript': list_transcripts, 'action': list_actions, 'reasons': list_reasons})
#     # df_all_data.to_csv(file_out2, index=False)
#
#
def make_new_AMT_input(file_in1, file_in2, file_out1, file_out2, list_verbs):
    list_videos = []
    list_video_urls = []
    list_actions = []
    list_reasons = []
    list_transcripts = []
    with open(file_in1) as json_file:
        dict_sentences_per_verb_MARKERS = json.load(json_file)

    with open(file_in2) as json_file:
        dict_concept_net = json.load(json_file)

    list_verbs = dict_sentences_per_verb_MARKERS.keys()
    nb_posts_per_verb = 5
    trial = 1  # 0 before
    # root_video_name = "https://github.com/OanaIgnat/miniclips/blob/main/trial1/"  # TODO:change .
    # root_video_name = "https://github.com/OanaIgnat/miniclips/blob/main/no_check/"  # TODO:change .
    root_video_name = "https://github.com/OanaIgnat/miniclips/blob/main/check_others/"  # TODO:change .

    # read already annotated AMT
    df_amt = pd.read_csv("data/AMT/input/for_spam_detect/all2.csv", usecols=["video1", "video2", "video3", "video4", "video5", "action1", "action2", "action3", "action4","action5"])
    list_videos_amt = list(df_amt["video1"]) + list(df_amt["video2"]) + list(df_amt["video3"]) + list(df_amt["video4"]) + list(df_amt["video5"])

    list_miniclips_github = []
    # for miniclip in glob.glob("../miniclips/no_check/*.mp4"):
    for miniclip in glob.glob("../miniclips/check_others/*.mp4"):
        list_miniclips_github.append(miniclip.split("/")[-1])
    for verb in list_verbs:
        reasons = str(
            dict_concept_net[verb] + ['I cannot find any reason mentioned verbally or shown visually in the video'])
            # + ['other (please write them in the provided box)']) #TODO: remove this if not necessary
        # for element in dict_sentences_per_verb_MARKERS[verb][:nb_posts_per_verb]:
        # for element in dict_sentences_per_verb_MARKERS[verb][nb_posts_per_verb * trial:nb_posts_per_verb * (trial + 1)]:
        for element in dict_sentences_per_verb_MARKERS[verb]:
            x = time.strptime(element["time_s"].split('.')[0], '%H:%M:%S')
            time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_start = str(datetime.timedelta(seconds=time_s))

            x = time.strptime(element["time_e"].split('.')[0], '%H:%M:%S')
            time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_end = str(datetime.timedelta(seconds=time_e))

            video_name = root_video_name + element["video"] + '+' + time_start + '+' + time_end + '.mp4?raw=true'
            video = element["video"] + '+' + time_start + '+' + time_end + '.mp4'
            if video not in list_miniclips_github:
                print(video)
                continue
            # if video_name in list_videos_amt:
            #     continue

            list_video_urls.append(video_name)
            # list_videos.append(element["video"] + '+' + time_start + '+' + time_end + '.mp4')
            verb_ing = getInflection(verb, 'VBG')[0]
            list_actions.append(verb_ing)
            list_reasons.append(reasons)
            list_transcripts.append(
                element["sentence_before"] + " " + element["sentence"] + " " + element["sentence_after"])

    df_AMT = pd.DataFrame({'video_url': list_video_urls, 'action': list_actions, 'reasons': list_reasons})
    # print(list_actions)

    # prepare spam checking data
    list_video_spam = ["https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video1.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video2.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video3.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video4.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video5.mp4?raw=true"] * 10000
    list_actions_spam = ["cleaning"]
    list_reasons_spam = ["['remove mask', 'clean arms and chest', 'put on mask', 'cleanse face', 'other (please write them in the provided box)']"]

    # split into chunks
    # nb_chunks = int(len(list_actions) / 5)
    nb_chunks = int(len(list_actions) / 4)
    print(nb_chunks)
    list_chunks_videos, list_chunks_actions, list_chunks_reasons = [], [], []
    for i in range(0, len(list_actions), nb_chunks):
        list_chunks_videos.append(list_video_urls[i:i + nb_chunks])
        list_chunks_actions.append(list_actions[i:i + nb_chunks])
        list_chunks_reasons.append(list_reasons[i:i + nb_chunks])

    # add rest data (if needed)
    if len(list_chunks_actions[0]) != len(list_chunks_actions[-1]):
        for i in range(len(list_chunks_actions[0]) - len(list_chunks_actions[-1])):
            list_chunks_videos[-1].append("nothing")
            list_chunks_actions[-1].append("nothing")
            list_chunks_reasons[-1].append("['nothing']")
    # add spam check
    list_chunks_videos.append(list_video_spam[:nb_chunks])
    list_chunks_actions.append(list_actions_spam * nb_chunks)
    list_chunks_reasons.append(list_reasons_spam * nb_chunks)

    df_AMT = pd.DataFrame(
        {'video1': list_chunks_videos[0], 'video2': list_chunks_videos[1], 'video3': list_chunks_videos[2],
         'video4': list_chunks_videos[3], 'video5': list_chunks_videos[4], 'action1': list_chunks_actions[0],
         'action2': list_chunks_actions[1], 'action3': list_chunks_actions[2], 'action4': list_chunks_actions[3],
         'action5': list_chunks_actions[4], 'reason1': list_chunks_reasons[0], 'reason2': list_chunks_reasons[1],
         'reason3': list_chunks_reasons[2], 'reason4': list_chunks_reasons[3], 'reason5': list_chunks_reasons[4]})
    df_AMT.to_csv(file_out1, index=False)

    dict_content_label = {}
    for video, action, transcript, reasons in zip(list_video_urls, list_actions, list_transcripts, list_reasons):
        if str((video, action)) not in dict_content_label.keys():
            dict_content_label[str((video, action))] = {"transcripts": [], "reasons": []}
        dict_content_label[str((video, action))]["transcripts"].append(transcript)
        dict_content_label[str((video, action))]["reasons"].append(reasons)

    # # add spam check data - no need, also reasons for spam are different than ones in general
    # reasons = "['company was coming', 'do not like dirtiness', 'habit', 'self care', 'declutter', 'remove dirt', 'I cannot find any reason mentioned verbally or shown visually in the video']"
    # action = "cleaning"
    # for i in range(5):
    #     dict_content_label[str(("https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video" + str(i) + ".mp4?raw=true", action))] = {"transcripts": [], "reasons": []}
    #     dict_content_label[str(("https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video" + str(i) + ".mp4?raw=true", action))]["reasons"].append(reasons)
    # dict_content_label[
    #     str(("https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video1.mp4?raw=true", action))][
    #     "transcripts"].append([""])

    with open(file_out2, 'w+') as fp:
        json.dump(dict_content_label, fp)
    # df_all_data = pd.DataFrame({'video': list_videos, 'transcript': list_transcripts, 'action': list_actions, 'reasons': list_reasons})
    # df_all_data.to_csv(file_out2, index=False)


def make_AMT_input_for_other_reasons(file_in1, file_in2, file_out1, file_out2, list_verbs):
    list_video_urls = []
    list_actions = []
    list_reasons = []
    list_transcripts = []
    with open(file_in1) as json_file:
        dict_sentences_per_verb_MARKERS = json.load(json_file)

    with open(file_in2) as json_file:
        dict_concept_net = json.load(json_file)

    list_verbs = dict_sentences_per_verb_MARKERS.keys()
    print(len(list_verbs))
    nb_posts_per_verb = 5
    trial = 1  # 0 before
    root_video_name = "https://github.com/OanaIgnat/miniclips/blob/main/check_others/"  # TODO:change .
    for verb in list_verbs:
        if verb == "jump":
            continue
        reasons = str(
            dict_concept_net[verb] + ['I cannot find any reason mentioned verbally or shown visually in the video'])
            # + ['other (please write them in the provided box)'])
        for element in dict_sentences_per_verb_MARKERS[verb]:
            # for element in dict_sentences_per_verb_MARKERS[verb][nb_posts_per_verb * trial:nb_posts_per_verb * (trial + 1)]:
            x = time.strptime(element["time_s"].split('.')[0], '%H:%M:%S')
            time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_start = str(datetime.timedelta(seconds=time_s))

            x = time.strptime(element["time_e"].split('.')[0], '%H:%M:%S')
            time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_end = str(datetime.timedelta(seconds=time_e))

            video_name = root_video_name + element["video"] + '+' + time_start + '+' + time_end + '.mp4?raw=true'
            list_video_urls.append(video_name)
            verb_ing = getInflection(verb, 'VBG')[0]
            list_actions.append(verb_ing)
            list_reasons.append(reasons)
            list_transcripts.append(
                element["sentence_before"] + " " + element["sentence"] + " " + element["sentence_after"])
    # split into chunks
    # nb_chunks = int(len(list_actions) / 5)
    # print(nb_chunks)
    list_chunks_videos, list_chunks_actions, list_chunks_reasons = [], [], []

    # for i in range(0, len(list_actions), nb_chunks):
    #     list_chunks_videos.append(list_video_urls[i:i + nb_chunks])
    #     list_chunks_actions.append(list_actions[i:i + nb_chunks])
    #     list_chunks_reasons.append(list_reasons[i:i + nb_chunks])

    # prepare spam checking data
    list_video_spam = ["https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video1.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video2.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video3.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video4.mp4?raw=true",
                       "https://github.com/OanaIgnat/miniclips/blob/main/spam_check/spam_video5.mp4?raw=true"] * 10000
    list_actions_spam = ["cleaning"]
    list_reasons_spam = [
        "['remove mask', 'clean arms and chest', 'put on mask', 'cleanse face', 'other (please write them in the provided box)']"]

    # split into chunks
    # nb_chunks = int(len(list_actions) / 5)
    nb_chunks = int(len(list_actions) / 4)
    print(nb_chunks)
    list_chunks_videos, list_chunks_actions, list_chunks_reasons = [], [], []
    for i in range(0, len(list_actions), nb_chunks):
        list_chunks_videos.append(list_video_urls[i:i + nb_chunks])
        list_chunks_actions.append(list_actions[i:i + nb_chunks])
        list_chunks_reasons.append(list_reasons[i:i + nb_chunks])

    # add rest data (if needed)
    if len(list_chunks_actions[0]) != len(list_chunks_actions[-1]):
        for i in range(len(list_chunks_actions[0]) - len(list_chunks_actions[-1])):
            list_chunks_videos[-1].append("nothing")
            list_chunks_actions[-1].append("nothing")
            list_chunks_reasons[-1].append("['nothing']")
    # add spam check
    list_chunks_videos.append(list_video_spam[:nb_chunks])
    list_chunks_actions.append(list_actions_spam * nb_chunks)
    list_chunks_reasons.append(list_reasons_spam * nb_chunks)

    df_AMT = pd.DataFrame(
        {'video1': list_chunks_videos[0], 'video2': list_chunks_videos[1], 'video3': list_chunks_videos[2],
         'video4': list_chunks_videos[3], 'video5': list_chunks_videos[4], 'action1': list_chunks_actions[0],
         'action2': list_chunks_actions[1], 'action3': list_chunks_actions[2], 'action4': list_chunks_actions[3],
         'action5': list_chunks_actions[4], 'reason1': list_chunks_reasons[0], 'reason2': list_chunks_reasons[1],
         'reason3': list_chunks_reasons[2], 'reason4': list_chunks_reasons[3], 'reason5': list_chunks_reasons[4]})
    df_AMT.to_csv(file_out1, index=False)

    # df_AMT = pd.DataFrame(
    #     {'video1': list_chunks_videos[0], 'video2': list_chunks_videos[1], 'video3': list_chunks_videos[2],
    #      'video4': list_chunks_videos[3], 'video5': list_chunks_videos[4], 'action1': list_chunks_actions[0],
    #      'action2': list_chunks_actions[1], 'action3': list_chunks_actions[2], 'action4': list_chunks_actions[3],
    #      'action5': list_chunks_actions[4], 'reason1': list_chunks_reasons[0], 'reason2': list_chunks_reasons[1],
    #      'reason3': list_chunks_reasons[2], 'reason4': list_chunks_reasons[3], 'reason5': list_chunks_reasons[4]})
    # df_AMT.to_csv(file_out1, index=False)

    ### if need to separate
    # index_trial = 1  # separate into multiple files, with 2 HITS per file
    # nb_HITS_per_file = 2
    # for chunk in range(0, len(list_chunks_videos[0]), nb_HITS_per_file):
    #     file_out = "data/AMT/input_others/trial" + str(index_trial) + ".csv"
    #     df_AMT = pd.DataFrame(
    #         {'video1': list_chunks_videos[0][chunk:chunk + nb_HITS_per_file],
    #          'video2': list_chunks_videos[1][chunk:chunk + nb_HITS_per_file],
    #          'video3': list_chunks_videos[2][chunk:chunk + nb_HITS_per_file],
    #          'video4': list_chunks_videos[3][chunk:chunk + nb_HITS_per_file],
    #          'video5': list_chunks_videos[4][chunk:chunk + nb_HITS_per_file],
    #          'action1': list_chunks_actions[0][chunk:chunk + nb_HITS_per_file],
    #          'action2': list_chunks_actions[1][chunk:chunk + nb_HITS_per_file],
    #          'action3': list_chunks_actions[2][chunk:chunk + nb_HITS_per_file],
    #          'action4': list_chunks_actions[3][chunk:chunk + nb_HITS_per_file],
    #          'action5': list_chunks_actions[4][chunk:chunk + nb_HITS_per_file],
    #          'reason1': list_chunks_reasons[0][chunk:chunk + nb_HITS_per_file],
    #          'reason2': list_chunks_reasons[1][chunk:chunk + nb_HITS_per_file],
    #          'reason3': list_chunks_reasons[2][chunk:chunk + nb_HITS_per_file],
    #          'reason4': list_chunks_reasons[3][chunk:chunk + nb_HITS_per_file],
    #          'reason5': list_chunks_reasons[4][chunk:chunk + nb_HITS_per_file]})
    #     df_AMT.to_csv(file_out, index=False)
    #     index_trial += 1


def edit_AMT_output(file_in1, file_out):
    list_workers_no_results = []
    ann_df = pd.read_csv(file_in1)
    answers = ann_df["Answer.taskAnswers"]
    answer_confidence_high_on = [[], [], [], [], []]
    answers_why = [[], [], [], [], []]
    answers_others = [[], [], [], [], []]
    answers_labels = [[], [], [], [], []]
    list_comments = [[], [], [], [], []]
    for i in range(len(answers)):
        ok = True
        data = json.loads(answers[i])[0]
        # print("_______________________")
        # print(data)
        if not data:
            list_workers_no_results.append((ann_df["HITId"][i], ann_df["WorkerId"][i]))
            print("HIT id not completed: " + str(ann_df["HITId"][i]) + " with worker id: " + str(ann_df["WorkerId"][i]))
            ok = False

        for j in range(5):
            if ok:
                candidate_labels = data['reasons-' + str(j)].keys()
                list_labels = []
                for label in candidate_labels:
                    if data['reasons-' + str(j)][label]:
                        list_labels.append(label)
                answers_labels[j].append(str(list_labels))
            else:
                answers_labels[j].append("")
            if ok:
                answer_confidence_high_on[j].append(data['confidence-' + str(j)]['high'])
            else:
                answer_confidence_high_on[j].append("")
            if ok:
                if data["how-" + str(j)]['both']:
                    answers_why[j].append("both")
                elif data["how-" + str(j)]['mentioned']:
                    answers_why[j].append("mentioned")
                elif data["how-" + str(j)]['shown']:
                    answers_why[j].append("shown")
                else:
                    print("Error in reading how-x AMT results")
            else:
                answers_why[j].append("")
            if 'others-' + str(j) in data:
                answers_others[j].append(data['others-' + str(j)])
            else:
                answers_others[j].append('none')

            if 'comments' in data:
                list_comments[j].append(data['comments'])
            else:
                list_comments[j].append('none')

    # print(len(list_comments), list_comments)
    # print(len(answers_why), answers_why)
    # print(len(answer_confidence_high_on), answer_confidence_high_on)
    # print(len(answers_labels), answers_labels)
    # print(len(answers_others), answers_others)
    df_answers = pd.DataFrame(
        {'Answer.category1.labels': answers_labels[0], 'Answer.category2.labels': answers_labels[1],
         'Answer.category3.labels': answers_labels[2], 'Answer.category4.labels': answers_labels[3],
         'Answer.category5.labels': answers_labels[4],
         'Answer.others1': answers_others[0], 'Answer.others2': answers_others[1],
         'Answer.others3': answers_others[2], 'Answer.others4': answers_others[3], 'Answer.others5': answers_others[4],
         'Answer.confidence1.high.on': answer_confidence_high_on[0],
         'Answer.confidence2.high.on': answer_confidence_high_on[1],
         'Answer.confidence3.high.on': answer_confidence_high_on[2],
         'Answer.confidence4.high.on': answer_confidence_high_on[3],
         'Answer.confidence5.high.on': answer_confidence_high_on[4],
         'Answer.why1': answers_why[0], 'Answer.why2': answers_why[1], 'Answer.why3': answers_why[2],
         'Answer.why4': answers_why[3], 'Answer.why5': answers_why[4],
         'Answer.comment1': list_comments[0], 'Answer.comment2': list_comments[1], 'Answer.comment3': list_comments[2],
         'Answer.comment4': list_comments[3], 'Answer.comment5': list_comments[4]
         })

    # df_answers = pd.DataFrame(
    #     {
    #      'Answer.others1': answers_others[0], 'Answer.others2': answers_others[1],
    #      'Answer.others3': answers_others[2], 'Answer.others4': answers_others[3], 'Answer.others5': answers_others[4]
    #      })

    input_video = ann_df[['Input.video1', 'Input.video2', 'Input.video3', 'Input.video4', 'Input.video5']]
    input_action = ann_df[['Input.action1', 'Input.action2', 'Input.action3', 'Input.action4', 'Input.action5']]
    input_reason = ann_df[['Input.reason1', 'Input.reason2', 'Input.reason3', 'Input.reason4', 'Input.reason5']]

    df_AMT = pd.concat([input_video, input_action, input_reason, df_answers], axis=1)
    # df_AMT = pd.concat([input_action, df_answers], axis=1)
    df_AMT.to_csv(file_out, index=False)


def read_AMT_output(file_in1, file_out):
    ann_df = pd.read_csv(file_in1)
    dict_content_label = {}
    dict_verb_label = {}
    for i in range(1, 6):
        for video, action, reasons, labels, conf, why, other_reasons, comments in zip(ann_df["Input.video" + str(i)],
                                                                            ann_df["Input.action" + str(i)],
                                                                            ann_df["Input.reason" + str(i)],
                                                                            ann_df["Answer.category" + str(i) + ".labels"],
                                                                            ann_df["Answer.confidence" + str(i) + ".high.on"],
                                                                            ann_df["Answer.why" + str(i)],
                                                                            ann_df["Answer.others" + str(i)],
                                                                            ann_df["Answer.comment" + str(i)]):

            if "spam" in video:
                print("Removing spam check content: " + video)
                continue
            if str((video, action)) not in dict_content_label.keys():
                dict_content_label[str((video, action))] = {"labels": [], "other_labels": [], "confidence": [], "why": [],
                                                            "comments": []}
            dict_content_label[str((video, action))]["labels"].append(labels)
            if action not in dict_verb_label.keys():
                dict_verb_label[action] = {"GT": ast.literal_eval(reasons), "labels": []}

            dict_content_label[str((video, action))]["other_labels"].append(other_reasons)
            dict_content_label[str((video, action))]["comments"].append(comments)

            if conf:
                dict_content_label[str((video, action))]["confidence"].append("high")
            else:
                dict_content_label[str((video, action))]["confidence"].append("low")
            dict_content_label[str((video, action))]["why"].append(why)

    nb_posts_at_least_1_high = 0
    nb_posts_at_least_1_low = 0
    nb_posts_at_least_1_mention = 0
    nb_posts_at_least_1_shown = 0
    nb_posts_at_least_1_both = 0
    list_labels, list_confidence, list_whys = [], [], []
    new_dict = dict_content_label.copy()
    for content in dict_content_label.keys():
        labels = dict_content_label[content]["labels"]
        action = ast.literal_eval(content)[1]
        if len(labels) < 2:
            print("nb of labels per posts is less than 2 .. removing content " + content)
            del new_dict[content]
        else:
            if len(labels) == 3:
                for label in labels:
                    list_label = ast.literal_eval(label)
                    if action == "thanking":
                        list_label = ['appreciate' if i == 'grateful' else i for i in list_label]
                    elif action == "buying":
                        list_label = ['need' if i == 'replace broken' else i for i in list_label]
                    elif action == "sleeping":
                        list_label = ['you are tired' if i == 'sleepy' else i for i in list_label]
                        list_label = ['need to restore energy' if i == 'get up early next day' else i for i in list_label]
                    elif action == "helping":
                        list_label = ['help in return' if i == 'express altruism' else i for i in list_label]
                        list_label = ['help in return' if i == 'accomplish mutual goal' else i for i in list_label]
                    elif action == "relaxing":
                        list_label = ['self care' if i == 'healthy' else i for i in list_label]
                    dict_verb_label[action]["labels"].append(list_label)
                confidences = dict_content_label[content]["confidence"]
                list_confidence.append(confidences)
                whys = dict_content_label[content]["why"]
                list_whys.append(whys)
                if "high" in confidences:
                    nb_posts_at_least_1_high += 1
                if "low" in confidences:
                    nb_posts_at_least_1_low += 1

                if "mentioned" in whys:
                    nb_posts_at_least_1_mention += 1
                if "shown" in whys:
                    nb_posts_at_least_1_shown += 1
                if "both" in whys:
                    nb_posts_at_least_1_both += 1

    print(nb_posts_at_least_1_high)
    print(nb_posts_at_least_1_low)
    print(nb_posts_at_least_1_mention)
    print(nb_posts_at_least_1_shown)
    print(nb_posts_at_least_1_both)
    print(len(dict_content_label.keys()))
    print(len(new_dict.keys()))

    with open(file_out, 'w+') as fp:
        json.dump(new_dict, fp)

    # return only dictionary values with len(items) == 3 - for agreement compute
    # new_dict_verb_label = dict_verb_label.copy()
    # new_dict_verb_label = dict_content_label.copy()
    # for key in dict_verb_label.keys():
    #     if len(dict_verb_label[key]["labels"]) != 3:
    #         del new_dict_verb_label[key]
    # print(len(new_dict_verb_label.keys()))

    return dict_verb_label, list_confidence, list_whys


def calculate_fleiss_kappa_agreement(list_labels):
    # print(list_labels)
    coder1 = [l[0] for l in list_labels]
    coder2 = [l[1] for l in list_labels]
    coder3 = [l[2] for l in list_labels]
    formatted_codes = [[1, i, coder1[i]] for i in range(len(coder1))] + [[2, i, coder2[i]] for i in
                                                                         range(len(coder2))] + [
                          [3, i, coder3[i]] for i in range(len(coder3))]
    ratingtask = agreement.AnnotationTask(data=formatted_codes)
    print('Fleiss\'s Kappa:', ratingtask.multi_kappa())


def agreement_labels_AMT(dict_verb_label):
    dict_binary = {}
    list_all_kappa = []
    list_all_multi_kappa = []
    list_micro_kappa = []
    list_verb_agreement = []
    for verb in dict_verb_label.keys():
        if verb == "remembering":
            continue
        if dict_verb_label[verb]["labels"]:
            print(verb)
            dict_binary[verb] = []
            gt_labels = dict_verb_label[verb]["GT"]
            all_labels = dict_verb_label[verb]["labels"]
            binary_labels_per_verb = []
            for labels in all_labels:
                binary_labels = []
                for label_GT in gt_labels:
                    if label_GT in labels:
                        binary_labels.append(1)
                    else:
                        binary_labels.append(0)
                binary_labels_per_verb.append(binary_labels)
            dict_binary[verb] = binary_labels_per_verb
            # print(dict_verb_label[verb])
            # print(dict_binary[verb])
            list_kappa = []
            list_multi_kappa = []
            print(len(binary_labels_per_verb))
            for i in range(0, len(binary_labels_per_verb), 3):
                chunk = binary_labels_per_verb[i:i + 3]
                # print(chunk)
                # [coder1, coder2, coder3] = chunk
                [coder1, coder2, coder3] = chunk
                # formatted_codes = [[1, i, coder1[i]] for i in range(len(coder1))] + [[2, i, coder2[i]] for i in
                #                                                                      range(len(coder2))] + [
                #                       [3, i, coder3[i]] for i in range(len(coder3))]
                formatted_codes = [[0, str(i), str(coder1[i])] for i in range(0, len(coder1))] + [[1, str(i), str(coder2[i])]
                                                                                           for i in range(0, len(coder2))] + \
                                  [[2, str(i), str(coder3[i])] for i in range(0, len(coder3))]

                ratingtask = agreement.AnnotationTask(data=formatted_codes)
                list_multi_kappa.append(ratingtask.multi_kappa())
                list_kappa.append(ratingtask.kappa())
                # if verb == "relaxing":
                #     if ratingtask.kappa() < 0.1:
                #         print(all_labels[i:i + 3], ratingtask.kappa())
                list_micro_kappa.append(ratingtask.kappa())
            list_verb_agreement.append([verb, mean(list_kappa)])
            print('Avg Cohen\'s Kappa per verb:', mean(list_kappa))
            print('Avg Fleiss\'s Kappa per verb:', mean(list_multi_kappa))
            list_all_kappa.append(mean(list_kappa))
            list_all_multi_kappa.append(mean(list_multi_kappa))
    # print("-----------------------------------------")
    print('Macro Avg Cohen\'s Kappa all verbs:', mean(list_all_kappa))
    print('Macro Avg Fleiss\'s Kappa all verbs:', mean(list_all_multi_kappa))
    print('Micro Avg Cohen\'s Kappa all verbs:', mean(list_micro_kappa))
    return list_verb_agreement

def agreement_AMT_output(dict_verb_label, list_confidence, list_whys):
    list_high_confidence_whys = []
    new_list_whys = []  # both overlaps with mention and shown
    print("####### Label agreement:")
    list_verb_agreement = agreement_labels_AMT(dict_verb_label)
    for post_whys, post_confidences in zip(list_whys, list_confidence):
        if "mentioned" in post_whys:
            post_whys = ["mentioned" if x == "both" else x for x in post_whys]
        else:
            post_whys = ["shown" if x == "both" else x for x in post_whys]
        new_list_whys.append(post_whys)
        if 'low' in post_confidences:
            # if False in post_confidences:
            continue
        list_high_confidence_whys.append(post_whys)
    print("####### Why (both, mentioned, shown) agreement ")
    calculate_fleiss_kappa_agreement(new_list_whys)
    print("After removing low confidence posts: ")
    calculate_fleiss_kappa_agreement(list_high_confidence_whys)
    print(len(list_whys), len(list_high_confidence_whys))
    return list_verb_agreement


def create_data_pipeline(file_in1, file_in2, file_out1):
    with open(file_in1) as json_file:
        annotations = json.load(json_file)
    with open(file_in2) as json_file:
        data = json.load(json_file)

    data_pipeline = {}
    print(len(annotations.keys()))
    for key in annotations:
        if key not in data:
            print("Error! " + key + "not in data")
        data_pipeline[key] = [data[key]["transcripts"][0], data[key]["reasons"][0], annotations[key]["labels"]]

    with open(file_out1, 'w+') as fp:
        json.dump(data_pipeline, fp)

    # data_web_trial = {}
    # for key in annotations:
    #     (video, verb) = ast.literal_eval(key)
    #     if verb not in data_web_trial.keys():
    #         data_web_trial[verb] = {"reasons": ast.literal_eval(data[key]["reasons"][0]), "answers": []}
    #     answers = []
    #     transcript = data[key]["transcripts"][0]
    #     labels = []
    #     for label in annotations[key]["labels"]:
    #         labels.append(ast.literal_eval(label))
    #     print(transcript)
    #     print(labels)
    #     break
        # answers.append([transcript, labels])

def merge_dictionaries(file_out):
    dict = {}
    # read_files = glob.glob("data/*MARKERS*.json")
    # put rachel first - best videos - matters when we choose first x videos per verb
    read_files = ['data/dict_sentences_per_verb_MARKERS_rachel.json',
                  'data/dict_sentences_per_verb_MARKERS_lavendaire.json',
                  'data/dict_sentences_per_verb_MARKERS_sadie.json',
                  'data/dict_sentences_per_verb_MARKERS_britany.json',
                  'data/dict_sentences_per_verb_MARKERS_jairwoo.json',
                  'data/dict_sentences_per_verb_MARKERS_pickup.json']

    read_files = ['data/dict_sentences_per_verb_all.json',
                  'data/dict_sentences_per_verb_all2.json']

    # read_files = ['data/all_sentence_transcripts_rachel.json',
    #               'data/all_sentence_transcripts_lavendaire.json',
    #               'data/all_sentence_transcripts_sadie.json',
    #               'data/all_sentence_transcripts_britany.json',
    #               'data/all_sentence_transcripts_jairwoo.json',
    #               'data/all_sentence_transcripts_pickup.json']
    for f in read_files:
        with open(f) as json_file:
            data = json.load(json_file)
        for key in data.keys():
            if key not in dict.keys():
                dict[key] = []
            for val in data[key]:
                dict[key].append(val)
    for key in dict.keys():
        print(key, str(len(dict[key])))
    print(len(dict.keys()))
    with open(file_out, 'w+') as fp:
        json.dump(dict, fp)


def make_spam_reject_file(file_in1, file_out, list_spammers):
    init_df = pd.read_csv(file_in1)
    new_df = init_df.copy()
    print("make_spam_reject_file:")
    print(list_spammers)
    for index, row in init_df.iterrows():
        if (row["HITId"], row["WorkerId"]) in list_spammers:
            new_df["Reject"][index] = "spammer"
        else:
            new_df["Approve"][index] = "X"

    new_df.to_csv(file_out, index=False)

def spam_check_AMT_output(file_in1, file_in2, file_out):
    list_spammers = []
    ann_df = pd.read_csv(file_in1)
    init_df = pd.read_csv(file_in2)
    ann_df = ann_df.replace(np.nan, '', regex=True)
    new_ann_df = ann_df.copy()
    hitids = list(init_df["HITId"])
    workerids = list(init_df["WorkerId"])
    answers_to_check = list(ann_df["Answer.category5.labels"])
    videos_to_check = list(ann_df["Input.video5"])
    no_spam = True

    for index, [hitid, workerid, video, answer] in enumerate(zip(hitids, workerids, videos_to_check, answers_to_check)):
        if "spam_video1" in video or "spam_video2" in video or "spam_video3" in video:
            answer_gt = "cleanse face"
        # elif "spam_video3" in video:
        #     answer_gt = "put on mask"
        elif "spam_video4" in video:
            answer_gt = "clean arms and chest"
        elif "spam_video5" in video:
            answer_gt = "remove mask"
        else:
            print("Error in video name ", video)
            answer_gt = ""
        if answer_gt in answer or ("spam_video3" in video and "put on mask" in answer):
        # if answer_gt == ast.literal_eval(answer)[0]:
            no_spam = True
        else:
            new_ann_df.drop(index, inplace=True)
            no_spam = False
            list_spammers.append((hitid, workerid))
        # if not no_spam:
        #     print(hitid, workerid, video, answer, answer_gt, no_spam)
    print("spammers: " + str(len(list_spammers)))
    print("total: " + str(len(answers_to_check)))

    # check if 2 answers in the same hit are from spammers - need to republish the hit
    list_hitids = []
    for hit_id, workerid in list_spammers:
        list_hitids.append(hit_id)
    counter = Counter(list_hitids)
    print(counter.most_common())

    # write results without spammer
    new_ann_df.to_csv(file_out, index=False)
    return list_spammers

def main():
    ######### CONCEPT NET
    # concept_net()
    # filter_concept_net()
    # list_vbs_to_filter_out = compare_CN_labels_with_transcripts(file_in1="data/dict_concept_net_filtered.json", file_in2='data/all_sentence_transcripts_all.json')
    list_vbs_to_filter_out = []
    # cluster_concept_net(list_vbs_to_filter_out, file_in="data/dict_concept_net_filtered.json", file_out="data/dict_concept_net_clustered.json")
    # stats_concept_net(file_in="data/dict_concept_net_clustered.json")
    # compare_CN_labels_with_transcripts(file_in="data/dict_concept_net_clustered.json")
    # select_actions_for_trial_annotation()

    # get_verb_hyponyms(verb="jump")

    # get_top_actions()
    ########## TRANSCRIPTS
    with open("data/dict_concept_net_clustered_manual.json") as json_file:
        dict_concept_net_clustered = json.load(json_file)
    verbs = dict_concept_net_clustered.keys()
    print("initial number verbs:" + str(len(verbs)))
    # # verbs = ["clean", "read", "write"] #trial 1,2,3
    verbs = ["celebrate", "clean", "cook", "dance", "drink", "drive", "eat", "help", "jump", "laugh", "learn", "paint",
             "play",
             "read", "relax", "remember", "run", "shop", "sing", "sleep", "talk", "travel", "walk", "work",
             "write"]  # 25

    # with open("data/dict_concept_net_filtered.json") as json_file:
    #     dict_concept_net_filtered = json.load(json_file)
    # verbs1 = dict_concept_net_filtered.keys()
    #
    # with open("data/dict_sentences_per_verb_all.json") as json_file:
    #     dict_sentences_per_verb_all = json.load(json_file)
    # verbs2 = dict_sentences_per_verb_all.keys()
    #
    # verbs = verbs1 - verbs2
    # print(len(verbs))
    # # name_channel = ["rachel", "lavendaire", "sadie", "britany", "pickup", "jairwoo"]
    # # name_channel = name_channel[5]
    # save_sentences_per_verb(list(verbs), file_in='data/all_sentence_transcripts_all.json',
    #                         file_out='data/dict_sentences_per_verb_all2.json')
    # #
    # # TODO send file_in to LIT1000: scp data/dict_sentences_per_verb.json oignat@lit1000.eecs.umich.edu:/tmp/pycharm_project_296/data/
    #
    # # filter_sentences_by_reason(file_in='data/dict_sentences_per_verb3.json')
    # filter_sentences_by_casual_markers(file_in='data/dict_sentences_per_verb_all3.json', file_out='data/dict_sentences_per_verb_all_MARKERS.json', list_verbs=verbs)
    # merge_dictionaries(file_out='data/dict_sentences_per_verb_all_MARKERS.json') #file_out='data/all_sentence_transcripts_all.json'
    # merge_dictionaries(file_out='data/dict_sentences_per_verb_all3.json') #file_out='data/all_sentence_transcripts_all.json'

    verbs = ["clean", "drink", "eat", "help", "learn", "play", "read", "work", "write", "thank",
             "buy", "sleep", "listen", "cook", "walk", "fall", "remember", "jump", "travel", "shop",
             "sell", "switch", "drive", "relax", "paint"]  # 25
    # make_dict_for_annotations(file_in1='data/dict_sentences_per_verb_all_MARKERS.json', file_in2="data/dict_concept_net_clustered_manual.json", list_verbs=verbs)
    # filter_out_sentence(file_in='data/dict_sentences_per_verb.json', list_verbs=["clean", "write"]) # run on lit1000, scp oignat@lit1000.eecs.umich.edu:/tmp/pycharm_project_296/data/dict_sentences_per_verb_reasons.json data/
    # regular_expr([" clean", " read"]) # not using this for now

    ########## ANNOTATIONS
    # change_json_for_web_annotations(list_actions=["clean", "read", "write", "sleep", "eat", "travel", "shop", "sew", "run", "listen"])
    # compare_annotations()
    # edit_annotations_for_ouput()
    # check_if_key_duplicate()

    ## AMT
    # verbs = ["clean", "read", "write"]  # trial 1,2,3
    file_in = 'data/dict_sentences_per_verb_MARKERS_for_annotation_trial1.json' #trial
    file_in1 = 'data/dict_sentences_per_verb_MARKERS_for_annotation_check_others.json' #only masters
    file_in2 = 'data/dict_sentences_per_verb_MARKERS_for_annotation_all50.json' #all data
    file_in3 = 'data/dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck.json' #for spam, no masters
    #
    # list_miniclips = []
    # for miniclip in glob.glob("../miniclips/no_check/*.mp4"):
    #     list_miniclips.append(miniclip.split("/")[-1])
    # for miniclip in glob.glob("../miniclips2/*.mp4"):
    #     list_miniclips.append(miniclip.split("/")[-1])
    # for miniclip in glob.glob("../miniclips2/no_check/*.mp4"):
    #         list_miniclips.append(miniclip.split("/")[-1])
    # for miniclip in glob.glob("../miniclips3/no_check/*.mp4"):
    #         list_miniclips.append(miniclip.split("/")[-1])
    #     # miniclip_name = miniclip.split("/")[-1]
    #     # name, time_s, time_e = miniclip_name.split("+")
    # list_miniclips = list(set(list_miniclips))
    # print(list_miniclips)
    # print(len(list_miniclips))
    #
    # dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck_remain = {}
    # with open(file_in3) as json_file:
    #     dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck = json.load(json_file)
    # for verb in list(dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck.keys()):
    #     dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck_remain[verb] = []
    #     for dict in dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck[verb]:
    #         x = time.strptime(dict["time_s"].split('.')[0], '%H:%M:%S')
    #         time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    #         x = time.strptime(dict["time_e"].split('.')[0], '%H:%M:%S')
    #         time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    #
    #         miniclip = dict["video"] + "+" + str(datetime.timedelta(seconds=time_s)) + "+" + \
    #                    str(datetime.timedelta(seconds=time_e)) + ".mp4"
    #         if miniclip in list_miniclips:
    #             continue
    #         dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck_remain[verb].append(dict)
    #     print(verb, len(dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck_remain[verb]))
    #
    # with open('data/dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck_remain.json', 'w+') as fp:
    #     json.dump(dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck_remain, fp)

    # with open(file_in1) as json_file:
    #     dict_sentences_per_verb_MARKERS_others = json.load(json_file)
    # with open(file_in2) as json_file:
    #     dict_sentences_per_verb_MARKERS_all50 = json.load(json_file)
    #
    # dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck = {}
    # for verb in dict_sentences_per_verb_MARKERS_all50.keys():
    #     if verb != "jump":
    #         dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck[verb] = []
    #         for dict in dict_sentences_per_verb_MARKERS_all50[verb]:
    #             if dict in dict_sentences_per_verb_MARKERS_others[verb]:
    #                 continue
    #             dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck[verb].append(dict)
    #         print(verb, len(dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck[verb]))
    # with open('data/dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck.json', 'w+') as fp:
    #     json.dump(dict_sentences_per_verb_MARKERS_for_annotation_all50_nocheck, fp)
    # make_new_AMT_input(file_in1=file_in3,
    # make_new_AMT_input(file_in1=file_in1,
    #                    file_in2='data/dict_concept_net_clustered_manual.json',
    #                    file_out1="data/AMT/input/for_spam_detect/all2.csv", file_out2="data/AMT/input/for_spam_detect/all_data2.json",
    #                    list_verbs=verbs)

    # make_AMT_input_for_other_reasons(file_in1='data/dict_sentences_per_verb_MARKERS_for_annotation_check_others.json',
    #                    file_in2='data/dict_concept_net_clustered_manual.json',
    #                    file_out1="data/AMT/input_others/trial_all2.csv", file_out2="data/AMT/input/trial3_all_data2.json",
    #                    list_verbs=verbs)

    ############# Process AMT Output
    # #TODO: remove others if not needed
    # for file_in in glob.glob("data/AMT/output_others/*.csv"):
    #     file_out = "data/AMT/output_others/refactor/" + file_in.split("/")[-1]
    # #### 1. edit AMT output
    # for file_in in glob.glob("data/AMT/output/for_spam_detect/*.csv"):
    #     file_out1 = "data/AMT/output/for_spam_detect/edited/" + file_in.split("/")[-1]
    #     print("file out: " + file_in.split("/")[-1])
    #     edit_AMT_output(file_in1=file_in, file_out=file_out1)
    #
    #     #### 2. spam detection
    #     file_out2 = "data/AMT/output/for_spam_detect/edited_no_spam/" + file_in.split("/")[-1]
    #     list_spammers = spam_check_AMT_output(file_in1=file_out1, file_in2=file_in, file_out=file_out2)
    #     file_out3 = "data/AMT/output/for_spam_detect/reject_approve/" + file_in.split("/")[-1]
    #     make_spam_reject_file(file_in1=file_in, file_out=file_out3, list_spammers=list_spammers)
        ## break

    # ####3. #TODO - manually make all_batches.csv from edited_no_spam
    file_out3 = "data/AMT/output/for_spam_detect/final_output/trial.json"
    dict_verb_label, list_confidence, list_whys = read_AMT_output(
            file_in1="data/AMT/output/for_spam_detect/edited_no_spam/all_batches.csv",
            file_out=file_out3)

    list_verb_agreement = agreement_AMT_output(dict_verb_label, list_confidence, list_whys)
    #
    # create_data_pipeline(file_in1=file_out3, file_in2="data/AMT/input/for_spam_detect/all_data.json",
    #                      file_out1="data/AMT/output/for_spam_detect/final_output/pipeline_trial.json")
    #                      ## file_out2="data/AMT/output/for_spam_detect/final_output/dict_web_trial.json")


if __name__ == '__main__':
    main()
