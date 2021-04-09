import itertools
import json
import re
import time
from collections import Counter

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
    # print(len(list_verbs))
    with open('data/verbs-all.json') as json_file:
        verbs = json.load(json_file)
    list_verbs = []
    for verb_l in verbs:
        list_verbs.append(verb_l[0])

    list_no_CN_motivation = []
    for verb in tqdm.tqdm(list_verbs[5000:]):  # 10000 verbs
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

    #     if "MotivatedByGoal" in edge['@id']:
    #         print(edge['@id'])
    #     if "HasLastSubevent" in edge['@id']:
    #         print(edge['@id'])
    #     if "HasFirstSubevent" in edge['@id']:
    #         print(edge['@id'])
    #     if "Causes" in edge['@id']:
    #         print(edge['@id'])


def filter_concept_net():
    # verbs with at least 3 causes - 92 verbs
    with open('data/dict_concept_net2.json') as json_file:
        dict_concept_net = json.load(json_file)

    filtered_dict = {}
    for key in dict_concept_net.keys():
        if len(dict_concept_net[key]) >= 2:
            filtered_dict[key] = dict_concept_net[key]
        else:
            print(key, dict_concept_net[key])

    print(len(filtered_dict))
    with open('data/dict_concept_net_filtered.json', 'w+') as fp:
        json.dump(filtered_dict, fp)


def compare_CN_labels_with_transcripts(file_in):
    with open(file_in) as json_file:
        dict_concept_net = json.load(json_file)
    vbs_few_reasons = dict_concept_net.keys()

    with open('data/all_sentence_transcripts.json') as json_file:
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

    with open(file_out, 'w+') as fp:
        json.dump(dict_sentences_per_verb, fp)


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

    threshold_distance = 20
    dict_sentences_per_verb_MARKERS = {}
    list_casual_markers = [" because ", " since ", " so that is why ", " thus ", " therefore "]
    for verb in list_verbs:
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
                    if abs(pos_marker - pos_verb) <= threshold_distance + 10:
                        dict_sentences_per_verb_MARKERS[verb].append(dict)
                        break

    with open(file_out, 'w+') as fp:
        json.dump(dict_sentences_per_verb_MARKERS, fp)

    for verb in list_verbs:
        print("--------- " + verb + " ---------------")
        print(len(dict_sentences_per_verb_MARKERS[verb]))


def get_verb_hyponyms(verb):
    from nltk.corpus import wordnet
    syns = wordnet.synsets(verb, pos='v')
    list_hyponym_names = []
    for syn in syns:
        if syn.lemmas()[0].name() == verb:
            list_hyponyms = syn.hyponyms()
            for hyponym in list_hyponyms:
                list_hyponym_names.append(hyponym.lemmas()[0].name().replace("_", " "))
    # print(list_hyponym_names)
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


def make_AMT_input(file_in1, file_in2, file_out1, file_out2, list_verbs):
    list_videos = []
    list_video_urls = []
    list_actions = []
    list_reasons = []
    list_transcripts = []
    with open(file_in1) as json_file:
        dict_sentences_per_verb_MARKERS = json.load(json_file)

    with open(file_in2) as json_file:
        dict_concept_net = json.load(json_file)

    nb_posts_per_verb = 5
    trial = 1 #0 before
    root_video_name = "https://github.com/OanaIgnat/miniclips/blob/master/"
    for verb in list_verbs:
        reasons = str(
            dict_concept_net[verb] + ['I cannot find any reason mentioned verbally or shown visually in the video']
            + ['other (please write them in the provided box)'])
        # for element in dict_sentences_per_verb_MARKERS[verb][:nb_posts_per_verb]:
        for element in dict_sentences_per_verb_MARKERS[verb][nb_posts_per_verb * trial:nb_posts_per_verb * (trial + 1)]:
            x = time.strptime(element["time_s"].split('.')[0], '%H:%M:%S')
            time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_start = str(datetime.timedelta(seconds=time_s))

            x = time.strptime(element["time_e"].split('.')[0], '%H:%M:%S')
            time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
            time_end = str(datetime.timedelta(seconds=time_e))

            video_name = root_video_name + element["video"] + '+' + time_start + '+' + time_end + '.mp4?raw=true'
            list_video_urls.append(video_name)
            # list_videos.append(element["video"] + '+' + time_start + '+' + time_end + '.mp4')
            verb_ing = getInflection(verb, 'VBG')[0]
            list_actions.append(verb_ing)
            list_reasons.append(reasons)
            list_transcripts.append(
                element["sentence_before"] + " " + element["sentence"] + " " + element["sentence_after"])

    df_AMT = pd.DataFrame({'video_url': list_video_urls, 'action': list_actions, 'reasons': list_reasons})
    df_AMT.to_csv(file_out1, index=False)

    dict_content_label = {}
    for video, action, transcript, reasons in zip(list_video_urls, list_actions, list_transcripts, list_reasons):
        if str((video, action)) not in dict_content_label.keys():
            dict_content_label[str((video, action))] = {"transcripts": [], "reasons": []}
        dict_content_label[str((video, action))]["transcripts"].append(transcript)
        dict_content_label[str((video, action))]["reasons"].append(reasons)

    with open(file_out2, 'w+') as fp:
        json.dump(dict_content_label, fp)
    # df_all_data = pd.DataFrame({'video': list_videos, 'transcript': list_transcripts, 'action': list_actions, 'reasons': list_reasons})
    # df_all_data.to_csv(file_out2, index=False)


def read_AMT_output(file_in1, file_out):
    ann_df = pd.read_csv(file_in1)
    dict_content_label = {}
    dict_verb_label = {}
    for video, action, reasons, labels, conf, why_both, why_mentioned, why_shown in zip(ann_df["Input.video_url"],
                                                                               ann_df["Input.action"],
                                                                               ann_df["Input.reasons"],
                                                                               ann_df["Answer.category.labels"],
                                                                               ann_df["Answer.confidence.high.on"],
                                                                               ann_df["Answer.why.both.on"],
                                                                               ann_df["Answer.why.mentioned.on"],
                                                                               ann_df["Answer.why.shown.on"]):
        if str((video, action)) not in dict_content_label.keys():
            dict_content_label[str((video, action))] = {"labels": [], "confidence": [], "why": []}
        if action not in dict_verb_label.keys():
            dict_verb_label[action] = {"GT": ast.literal_eval(reasons), "labels": []}
        dict_content_label[str((video, action))]["labels"].append(labels)
        dict_verb_label[action]["labels"].append(ast.literal_eval(labels))
        if conf:
            dict_content_label[str((video, action))]["confidence"].append("high")
        else:
            dict_content_label[str((video, action))]["confidence"].append("low")
        if why_both:
            dict_content_label[str((video, action))]["why"].append("both")
        if why_mentioned:
            dict_content_label[str((video, action))]["why"].append("mentioned")
        if why_shown:
            dict_content_label[str((video, action))]["why"].append("shown")

    nb_posts_at_least_1_high = 0
    nb_posts_at_least_1_low = 0
    nb_posts_at_least_1_mention = 0
    nb_posts_at_least_1_shown = 0
    nb_posts_at_least_1_both = 0
    list_labels, list_confidence, list_whys = [], [], []
    for content in dict_content_label.keys():
        labels = dict_content_label[content]["labels"]
        list_labels.append(labels)
        if len(labels) != 3:
            print("error!!, nb of labels per posts is not 3")
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

    with open(file_out, 'w+') as fp:
        json.dump(dict_content_label, fp)

    return dict_verb_label, list_confidence, list_whys


def calculate_fleiss_kappa_agreement(list_labels):
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
    for verb in dict_verb_label.keys():
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
        for i in range(0, len(binary_labels_per_verb), 3):
            chunk = binary_labels_per_verb[i:i + 3]
            # print(chunk)
            [coder1, coder2, coder3] = chunk
            formatted_codes = [[1, i, coder1[i]] for i in range(len(coder1))] + [[2, i, coder2[i]] for i in
                                                                                 range(len(coder2))] + [
                                  [3, i, coder3[i]] for i in range(len(coder3))]

            ratingtask = agreement.AnnotationTask(data=formatted_codes)

            list_kappa.append(ratingtask.multi_kappa())
        print('Avg Fleiss\'s Kappa per verb:', mean(list_kappa))
        list_all_kappa.append(mean(list_kappa))
    print('Avg Fleiss\'s Kappa all verbs:', mean(list_all_kappa))


def agreement_AMT_output(dict_verb_label, list_confidence, list_whys):
    list_high_confidence_whys = []
    new_list_whys = []  # both overlaps with mention and shown
    print("####### Label agreement:")
    agreement_labels_AMT(dict_verb_label)
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


def create_data_pipeline(file_in1, file_in2, file_out):
    with open(file_in1) as json_file:
        annotations = json.load(json_file)
    with open(file_in2) as json_file:
        data = json.load(json_file)

    data_pipeline = {}
    for key in annotations:
        if key not in data:
            print("Error! " + key + "not in data")
        data_pipeline[key] = [data[key]["transcripts"][0], data[key]["reasons"][0], annotations[key]["labels"]]

    with open(file_out, 'w+') as fp:
        json.dump(data_pipeline, fp)


def main():
    ######### CONCEPT NET
    # concept_net()
    # filter_concept_net()
    # list_vbs_to_filter_out = compare_CN_labels_with_transcripts(file_in="data/dict_concept_net_filtered.json")
    # cluster_concept_net(list_vbs_to_filter_out, file_in="data/dict_concept_net_filtered.json", file_out="data/dict_concept_net_clustered.json")
    # stats_concept_net(file_in="data/dict_concept_net_clustered.json")
    # compare_CN_labels_with_transcripts(file_in="data/dict_concept_net_clustered.json")
    # select_actions_for_trial_annotation()

    # get_verb_hyponyms(verb="clean")

    # get_top_actions()
    ########## TRANSCRIPTS
    # with open("data/dict_concept_net_clustered.json") as json_file:
    #     dict_concept_net_clustered = json.load(json_file)
    # verbs = dict_concept_net_clustered.keys()
    # print(len(verbs))
    verbs = ["clean", "read", "write"]
    # save_sentences_per_verb(list(verbs), file_in='data/all_sentence_transcripts_rachel.json',
    #                         file_out='data/dict_sentences_per_verb_rachel.json')

    # TODO send file_in to LIT1000: scp data/dict_sentences_per_verb.json oignat@lit1000.eecs.umich.edu:/tmp/pycharm_project_296/data/

    # filter_sentences_by_reason(file_in='data/dict_sentences_per_verb3.json')
    # filter_sentences_by_casual_markers(file_in='data/dict_sentences_per_verb_rachel.json', file_out='data/dict_sentences_per_verb_MARKERS_rachel.json', list_verbs=verbs)
    # filter_out_sentence(file_in='data/dict_sentences_per_verb.json', list_verbs=["clean", "write"]) # run on lit1000, scp oignat@lit1000.eecs.umich.edu:/tmp/pycharm_project_296/data/dict_sentences_per_verb_reasons.json data/
    # regular_expr([" clean", " read"]) # not using this for now

    ########## ANNOTATIONS
    # change_json_for_web_annotations(list_actions=["clean", "read", "write", "sleep", "eat", "travel", "shop", "sew", "run", "listen"])
    # compare_annotations()
    # edit_annotations_for_ouput()
    # check_if_key_duplicate()

    ## AMT
    # make_AMT_input(file_in1='data/dict_sentences_per_verb_MARKERS_rachel.json',
    #                file_in2='data/dict_concept_net_clustered_manual.json',
    #                file_out1="data/AMT/input/trial2.csv", file_out2="data/AMT/input/trial2_all_data.json",
    #                list_verbs=verbs)

    list_labels, list_confidence, list_whys = read_AMT_output(file_in1="data/AMT/output/trial1_Batch_4385951_batch_results.csv",
                                                          file_out="data/AMT/output/trial1.json")
    agreement_AMT_output(list_labels, list_confidence, list_whys)

    # create_data_pipeline(file_in1="data/AMT/output/trial1.json", file_in2="data/AMT/input/trial1_all_data.json",
    #                      file_out="data/AMT/output/pipeline_trial1.json")


if __name__ == '__main__':
    main()
