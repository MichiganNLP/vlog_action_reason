import datetime
import json
import re
import requests
import tqdm
import pprint
import time
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from spellchecker import SpellChecker
spell = SpellChecker()
from sentence_transformers import SentenceTransformer

def concept_net():
    dict_concept_net = {}
    # for verb in ["clean", "read", "eat", "drink", "write"]:
    for verb in ["clean", "read", "eat", "drink", "write", "work", "study", "browse", "make", "run", "walk", "wash",
                 "listen", "sing", "relax", "sleep", "meditate"]:
        dict_concept_net[verb] = []
        obj = requests.get(
            'http://api.conceptnet.io/query?start=/c/en/' + verb + '&rel=/r/MotivatedByGoal&limit=100').json()
        print(verb + ": " + str(len(obj['edges'])))
        for edge in obj['edges']:
            motivation = edge['@id'][:-2].split('c/en/')[-1]
            dict_concept_net[verb].append(motivation)

    with open('data/dict_concept_net.json', 'w+') as fp:
        json.dump(dict_concept_net, fp)

    #     if "MotivatedByGoal" in edge['@id']:
    #         print(edge['@id'])
    #     if "HasLastSubevent" in edge['@id']:
    #         print(edge['@id'])
    #     if "HasFirstSubevent" in edge['@id']:
    #         print(edge['@id'])
    #     if "Causes" in edge['@id']:
    #         print(edge['@id'])

def cluster_concept_net(list_verbs):
    with open('data/dict_concept_net.json') as json_file:
        dict_concept_net = json.load(json_file)
    dict_concept_net_clustered = {}
    model = SentenceTransformer('stsb-roberta-base')  # models: https://www.sbert.net/docs/pretrained_models.html#semantic-textual-similarity

    for verb in list_verbs:
        dict_concept_net_clustered[verb] = []
        conceptnet_labels = dict_concept_net[verb]
        sentence_labels = []
        for label in conceptnet_labels:
            misspelled_words = spell.unknown(label.split("_")) # check and correct spelling
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
    with open('data/dict_concept_net_clustered.json', 'w+') as fp:
        json.dump(dict_concept_net_clustered, fp)


def save_sentences_per_verb(list_verbs):
    # with open('data/all_sentence_transcripts.json') as json_file:
    with open('data/all_sentence_transcripts_rachel.json') as json_file:
        data = json.load(json_file)

    dict_sentences_per_verb = {}
    for video in tqdm.tqdm(data.keys()):
        sentences_time = data[video]
        for index, s_t in enumerate(sentences_time):
            sentence = s_t[0]
            time_s = s_t[1]
            time_e = s_t[2]
            time_s = str(datetime.timedelta(seconds=time_s))
            time_e = str(datetime.timedelta(seconds=time_e))
            for verb in list_verbs:
                # for key in ["", "my", "the", "a", "some"]:
                for key in [""]:
                    # composed_verb = " ".join((verb.split()[0] + " " + key + " " + verb.split()[1]).split())
                    composed_verb = verb
                    if verb == "writing":
                        verb = "write"
                    elif verb == " read ":
                        verb = "read"
                    else:
                        if verb[-3:] == "ing":
                            verb = verb[:-3]
                    if composed_verb in sentence:
                        if verb not in dict_sentences_per_verb.keys():
                            dict_sentences_per_verb[verb] = []
                        if index - 1 >= 0:
                            sentence_before = sentences_time[index - 1][0]
                        else:
                            sentence_before = ""
                        if index + 1 < len(sentences_time):
                            sentence_after = sentences_time[index + 1][0]
                        else:
                            sentence_after = ""
                        # sentence_after2 = sentences_time[index + 2][0]
                        sentence = " ".join((sentence_before + " " + sentence + " " + sentence_after).split())
                        # sentence = " ".join((sentence_before + " " + sentence + " " + sentence_after + " " + sentence_after2).split())
                        dict_sentences_per_verb[verb].append({"sentence": sentence, "time_s": time_s, "time_e": time_e, "video": video})

    pp = pprint.PrettyPrinter()
    pp.pprint(dict_sentences_per_verb)

    for verb in list_verbs:
        if verb == "writing":
            verb = "write"
        elif verb == " read ":
            verb = "read"
        else:
            if verb[-3:] == "ing":
                verb = verb[:-3]
        print(verb + " " + str(len(dict_sentences_per_verb[verb])))

    # with open('data/dict_sentences_per_verb.json', 'w+') as fp:
    with open('data/dict_sentences_per_verb_rachel.json', 'w+') as fp:
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

def change_json_for_web_annotations():
    with open('data/dict_sentences_per_verb_rachel.json') as json_file:
        data = json.load(json_file)

    with open('data/dict_concept_net.json') as json_file:
        dict_concept_net = json.load(json_file)

    web = {}
    # web["clean"] = {"sentences": [], "reasons": []}
    for action in tqdm.tqdm(data.keys()):
        web[action] = {"sentences": [], "reasons": []}
        for dict_ in data[action][:15]:
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

    with open('data/annotation_input/dict_web.json', 'w+') as fp:
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

def main():
    # concept_net()
    # cluster_concept_net(list_verbs=["clean", "drink", "read", "write"])

    verbs = [["read this", "clean this", "drink this"], ["making bed", "cleaning sink", "reading book", "drinking tea"],
             ["is the best way to", "so that is why"], ["clean ", "cleaning", "cleaned"],
             [" read ", "reading", "drink", "drinking", "write", "writing"]]
    # save_sentences_per_verb(verbs[-1])
    # regular_expr([" clean", " read"]) # not using this for now

    # change_json_for_web_annotations()
    # compare_annotations()
    # edit_annotations_for_ouput()
    # check_if_key_duplicate()

if __name__ == '__main__':
    main()
