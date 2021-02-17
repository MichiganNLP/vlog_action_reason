import datetime
import glob
import json

import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
nlp = spacy.load('en_core_web_sm')
import srt
from transformers.modeling_bert import BertModel
# from bert import create_bert_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial



def read_annotations():
    input_file = "/home/user/Action_Recog/OTHER/COIN.json"
    with open(input_file) as f:
        dict_coin = json.load(f)
    dict_video_GT_labels = {}
    for k in dict_coin["database"].keys():
        dict_video_GT_labels[k] = [(annotation["label"], annotation["segment"]) for annotation in  dict_coin["database"][k]["annotation"]]
    # print(dict_video_GT_labels["aR615-SqEls"])
    return dict_video_GT_labels


def transcript_split(video):
    f = open("data/proc/" + video + ".srt", "r")
    data_video = f.read()
    subs_video = list(srt.parse(data_video))
    list_transcript = []
    all_content = ""
    for c in subs_video:
        start = datetime.timedelta.total_seconds(c.start)
        end = datetime.timedelta.total_seconds(c.end)
        content = c.content
        all_content += content
        all_content += " "
        list_transcript.append((content, [start, end]))
    return list_transcript, all_content


def extract_verb_phrase(sentence, GT):
    sentence = nlp(sentence)
    chunks = sentence.noun_chunks
    list_verb_phrases = []
    # print(sentence)
    verb_phrase, verb, verb_1, verb_2, aux, aux_1, noun = "", "", "", "", "", "", ""
    for token in sentence:
        # print(token.text, token.tag_, token.head.text, token.dep_)
        # if token.dep_ == "ROOT" and token.tag_ == "VB":
        if token.tag_ == "VB":
            if verb_phrase:
                list_verb_phrases.append(verb_phrase.lstrip())
            verb = token.text
            verb_phrase = verb

        elif token.dep_ == "dobj":
            # print(sentence)
            # print([c for c in chunks])
            # print(token.text, token.tag_, token.head.text, token.dep_)
            dobj = token.text
            verb_1 = token.head.text
            if verb == verb_1:
                for c in chunks:
                    if str(dobj) in str(c):
                        dobj = str(c)
                        break
                verb_phrase += " " + dobj #TODO: add location (eg. pour some water to the soy milk maker )
            # list_verb_phrases.append(verb_phrase)

        elif token.dep_ == "prep":
            verb_2 = token.head.text
            aux = token.text
            if verb_2 == verb:
                verb_phrase += " " + aux
        elif token.dep_ == "pobj":
            aux_1 = token.head.text
            noun = token.text
            if aux == aux_1:
                for c in chunks:
                    if str(noun) in str(c):
                        noun = str(c)
                        break
                verb_phrase += " " + noun
    if not verb_phrase and GT:
        verb_phrase = sentence
    list_verb_phrases.append(verb_phrase.lstrip())
    # if verb_phrase == "":
    #     verb_phrase = sentence
    # print(list_verb_phrases)
    # print("--------------------------------")
    return list_verb_phrases

def transcript_to_verb_phrases(all_content, GT): #TODO: get the time from transcript for each verb phrase
    c = all_content.replace("'s", " is")
    c = c.replace("'re", " are")
    c = c.replace("'ve", " have")
    c = c.replace("'m", " am")
    c = c.replace("n't", " not")
    c = c.replace("just", "")
    c = c.replace("only", "")
    c = c.replace("really", "")
    c = c.replace("gonna", "going")  # be careful with spaces!!!
    c = c.replace("gon na", "going")
    # c = c.replace(" so that ", " so that is why ")
    c = c.replace(" I also ", ". I also ")
    c = c.replace(" so next ", ". so next ")
    c = c.replace(" Hobbit ", " habit ")
    c = " ".join(c.split()) # remove double spaces
    doc = nlp(c)
    sentences = [sent.string.strip() for sent in doc.sents]
    all_verb_phrases = []
    for sentence in sentences:
        list_verb_phrases = extract_verb_phrase(sentence, GT)
        for vb in list_verb_phrases:
            if vb:
                all_verb_phrases.append(vb)
    return all_verb_phrases

# def compute_sentence_similarity(sentence_GT, sentence_transcript):
#     sentence_GT = nlp(sentence_GT)
#     sentence_transcript = nlp(sentence_transcript)
#     return sentence_GT.similarity(sentence_transcript)

def compute_sentence_similarity(sentence_GT, list_transcript):
    # with open('data/bert_GT_coin.json') as f:
    with open('../data/other_datasets/bert_GT_vb_phrase_coin.json') as f:
        bert_GT_coin = json.loads(f.read())

    # with open('data/bert_transcript_coin.json') as f:
    with open('../data/other_datasets/bert_vb_phrase_coin.json') as f:
        bert_transcript_coin = json.loads(f.read())


    sentence_GT = bert_GT_coin[sentence_GT]
    list_similarities = []
    for sentence_transcript in list_transcript:
        sentence_transcript = bert_transcript_coin[sentence_transcript]
        # similarity = cosine_similarity(sentence_GT, sentence_transcript)
        similarity = 1 - spatial.distance.cosine(sentence_GT, sentence_transcript)
        list_similarities.append(similarity)
    return list_similarities


def find_matches():
    dict_video_GT_labels = read_annotations()

    list_srt = glob.glob("data/proc/*.srt")
    for video in list_srt:
        video = video.split(".srt")[0].split("/")[-1]
        # if video != "jXYZEb-56Wc":
        if video != "aR615-SqEls":
            continue
        print("------------------video " + video + "-------------------------------")
        # print(dict_video_GT_labels[video])
        print("---------------")
        list_transcript, all_content = transcript_split(video)
        list_transcript_verb_phrases = transcript_to_verb_phrases(all_content, GT=False)

        list_GT_actions = [a for (a, time) in dict_video_GT_labels[video]]

        GT_verb_phrases = []
        for sentence in list_GT_actions:
            list_verb_phrases = extract_verb_phrase(sentence, GT=True)

            for vb in list_verb_phrases:
                GT_verb_phrases.append(vb)

        # list_transcript = [t for (t, time) in list_transcript]
        create_bert_embeddings(list_transcript_verb_phrases, path_output="../data/other_datasets/bert_vb_phrase_coin.json")
        create_bert_embeddings(GT_verb_phrases, path_output="../data/other_datasets/bert_GT_vb_phrase_coin.json")

        dict_sim = {}
        for index, sentence_GT in enumerate(GT_verb_phrases):
            print(sentence_GT)
            print("------------------------")
            dict_sim[sentence_GT] = []
            list_similarities = compute_sentence_similarity(sentence_GT, list_transcript_verb_phrases)
            for (sentence_transcript, similarity) in zip(list_transcript_verb_phrases, list_similarities):
                dict_sim[sentence_GT].append((sentence_transcript, similarity))

            # break
            # print(dict_sim.items())
            items = list(dict_sim.values())[index]
            sorted_dict_sim = sorted(items, key=lambda x: x[1], reverse=True)
            print(sorted_dict_sim[:10])
            print("---------------")
            # print([(v, sim) for (v, sim) in sorted_dict_sim if "water" in v])




def main():
    find_matches()
    # extract_verb_phrase("pour some water to the soy milk maker")
    # extract_verb_phrase("filtrate with a filter")

if __name__ == '__main__':
    main()