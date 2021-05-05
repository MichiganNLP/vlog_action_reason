from collections import Counter

from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging  # if error - change runtime and try again
import json
import tqdm
import numpy as np
import spacy
import ast

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from statistics import mean

from data_prep_annotation import get_verb_hyponyms

nlp = spacy.load('en_core_web_sm')


def read_annotations(file_in, file_out):
    with open(file_in) as json_file:
        dict_AMT_annotations = json.load(json_file)

    dict_GT_text_label = {}
    count_not_ok = 0
    for key in dict_AMT_annotations.keys():
        verb = ast.literal_eval(key)[1]
        transcript = dict_AMT_annotations[key][0]
        reasons = ast.literal_eval(dict_AMT_annotations[key][1])
        list_all_answers = []
        for ans in ast.literal_eval(dict_AMT_annotations[key][2][0]):
            list_all_answers.append(ans)
        for ans in ast.literal_eval(dict_AMT_annotations[key][2][1]):
            list_all_answers.append(ans)
        if len(dict_AMT_annotations[key][2]) == 3:  # can have only 2 workers
            for ans in ast.literal_eval(dict_AMT_annotations[key][2][2]):
                list_all_answers.append(ans)
        # remove duplicates
        # list_all_answers_union = list(set(list_all_answers)) # take union answers
        list_all_answers_maj = [k for k, v in Counter(list_all_answers).items() if v >= 2]  # take majority answers
        if not list_all_answers_maj:  # TODO: Check these cases ..
            print(key + " " + str(list_all_answers) + " not okay ..")
            count_not_ok += 1
        if verb not in dict_GT_text_label.keys():
            dict_GT_text_label[verb] = {"reasons": reasons, "answers": []}
        if list_all_answers_maj:
            dict_GT_text_label[verb]["answers"].append([transcript, list_all_answers_maj])

    print(len(dict_AMT_annotations.keys()))
    print(count_not_ok)
    with open(file_out, 'w+') as fp:
        json.dump(dict_GT_text_label, fp)

    return dict_GT_text_label


def find_reason(description):
    reasons = []
    keys = ['to', 'so', 'why', 'for', 'because']
    for keyword in ["[ARGM-CAU: ", "ARGM-PRP: "]:
        if keyword in description:
            list_splits = description.split(keyword)
            # print(list_splits)
            for spliting in list_splits[1:]:
                if "]" in spliting:
                    reason = spliting.split("]")[0]
                    if reason not in keys:
                        reasons.append(reason)
    return reasons


# def get_SRL(sentence, SRL_predictor, lemmatizer, list_hyponyms, verb):
def get_SRL(sentence, SRL_predictor, lemmatizer, verb):
    p = SRL_predictor.predict(sentence)
    if p['verbs']:
        for i in range(len(p['verbs'])):
            lemmatized_verb = lemmatizer(p['verbs'][i]['verb'], 'VERB')[0]
            # print(verb, p['verbs'][i]['verb'], lemmatized_verb)
            if verb == lemmatized_verb:  # or lemmatized_verb in list_hyponyms # add hyponyms too: eg. clean is wash, wipe, cleanse, mop, - might need
                # print("yes")
                description = p['verbs'][i]['description']
                if 'ARGM-CAU' in description or 'ARGM-PRP' in description or 'PURPOSE' in description:
                    reasons = find_reason(description)
                    return reasons
            # elif lemmatized_verb in list_hyponyms:


# def similarity_CN_transcript(list_GT_text_label, conceptnet_labels, list_hyponyms, verb, is_SRL):
#     from sentence_transformers import SentenceTransformer, util
#     model = SentenceTransformer(
#         'stsb-roberta-base')  # models: https://www.sbert.net/docs/pretrained_models.html#semantic-textual-similarity
#
#     list_transcripts = []
#     for [text, _] in list_GT_text_label:
#         list_transcripts.append(text)
#
#     if is_SRL:
#         lemmatizer = nlp.vocab.morphology.lemmatizer
#
#         list_SRL_reasons = []
#         from allennlp.predictors.predictor import Predictor
#         # import allennlp_models.tagging
#         SRL_predictor = Predictor.from_path(
#             "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
#         for [text, _] in tqdm.tqdm(list_GT_text_label):
#             reasons = get_SRL(text, SRL_predictor, lemmatizer, list_hyponyms, verb)
#             if reasons:
#                 list_SRL_reasons.append(reasons[0])  # TODO - take all possible reasons?
#             else:
#                 list_SRL_reasons.append(text)
#         list_emb_reasons = model.encode(list_SRL_reasons, convert_to_tensor=True)
#         list_emb_transcripts = list_emb_reasons
#         list_transcripts = list_SRL_reasons
#     else:  # if SRL we compare the reasons extracted from transcripts with CN labels, else we compare CN with the transcripts.
#         list_emb_transcripts = model.encode(list_transcripts, convert_to_tensor=True)
#     # Compute embedding for both lists
#     list_emb_reasons = model.encode(conceptnet_labels, convert_to_tensor=True)
#
#     # Compute cosine-similarits
#     cosine_scores = util.pytorch_cos_sim(list_emb_transcripts, list_emb_reasons)
#
#     # Find the pairs with the highest cosine similarity scores
#     pairs = []
#     for i in range(len(list_emb_transcripts)):
#         for j in range(len(list_emb_reasons)):
#             pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
#
#     # Sort scores in decreasing order
#     pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
#
#     for pair in pairs[:10]:
#         i, j = pair['index']
#         print("{} \t\t {} \t\t Score: {:.4f}".format(list_transcripts[i], conceptnet_labels[j], pair['score']))
#
#     print(cosine_scores.shape)
#     print(len(list_transcripts))
#     print(len(conceptnet_labels))
#     # TODO: finish - add threshold
#     # TODO: Note, in the above approach we use a brute-force approach to find the highest scoring pairs,
#     # which has a quadratic complexity. For long lists of sentences, this might be infeasible.
#     # If you want find the highest scoring pairs in a long list of sentences, have a look at Paraphrase Mining.
#     # https://www.sbert.net/docs/usage/semantic_textual_similarity.html


def similarity_CN_transcript(is_SRL, file_in, file_out):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(
        'stsb-roberta-base')  # models: https://www.sbert.net/docs/pretrained_models.html#semantic-textual-similarity
    # if is_SRL:
    #     lemmatizer = nlp.vocab.morphology.lemmatizer
    #     from allennlp.predictors.predictor import Predictor
    #     SRL_predictor = Predictor.from_path(
    #         "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    with open(file_in) as json_file:
        dict_GT_text_label = json.load(json_file)

    if is_SRL:
        threshold = 0.2  # TODO find best threshold
    else:
        threshold = 0.2  # TODO find best threshold
    dict_results = {"gt": {}, "predicted": {}}

    for verb in tqdm.tqdm(list(dict_GT_text_label)):
        candidate_labels = dict_GT_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_GT_text_label = dict_GT_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_GT_text_label]
        for [transcript, annotated_labels] in list_GT_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

        # if SRL we compare the reasons extracted from transcripts with CN labels, else we compare CN with the transcripts.
        if not is_SRL:
            list_emb_transcripts = model.encode(transcripts, convert_to_tensor=True)
        else:
            list_SRL_reasons = []
            for transcript in tqdm.tqdm(transcripts):
                list_casual_markers = [" because ", " since ", " so that is why ", " thus ", " therefore "]
                for marker in list_casual_markers:
                    if marker in transcript:
                        pos_marker = transcript.find(marker)
                        reason = transcript[pos_marker - 100:pos_marker + 100]
                        list_SRL_reasons.append(reason)
                        break
                # reasons = get_SRL(transcript, SRL_predictor, lemmatizer, verb)
                # if reasons:
                #     list_SRL_reasons.append(reasons[0])  # TODO - take all possible reasons?
                # else:
                #     list_SRL_reasons.append(transcript) # if no detected reason, take whole transcript
            list_emb_reasons = model.encode(list_SRL_reasons, convert_to_tensor=True)
            list_emb_transcripts = list_emb_reasons

        # Compute embedding for both lists
        list_emb_reasons = model.encode(candidate_labels, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = util.pytorch_cos_sim(list_emb_transcripts, list_emb_reasons)

        # Find the pairs with the cosine similarity score > threshold
        for i in range(len(list_emb_transcripts)):
            list_predicted_labels = []
            for j in range(len(list_emb_reasons)):
                if cosine_scores[i][j] > threshold:
                    list_predicted_labels.append(candidate_labels[j])
                if str((verb, transcripts[i])) not in dict_results["predicted"].keys():
                    dict_results["predicted"][str((verb, transcripts[i]))] = []
                if not list_predicted_labels:
                    list_predicted_labels = [
                        "I cannot find any reason mentioned verbally or shown visually in the video"]
                dict_results["predicted"][str((verb, transcripts[i]))].append(list_predicted_labels)

    with open(file_out, 'w+') as fp:
        json.dump(dict_results, fp)

    return dict_results


def Roberta_multilabel(list_GT_text_label, conceptnet_labels):
    from simpletransformers.classification import MultiLabelClassificationModel
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    one_hot_list_text_label = label_to_onehot(list_GT_text_label, conceptnet_labels)

    # Preparing train data
    end = len(one_hot_list_text_label) // 10
    train_data = one_hot_list_text_label[:len(one_hot_list_text_label) - end]
    train_df = pd.DataFrame(train_data, columns=["text", "labels"])
    # Preparing eval data
    eval_data = one_hot_list_text_label[-end:]
    eval_df = pd.DataFrame(eval_data)

    model = MultiLabelClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=len(conceptnet_labels),
        args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs": 5},
    )
    # Train the model
    model.train_model(train_df)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)


def NLI(file_in, file_out):
    with open(file_in) as json_file:
        dict_GT_text_label = json.load(json_file)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
    tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

    threshold = 0.5  # better than 0.8
    dict_results = {"gt": {}, "predicted": {}}

    for verb in dict_GT_text_label:
        candidate_labels = dict_GT_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_GT_text_label = dict_GT_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_GT_text_label]
        for [transcript, annotated_labels] in list_GT_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

        for premise in tqdm.tqdm(transcripts):
            list_predicted_labels = []
            for label in candidate_labels:
                hypothesis = f'The reason for {verb} is {label}.'
                # run through model pre-trained on MNLI
                x = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation_strategy='only_first')
                logits = nli_model(x)[0]
                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)

                true_prob = probs[:, 1].item()
                if true_prob > threshold:
                    list_predicted_labels.append(label)
            if str((verb, premise)) not in dict_results["predicted"].keys():
                dict_results["predicted"][str((verb, premise))] = []
            if not list_predicted_labels:
                list_predicted_labels = ["I cannot find any reason mentioned verbally or shown visually in the video"]
            dict_results["predicted"][str((verb, premise))].append(list_predicted_labels)
        print(verb)

    with open(file_out, 'w+') as fp:
        json.dump(dict_results, fp)

    return dict_results


def transform_one_hot(reasons_pred, all_reasons):
    one_hot_label = []
    for label in all_reasons:
        if label in reasons_pred:
            one_hot_label.append(1)
        else:
            one_hot_label.append(0)
    return one_hot_label


def transform_text_to_indices(reasons_pred, all_reasons):
    index_label = []
    for label in reasons_pred:
        index = all_reasons.index(label.strip())
        index_label.append(index)
    return index_label


def compute_metrics(file_in1, file_in2, print_per_verb):
    with open(file_in1) as json_file:
        dict_results = json.load(json_file)

    print("Compute metrics for: " + file_in2)
    if "majority" in file_in1:
        print("############# Majority class: #########")
    elif "NLI" in file_in1:
        print("NLI:")
    elif "cosine" in file_in1:
        print("cosine:")

    with open(file_in2) as json_file:
        dict_web_trial1 = json.load(json_file)

    list_predicted = []
    list_gt = []
    list_reasons = []
    list_verbs = []
    for key in dict_results["gt"]:
        if key not in dict_results["predicted"]:
            print("error in keys in dict_results!!!")
        verb = ast.literal_eval(key)[0]
        list_predicted.append(dict_results["predicted"][key][0])
        list_gt.append(dict_results["gt"][key][0])
        list_reasons.append(dict_web_trial1[verb]["reasons"])
        list_verbs.append(verb)

    list_gt_labels, list_p_labels = [], []
    list_acc_scores, list_prec_scores, list_recall_scores, list_f1_scores = [], [], [], []
    verb_initial = list_verbs[0]
    for reasons_pred, reasons_gt, all_reasons, verb in zip(list_predicted, list_gt, list_reasons, list_verbs):
        if verb != verb_initial:
            y_all = MultiLabelBinarizer().fit_transform(list_gt_labels + list_p_labels)
            y_true = y_all[:len(list_gt_labels)]
            y_pred = y_all[len(list_gt_labels):]
            flat_y_true = [item for sublist in y_true for item in sublist]
            flat_y_pred = [item for sublist in y_pred for item in sublist]

            list_gt_labels, list_p_labels = [], []
            acc = accuracy_score(flat_y_true, flat_y_pred) * 100
            prec = precision_score(flat_y_true, flat_y_pred) * 100
            rec = recall_score(flat_y_true, flat_y_pred) * 100
            f1 = f1_score(flat_y_true, flat_y_pred) * 100
            list_acc_scores.append(acc)
            list_prec_scores.append(prec)
            list_recall_scores.append(rec)
            list_f1_scores.append(f1)
            if print_per_verb:
                print(verb_initial)
                print("accuracy_score: %.2f | precision_score: %.2f | recall_score: %.2f | f1_score: %.2f" % (
                acc, prec, rec, f1))
                print("-----------------------")
            verb_initial = verb
        one_hot_pred = transform_text_to_indices(reasons_pred, all_reasons)
        one_hot_gt = transform_text_to_indices(reasons_gt, all_reasons)
        list_gt_labels.append(tuple(one_hot_gt))
        list_p_labels.append(tuple(one_hot_pred))
        # print("reasons_pred: ", reasons_pred, str(one_hot_pred))
        # print("reasons_gt: ", reasons_gt, str(one_hot_gt))
        # print("all_reasons: ", all_reasons)
        # print("-------------------------------------------")
    y_all = MultiLabelBinarizer().fit_transform(list_gt_labels + list_p_labels)
    y_true = y_all[:len(list_gt_labels)]
    y_pred = y_all[len(list_gt_labels):]
    flat_y_true = [item for sublist in y_true for item in sublist]
    flat_y_pred = [item for sublist in y_pred for item in sublist]
    acc = accuracy_score(flat_y_true, flat_y_pred) * 100
    prec = precision_score(flat_y_true, flat_y_pred) * 100
    rec = recall_score(flat_y_true, flat_y_pred) * 100
    f1 = f1_score(flat_y_true, flat_y_pred) * 100
    list_acc_scores.append(acc)
    list_prec_scores.append(prec)
    list_recall_scores.append(rec)
    list_f1_scores.append(f1)
    if print_per_verb:
        print(verb_initial)
        print(
            "accuracy_score: %.2f | precision_score: %.2f | recall_score: %.2f | f1_score: %.2f" % (acc, prec, rec, f1))
        print("-----------------------")
    acc, prec, rec, f1 = mean(list_acc_scores), mean(list_prec_scores), mean(list_recall_scores), mean(list_f1_scores)
    print("Avg scores:")
    print("accuracy_score: %.2f | precision_score: %.2f | recall_score: %.2f | f1_score: %.2f" % (acc, prec, rec, f1))
    print(" %.2f & %.2f & %.2f & %.2f" % (acc, prec, rec, f1))


# transform labels in one-hot vectors
def label_to_onehot(list_GT_text_label, conceptnet_labels):
    one_hot_list_text_label = []
    for [text, list_labels] in list_GT_text_label:
        one_hot = []
        for label in conceptnet_labels:
            if label in list_labels:
                one_hot.append(1)
            else:
                one_hot.append(0)
        one_hot_list_text_label.append([text, one_hot])
    return one_hot_list_text_label


def majority_class_baseline(file_in, file_out):
    with open(file_in) as json_file:
        dict_GT_text_label = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}

    for verb in dict_GT_text_label:
        list_GT_text_label = dict_GT_text_label[verb]["answers"]
        list_GT_labels = [l[1] for l in dict_GT_text_label[verb]["answers"]]
        all_labels = [item for sublist in list_GT_labels for item in sublist]
        majority_class = Counter(all_labels).most_common()[0][0]  # get only the first most frequent answer

        for [transcript, annotated_labels] in list_GT_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

            if str((verb, transcript)) not in dict_results["predicted"].keys():
                dict_results["predicted"][str((verb, transcript))] = []
            dict_results["predicted"][str((verb, transcript))].append([majority_class])

    with open(file_out, 'w+') as fp:
        json.dump(dict_results, fp)

    return dict_results


def split_train_test(file_in, file_out1, file_out2):
    with open(file_in) as json_file:
        data = json.load(json_file)

    dict_train = {}  # 90%
    dict_test = {}  # 10%
    for verb in data.keys():
        all_reasons = data[verb]["reasons"]
        all_answers = data[verb]["answers"]
        all_reasons_train = all_answers[: int(len(all_answers) * .80)]
        all_reasons_test = all_answers[int(len(all_answers) * .80):]
        if verb not in dict_train.keys():
            dict_train[verb] = {"reasons": all_reasons, "answers": all_reasons_train}
        if verb not in dict_test.keys():
            dict_test[verb] = {"reasons": all_reasons, "answers": all_reasons_test}

    with open(file_out1, 'w+') as fp:
        json.dump(dict_train, fp)
    with open(file_out2, 'w+') as fp:
        json.dump(dict_test, fp)


# def calculate_metrics(list_GT_text_label, list_predicted_output, conceptnet_labels):
#     from sklearn.metrics import label_ranking_average_precision_score, hamming_loss, accuracy_score, jaccard_score
#
#     one_hot_list_text_label_GT = label_to_onehot(list_GT_text_label, conceptnet_labels)
#     one_hot_list_text_label_P = label_to_onehot(list_predicted_output, conceptnet_labels)
#
#     list_gt_labels = []
#     list_p_labels = []
#     for [text, label] in one_hot_list_text_label_GT:
#         list_gt_labels.append(label)
#     for [text, label] in one_hot_list_text_label_P:
#         list_p_labels.append(label)
#
#     y_true = np.array(list_gt_labels)
#     y_pred = np.array(list_p_labels)
#     # label_ranking_average_precision_score(y_true, y_score)
#
#     print("label_ranking_average_precision_score:", label_ranking_average_precision_score(y_true, y_pred))
#     print("accuracy_score:", accuracy_score(y_true, y_pred))
#     print("jaccard_score:", jaccard_score(y_true, y_pred, average='samples'))
#     print("Hamming_loss: (smaller is better)", hamming_loss(y_true, y_pred))
#

def main():
    file_output = "data/baselines/dict_web_trial.json"
    file_input = "data/AMT/output/for_spam_detect/final_output/pipeline_trial.json"

    file_output = "data/baselines/dict_web_trial_objects.json"
    file_input = "data/video_baselines/pipeline_objects.json"

    # file_output = "data/baselines/dict_web_trial_captions.json"
    # file_input = "data/video_baselines/pipeline_captions.json"


    dict_GT_text_label = read_annotations(file_in=file_input, file_out=file_output)
    # verb = "clean"
    ## list_hyponyms = get_verb_hyponyms(verb="clean")

    # method = "majority"
    # method = "cosine"
    method = "NLI"
    file_out_train = "data/baselines/dict_web_trial_train.json"
    file_out_test = "data/baselines/dict_web_trial_test.json"
    split_train_test(file_in=file_output, file_out1=file_out_train, file_out2=file_out_test)

    if method == "majority":
        majority_class_baseline(file_in=file_out_test, file_out="data/AMT/output/dict_majority_results_trial1.json")
        compute_metrics(file_in1="data/AMT/output/dict_majority_results_trial1.json", file_in2=file_out_test,
                        print_per_verb=False)
    elif method == "cosine":
        similarity_CN_transcript(is_SRL=False, file_in=file_out_test,
                                 file_out="data/AMT/output/dict_cosine_results_trial1.json")
        compute_metrics(file_in1="data/AMT/output/dict_cosine_results_trial1.json", file_in2=file_out_test,
                        print_per_verb=False)
    elif method == "NLI":  # TODO run remote
        NLI(file_in=file_out_test, file_out="data/AMT/output/dict_NLI_results_trial1.json")
        compute_metrics(file_in1="data/AMT/output/dict_NLI_results_trial1.json", file_in2=file_out_test,
                        print_per_verb=False)

    ## Roberta_multilabel(list_GT_text_label, conceptnet_labels)


if __name__ == '__main__':
    main()
