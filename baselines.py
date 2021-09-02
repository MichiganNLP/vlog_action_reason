import ast
import json
import logging  # if error - change runtime and try again
from collections import Counter

import pandas as pd
import spacy
import torch
import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from statistics import mean

nlp = spacy.load('en_core_web_sm')

torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReasonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_annotations(file_in, file_out):
    with open(file_in) as json_file:
        dict_AMT_annotations = json.load(json_file)

    dict_GT_text_label = {}
    count_not_ok = 0
    for key in dict_AMT_annotations.keys():
        verb = ast.literal_eval(key)[1]
        transcript = dict_AMT_annotations[key][0]
        reasons = ast.literal_eval(dict_AMT_annotations[key][1])
        why = dict_AMT_annotations[key][3]
        list_why_maj = [k for k, v in Counter(why).items() if v >= 2]  # take majority answers
        list_all_answers = []
        for ans in ast.literal_eval(dict_AMT_annotations[key][2][0]):
            list_all_answers.append(ans)
        for ans in ast.literal_eval(dict_AMT_annotations[key][2][1]):
            list_all_answers.append(ans)
        if len(dict_AMT_annotations[key][2]) >= 3:  # can have only 2 workers
            for ans in ast.literal_eval(dict_AMT_annotations[key][2][2]):
                list_all_answers.append(ans)
        if len(dict_AMT_annotations[key][2]) >= 4:  # can have 4 workers
            for ans in ast.literal_eval(dict_AMT_annotations[key][2][3]):
                list_all_answers.append(ans)
        if len(dict_AMT_annotations[key][2]) >= 5:  # can have 5 workers
            for ans in ast.literal_eval(dict_AMT_annotations[key][2][4]):
                list_all_answers.append(ans)

        list_all_answers_maj = [k for k, v in Counter(list_all_answers).items() if v >= 2]  # take majority answers
        if not list_all_answers_maj:
            print(key + " " + str(list_all_answers) + " not okay ..")
            count_not_ok += 1
            list_all_answers_maj = list_all_answers  # TODO remove these or not ..
        if verb not in dict_GT_text_label.keys():
            dict_GT_text_label[verb] = {"reasons": reasons, "answers": []}
        if list_all_answers_maj:
            dict_GT_text_label[verb]["answers"].append([transcript, list_all_answers_maj, list_why_maj])

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


def similarity_CN_transcript(threshold, model, is_SRL, file_in, file_out):
    with open(file_in) as json_file:
        dict_GT_text_label = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}

    for verb in dict_GT_text_label:
        candidate_labels = dict_GT_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_GT_text_label = dict_GT_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_GT_text_label]
        for [transcript, annotated_labels, _] in list_GT_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

        # if SRL we compare the reasons extracted from transcripts with CN labels, else we compare CN with the transcripts.
        if not is_SRL:
            list_emb_transcripts = model.encode(transcripts, convert_to_tensor=True)
        else:
            list_SRL_reasons = []
            for transcript in transcripts:
                list_casual_markers = [" because ", " since ", " so that is why ", " thus ", " therefore "]
                for marker in list_casual_markers:
                    if marker in transcript:
                        pos_marker = transcript.find(marker)
                        reason = transcript[pos_marker - 100:pos_marker + 100]
                        list_SRL_reasons.append(reason)
                        break
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

    # Preparing dev data
    end = len(one_hot_list_text_label) // 10
    dev_data = one_hot_list_text_label[:len(one_hot_list_text_label) - end]
    dev_df = pd.DataFrame(dev_data, columns=["text", "labels"])
    # Preparing eval data
    eval_data = one_hot_list_text_label[-end:]
    eval_df = pd.DataFrame(eval_data)

    model = MultiLabelClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=len(conceptnet_labels),
        args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_dev_epochs": 5},
    )
    # dev the model
    model.dev_model(dev_df)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)


def NLI(threshold, nli_model, tokenizer, file_in, file_out):
    with open(file_in) as json_file:
        dict_GT_text_label = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}

    nli_model = nli_model.to(DEVICE)

    for verb in tqdm.tqdm(dict_GT_text_label.keys()):
        candidate_labels = dict_GT_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_GT_text_label = dict_GT_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_GT_text_label]
        for [transcript, annotated_labels, _] in list_GT_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

        for premise in transcripts:
            # for premise in transcripts:
            list_predicted_labels = []
            for label in candidate_labels:
                hypothesis = f'The reason for {verb} is {label}.'
                # run through model pre-deved on MNLI
                x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                     truncation_strategy='only_first').to(DEVICE)
                logits = nli_model(x)[0]
                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [nli_model.config.label2id["contradiction"], nli_model.config.label2id["entailment"]]]
                probs = entail_contradiction_logits.softmax(dim=1)

                true_prob = probs[:, 1].item()
                if true_prob > threshold:
                    list_predicted_labels.append(label)
            if str((verb, premise)) not in dict_results["predicted"].keys():
                dict_results["predicted"][str((verb, premise))] = []
            if not list_predicted_labels:
                list_predicted_labels = ["I cannot find any reason mentioned verbally or shown visually in the video"]
            dict_results["predicted"][str((verb, premise))].append(list_predicted_labels)
        # print(verb)

    with open(file_out, 'w+') as fp:
        json.dump(dict_results, fp)

    return dict_results


def prepare_data(dict_GT_text_label):
    all_list_labels, all_list_premises, all_list_hypotheses = [], [], []
    for verb in list(dict_GT_text_label.keys()):
        candidate_labels = dict_GT_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_GT_text_label = dict_GT_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_GT_text_label]
        labels = [l[1] for l in list_GT_text_label]
        for i, premise in enumerate(transcripts):
            for label in candidate_labels:
                hypothesis = f'The reason for {verb} is {label}.'
                all_list_hypotheses.append(hypothesis)
                all_list_premises.append(premise)
                if label in labels[i]:
                    all_list_labels.append(1)
                else:
                    all_list_labels.append(0)

    return all_list_labels, all_list_premises, all_list_hypotheses


def NLI_finetune(nli_model, tokenizer, file_in_dev, file_in_test, file_out):
    with open(file_in_dev) as json_file:
        dict_GT_text_label_dev = json.load(json_file)
    all_list_labels_dev, all_list_premises_dev, all_list_hypotheses_dev = prepare_data(dict_GT_text_label_dev)
    # all_list_labels_dev, all_list_premises_dev, all_list_hypotheses_dev = all_list_labels_dev[
    #                                                                             :10], all_list_premises_dev[
    #                                                                                   :10], all_list_hypotheses_dev[
    #                                                                                         :10]

    with open(file_in_test) as json_file:
        dict_GT_text_label_test = json.load(json_file)
    all_list_labels_eval, all_list_premises_eval, all_list_hypotheses_eval = prepare_data(dict_GT_text_label_test)
    # all_list_labels_eval, all_list_premises_eval, all_list_hypotheses_eval = all_list_labels_dev[
    #                                                                          :10], all_list_premises_dev[
    #                                                                                :10], all_list_hypotheses_dev[:10]

    dev_encodings = tokenizer(all_list_premises_dev, all_list_hypotheses_dev, truncation=True, padding=True)
    test_encodings = tokenizer(all_list_premises_eval, all_list_hypotheses_eval, truncation=True, padding=True)

    dev_dataset = ReasonDataset(dev_encodings, all_list_labels_dev)
    test_dataset = ReasonDataset(test_encodings, all_list_labels_eval)

    def compute_metrics(pred):
        labels = pred.label_ids
        # print(labels)
        # print(pred.predictions)
        preds = pred.predictions[:, [nli_model.config.label2id["contradiction"], nli_model.config.label2id["entailment"]]].argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        print(acc, precision, recall, f1)
        return {
            'f1': f1
        }

    from transformers import dever, devingArguments
    deving_args = devingArguments(
        output_dir='finetune_model/NLI/results',  # output directory
        num_dev_epochs=1,  # total # of deving epochs
        per_device_dev_batch_size=2,  # batch size per device during deving
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='finetune_model/NLI/logs',  # directory for storing logs
    )

    dever = dever(
        model=nli_model,  # the instantiated 🤗 Transformers model to be deved
        args=deving_args,
        compute_metrics=compute_metrics,  # deving arguments, defined above
        dev_dataset=dev_dataset,  # deving dataset
        eval_dataset=test_dataset  # evaluation dataset
    )
    print("deving")
    ######### deving
    dever.dev()
    print("eval")
    ########## eval
    dever.evaluate(ignore_keys=["encoder_last_hidden_state"])


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

    ### Santi's test
    # list_verbs = ['clean', 'clean', 'jump']
    # list_gt = [["guests are coming"], ["guests are coming", "dirty"], ["fitness"]]
    # list_reasons = [["dirty", "guests are coming", "tidy"], ["dirty", "guests are coming", "tidy"], ["is fun!", "fitness"]]
    # list_predicted = [["dirty"], ["guests are coming", "tidy"], ["fitness", "is fun!"]]

    list_gt_labels, list_p_labels = [], []
    list_acc_scores, list_prec_scores, list_recall_scores, list_f1_scores = [], [], [], []
    verb_initial = list_verbs[0]
    all_reasons_initial = list_reasons[0]
    list_verb_f1 = []

    for reasons_pred, reasons_gt, all_reasons, verb in zip(list_predicted, list_gt, list_reasons, list_verbs):
        if verb != verb_initial:
            list_all = list_gt_labels + list_p_labels
            y_all = MultiLabelBinarizer(classes = range(len(all_reasons_initial))).fit_transform(list_all)
            y_true = y_all[:len(list_gt_labels)]
            y_pred = y_all[len(list_gt_labels):]
            flat_y_true = [item for sublist in y_true for item in sublist]
            flat_y_pred = [item for sublist in y_pred for item in sublist]

            list_gt_labels, list_p_labels = [], []
            acc = accuracy_score(flat_y_true, flat_y_pred) * 100
            prec = precision_score(y_true, y_pred, average="samples") * 100
            rec = recall_score(y_true, y_pred, average="samples") * 100
            f1 = f1_score(y_true, y_pred, average="samples") * 100
            list_acc_scores.append(acc)
            list_prec_scores.append(prec)
            list_recall_scores.append(rec)
            list_f1_scores.append(f1)
            if print_per_verb:
                print(verb_initial)
                print("accuracy_score: %.2f | precision_score: %.2f | recall_score: %.2f | f1_score: %.2f" % (
                    acc, prec, rec, f1))
                print("-----------------------")
                list_verb_f1.append([verb_initial, round(f1, 2)])

            verb_initial = verb
            all_reasons_initial = all_reasons
        one_hot_pred = transform_text_to_indices(reasons_pred, all_reasons)
        one_hot_gt = transform_text_to_indices(reasons_gt, all_reasons)
        list_gt_labels.append(tuple(one_hot_gt))
        list_p_labels.append(tuple(one_hot_pred))
        # print("reasons_pred: ", reasons_pred, str(one_hot_pred))
        # print("reasons_gt: ", reasons_gt, str(one_hot_gt))
        # print("all_reasons: ", all_reasons)
        # print("-------------------------------------------")
    list_all = list_gt_labels + list_p_labels
    y_all = MultiLabelBinarizer(classes = range(len(all_reasons_initial))).fit_transform(list_all)
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
        list_verb_f1.append([verb_initial, round(f1, 2)])
    acc, prec, rec, f1 = mean(list_acc_scores), mean(list_prec_scores), mean(list_recall_scores), mean(list_f1_scores)
    print("Avg scores:")
    print("accuracy_score: %.2f | precision_score: %.2f | recall_score: %.2f | f1_score: %.2f" % (acc, prec, rec, f1))
    print(" %.2f & %.2f & %.2f & %.2f" % (acc, prec, rec, f1))

    list_verb_f1.sort(key=lambda x: x[1])
    print(list_verb_f1)



# transform labels in one-hot vectors
def label_to_onehot(list_GT_text_label, conceptnet_labels):
    one_hot_list_text_label = []
    for [text, list_labels, _] in list_GT_text_label:
        one_hot = []
        for label in conceptnet_labels:
            if label in list_labels:
                one_hot.append(1)
            else:
                one_hot.append(0)
        one_hot_list_text_label.append([text, one_hot])
    return one_hot_list_text_label


def majority_class_baseline(file_in1, file_in2, file_out):
    with open(file_in1) as json_file:
        dict_GT_text_label_dev = json.load(json_file)

    with open(file_in2) as json_file:
        dict_GT_text_label_test = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}

    for verb in dict_GT_text_label_dev:
        list_GT_labels_dev = [l[1] for l in dict_GT_text_label_dev[verb]["answers"]]
        all_labels = [item for sublist in list_GT_labels_dev for item in sublist]
        counter = Counter(all_labels).most_common()
        max_nb_times = counter[0][1]
        majority_class_list = []
        for (reason, nb_times) in counter:
            if nb_times == max_nb_times:
                majority_class_list.append(reason)
        # print(verb, majority_class_list)

        list_GT_text_label_test = dict_GT_text_label_test[verb]["answers"]
        # print(list_GT_text_label_test)
        for [transcript, annotated_labels, _] in list_GT_text_label_test:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

            if str((verb, transcript)) not in dict_results["predicted"].keys():
                dict_results["predicted"][str((verb, transcript))] = []
            dict_results["predicted"][str((verb, transcript))].append(majority_class_list)

    with open(file_out, 'w+') as fp:
        json.dump(dict_results, fp)

    return dict_results


def split_dev_test_by_modality(file_in, file_out1, file_out2):
    with open(file_in) as json_file:
        data = json.load(json_file)

    dict_text = {}
    dict_vision = {}
    nb_reasons_text = 0
    nb_reasons_vis = 0
    nb_videos_text = 0
    nb_videos_vis = 0
    for verb in data.keys():
        all_reasons = data[verb]["reasons"]
        all_answers = data[verb]["answers"]
        all_answers_text = []
        all_answers_vision = []
        for ans in all_answers:
            if ans[2]:
                if ans[2][0] in ["both", "shown"]:
                    all_answers_vision.append(ans)
                else:
                    all_answers_text.append(ans)
        if all_answers_text:
            if verb not in dict_text.keys():
                dict_text[verb] = {"reasons": all_reasons, "answers": all_answers_text}
            nb_reasons_text += len(all_reasons)
            nb_videos_text += len(all_answers_text)

        if all_answers_vision:
            if verb not in dict_vision.keys():
                dict_vision[verb] = {"reasons": all_reasons, "answers": all_answers_vision}
            nb_reasons_vis += len(all_reasons)
            nb_videos_vis += len(all_answers_vision)

    with open(file_out1, 'w+') as fp:
        json.dump(dict_text, fp)
    with open(file_out2, 'w+') as fp:
        json.dump(dict_vision, fp)

    print("Stats text & vision")
    print("Nb actions: ")
    print(len(dict_text), len(dict_vision))
    print("Nb videos:")
    print(nb_videos_text, nb_videos_vis)
    print("Nb reasons:")
    print(nb_reasons_text, nb_reasons_vis)

def split_dev_test(file_in, file_out1, file_out2):
    with open(file_in) as json_file:
        data = json.load(json_file)

    dict_dev = {}  # 80%
    dict_test = {}  # 20%
    nb_reasons_dev = 0
    nb_reasons_test = 0
    nb_videos_dev = 0
    nb_videos_test = 0
    for verb in data.keys():
        all_reasons = data[verb]["reasons"]
        all_answers = data[verb]["answers"]
        all_answers_dev = all_answers[: int(len(all_answers) * .80)]
        all_answers_test = all_answers[int(len(all_answers) * .80):]
        if verb not in dict_dev.keys():
            dict_dev[verb] = {"reasons": all_reasons, "answers": all_answers_dev}
        if verb not in dict_test.keys():
            dict_test[verb] = {"reasons": all_reasons, "answers": all_answers_test}
        nb_reasons_dev += len(all_reasons)
        nb_reasons_test += len(all_reasons)
        nb_videos_dev += len(all_answers_dev)
        nb_videos_test += len(all_answers_test)

    with open(file_out1, 'w+') as fp:
        json.dump(dict_dev, fp)
    with open(file_out2, 'w+') as fp:
        json.dump(dict_test, fp)

    print("Stats dev & test")
    print("Nb actions: ")
    print(len(dict_dev), len(dict_test))
    print("Nb videos:")
    print(nb_videos_dev, nb_videos_test)
    print("Nb reasons:")
    print(nb_reasons_dev, nb_reasons_test)


def split_data_santi(file_in1, file_in2, file_in3, file_out1, file_out2):
    with open(file_in1) as json_file:
        dict_web_trial_dev = json.load(json_file)
    with open(file_in2) as json_file:
        dict_web_trial_test = json.load(json_file)
    with open(file_in3) as json_file:
        dict_sentences_per_verb_MARKERS_for_annotation_all50 = json.load(json_file)

    dict_dev = {}  # 80%
    dict_test = {}  # 20%
    verbs = ['buy', 'clean', 'cook', 'drink', 'drive', 'eat', 'fall', 'help', 'learn',
             'listen', 'paint', 'play', 'read', 'relax', 'sell', 'shop', 'sleep',
             'switch', 'thank', 'travel', 'walk', 'work', 'write']
    for verb in dict_web_trial_dev.keys():
        old_verb = verb
        if verb == "buying":
            verb = "buy"
        elif verb == "cleaning":
            verb = "clean"
        elif verb == "cooking":
            verb = "cook"
        elif verb == "drinking":
            verb = "drink"
        elif verb == "driving":
            verb = "drive"
        elif verb == "eating":
            verb = "eat"
        elif verb == "falling":
            verb = "fall"
        elif verb == "helping":
            verb = "help"
        elif verb == "listening":
            verb = "listen"
        elif verb == "learning":
            verb = "learn"
        elif verb == "painting":
            verb = "paint"
        elif verb == "playing":
            verb = "play"
        elif verb == "painting":
            verb = "paint"
        elif verb == "reading":
            verb = "read"
        elif verb == "relaxing":
            verb = "relax"
        elif verb == "remembering":
            verb = "remember"
        elif verb == "selling":
            verb = "sell"
        elif verb == "shopping":
            verb = "shop"
        elif verb == "sleeping":
            verb = "sleep"
        elif verb == "switching":
            verb = "switch"
        elif verb == "thanking":
            verb = "thank"
        elif verb == "travelling":
            verb = "travel"
        elif verb == "walking":
            verb = "walk"
        elif verb == "working":
            verb = "work"
        elif verb == "writing":
            verb = "write"
        data_all = dict_sentences_per_verb_MARKERS_for_annotation_all50[verb]
        dict_dev[verb] = []
        for sentence_full, answers in dict_web_trial_dev[old_verb]["answers"]:
            for data in data_all:
                full_s = data["sentence_before"] + " " + data["sentence"] + " " + data["sentence_after"]
                if full_s == sentence_full:
                    data["answers"] = answers
                    data["reasons"] = dict_web_trial_dev[old_verb]["reasons"]
                    dict_dev[verb].append(data)
                    break

    for verb in dict_web_trial_test.keys():
        old_verb = verb
        if verb == "buying":
            verb = "buy"
        elif verb == "cleaning":
            verb = "clean"
        elif verb == "cooking":
            verb = "cook"
        elif verb == "drinking":
            verb = "drink"
        elif verb == "driving":
            verb = "drive"
        elif verb == "eating":
            verb = "eat"
        elif verb == "falling":
            verb = "fall"
        elif verb == "helping":
            verb = "help"
        elif verb == "listening":
            verb = "listen"
        elif verb == "learning":
            verb = "learn"
        elif verb == "painting":
            verb = "paint"
        elif verb == "playing":
            verb = "play"
        elif verb == "painting":
            verb = "paint"
        elif verb == "reading":
            verb = "read"
        elif verb == "relaxing":
            verb = "relax"
        elif verb == "remembering":
            verb = "remember"
        elif verb == "selling":
            verb = "sell"
        elif verb == "shopping":
            verb = "shop"
        elif verb == "sleeping":
            verb = "sleep"
        elif verb == "switching":
            verb = "switch"
        elif verb == "thanking":
            verb = "thank"
        elif verb == "travelling":
            verb = "travel"
        elif verb == "walking":
            verb = "walk"
        elif verb == "working":
            verb = "work"
        elif verb == "writing":
            verb = "write"
        data_all = dict_sentences_per_verb_MARKERS_for_annotation_all50[verb]
        dict_test[verb] = []
        for sentence_full, answers in dict_web_trial_test[old_verb]["answers"]:
            for data in data_all:
                full_s = data["sentence_before"] + " " + data["sentence"] + " " + data["sentence_after"]
                if full_s == sentence_full:
                    data["answers"] = answers
                    data["reasons"] = dict_web_trial_test[old_verb]["reasons"]
                    dict_test[verb].append(data)
                    break

    with open(file_out1, 'w+') as fp:
        json.dump(dict_dev, fp)
    with open(file_out2, 'w+') as fp:
        json.dump(dict_test, fp)

## remove reasons that were not selected at all
def remove_unselected_reasons(file_output):
    with open(file_output) as json_file:
        dict_web_trial = json.load(json_file)
    for verb in dict_web_trial:
        reasons = dict_web_trial[verb]["reasons"]
        answers = []
        for [_, ans] in dict_web_trial[verb]["answers"]:
            for a in ans:
                answers.append(a)
        if 'I cannot find any reason mentioned verbally or shown visually in the video' not in answers:
            answers.append('I cannot find any reason mentioned verbally or shown visually in the video')
        answers = list(set(answers))
        diff = set(reasons) - set(answers)
        if diff:
            print(verb, answers, reasons, diff)
            print("----------------------------------")

def main():
    file_output = "data/baselines/dict_web_trial.json"
    file_input = "data/AMT/output/for_spam_detect/final_output/pipeline_trial.json"
    # remove_unselected_reasons(file_output)

    # file_output = "data/baselines/dict_web_trial_objects.json"
    # file_input = "data/video_baselines/pipeline_objects.json"
    # #
    # file_output = "data/baselines/dict_web_trial_captions.json"
    # file_input = "data/video_baselines/pipeline_captions.json"
    #
    # file_output = "data/baselines/dict_web_trial_objects_captions.json"
    # file_input = "data/video_baselines/pipeline_objects_captions.json"

    dict_GT_text_label = read_annotations(file_in=file_input, file_out=file_output)
    # verb = "clean"
    ## list_hyponyms = get_verb_hyponyms(verb="clean")

    # method = "majority"
    method = "cosine"
    # method = "NLI"

    file_out_dev = "data/baselines/dict_web_trial_dev.json"
    file_out_test = "data/baselines/dict_web_trial_test.json"


    # split_dev_test(file_in=file_output, file_out1=file_out_dev, file_out2=file_out_test)
    # split_dev_test(file_in=file_output, file_out1=file_out_test, file_out2=file_out_dev) #changed
    # split_dev_test_by_modality(file_in=file_output, file_out1="data/baselines/dict_web_trial_text.json",
    #                              file_out2="data/baselines/dict_web_trial_visual.json")

    # file_out_test = "data/baselines/dict_web_trial_text.json"
    # file_out_test = "data/baselines/dict_web_trial_visual.json"

    # split_data_santi(file_in1=file_out_dev, file_in2=file_out_test, file_in3="data/dict_sentences_per_verb_MARKERS_for_annotation_all50.json",
    #                  file_out1="data/baselines/dict_web_trial_dev_santi.json", file_out2="data/baselines/dict_web_trial_test_santi.json")

    if method == "majority":
        majority_class_baseline(file_in1=file_out_dev, file_in2=file_out_test, file_out="data/AMT/output/dict_majority_results_trial1.json")
        compute_metrics(file_in1="data/AMT/output/dict_majority_results_trial1.json", file_in2=file_out_test,
                        print_per_verb=True)

    elif method == "cosine":
        model = SentenceTransformer('stsb-roberta-base')  # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity

        threshold = 0.1 #finetuned on dev
        similarity_CN_transcript(threshold, model, is_SRL=True, file_in=file_out_test,
                                 file_out="data/AMT/output/dict_cosine_results_trial1.json")
        compute_metrics(file_in1="data/AMT/output/dict_cosine_results_trial1.json", file_in2=file_out_test,
                        print_per_verb=False)

    elif method == "NLI":  # TODO run remote
        # finetune = True
        finetune = False

        from transformers import pipeline
        # nli_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        nli_pipeline = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        nli_model = nli_pipeline.model
        tokenizer = nli_pipeline.tokenizer

        if finetune:
            file_out = "data/AMT/output/dict_NLI_finetune_results_trial1.json"
            NLI_finetune(nli_model, tokenizer, file_in_dev=file_out_dev, file_in_test=file_out_test,
                         file_out=file_out)
        else:
            # threshold = 0.8  # for transcript
            threshold = 0.1  # for video
            file_out = "data/AMT/output/dict_NLI_results_trial1.json"
            NLI(threshold, nli_model, tokenizer, file_in=file_out_test, file_out=file_out)  # TODO: Check painting
            compute_metrics(file_in1=file_out, file_in2=file_out_test, print_per_verb=True)

    # Roberta_multilabel(list_GT_text_label, conceptnet_labels)


if __name__ == '__main__':
    main()
