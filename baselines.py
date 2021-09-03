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


def similarity_CN_transcript(threshold, model, is_SRL, file_in, file_out):
    with open(file_in) as json_file:
        dict_gt_text_label = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}
    for verb in dict_gt_text_label:
        candidate_labels = dict_gt_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_gt_text_label = dict_gt_text_label[verb]["answers"]

        transcripts = [text_label[0] for text_label in list_gt_text_label]
        for [transcript, annotated_labels, _] in list_gt_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

        # if SRL, compare the reasons extracted from transcripts with CN labels, else, compare CN with the transcripts
        if not is_SRL:
            list_emb_transcripts = model.encode(transcripts, convert_to_tensor=True)
        else:
            list_srl_reasons = []
            for transcript in transcripts:
                list_casual_markers = [" because ", " since ", " so that is why ", " thus ", " therefore "]
                for marker in list_casual_markers:
                    if marker in transcript:
                        pos_marker = transcript.find(marker)
                        reason = transcript[pos_marker - 100:pos_marker + 100]
                        list_srl_reasons.append(reason)
                        break
            list_emb_reasons = model.encode(list_srl_reasons, convert_to_tensor=True)
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


def NLI(threshold, nli_model, tokenizer, file_in, file_out):
    with open(file_in) as json_file:
        dict_gt_text_label = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}
    nli_model = nli_model.to(DEVICE)

    for verb in tqdm.tqdm(dict_gt_text_label.keys()):
        candidate_labels = dict_gt_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_gt_text_label = dict_gt_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_gt_text_label]
        for [transcript, annotated_labels, _] in list_gt_text_label:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

        for premise in transcripts:
            list_predicted_labels = []
            for label in candidate_labels:
                hypothesis = f'The reason for {verb} is {label}.'
                # run through model pre-deved on MNLI
                x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                     truncation_strategy='only_first').to(DEVICE)
                logits = nli_model(x)[0]
                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [nli_model.config.label2id["contradiction"],
                                                         nli_model.config.label2id["entailment"]]]
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


def prepare_data(dict_gt_text_label):
    all_list_labels, all_list_premises, all_list_hypotheses = [], [], []
    for verb in list(dict_gt_text_label.keys()):
        candidate_labels = dict_gt_text_label[verb]["reasons"][:-1]  # without "I cannot find"
        list_gt_text_label = dict_gt_text_label[verb]["answers"]

        transcripts = [l[0] for l in list_gt_text_label]
        labels = [l[1] for l in list_gt_text_label]
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
        dict_gt_text_label_dev = json.load(json_file)
    all_list_labels_dev, all_list_premises_dev, all_list_hypotheses_dev = prepare_data(dict_gt_text_label_dev)
    # all_list_labels_dev, all_list_premises_dev, all_list_hypotheses_dev = all_list_labels_dev[
    #                                                                             :10], all_list_premises_dev[
    #                                                                                   :10], all_list_hypotheses_dev[
    #                                                                                         :10]

    with open(file_in_test) as json_file:
        dict_gt_text_label_test = json.load(json_file)
    all_list_labels_eval, all_list_premises_eval, all_list_hypotheses_eval = prepare_data(dict_gt_text_label_test)
    # all_list_labels_eval, all_list_premises_eval, all_list_hypotheses_eval = all_list_labels_dev[
    #                                                                          :10], all_list_premises_dev[
    #                                                                                :10], all_list_hypotheses_dev[:10]

    dev_encodings = tokenizer(all_list_premises_dev, all_list_hypotheses_dev, truncation=True, padding=True)
    test_encodings = tokenizer(all_list_premises_eval, all_list_hypotheses_eval, truncation=True, padding=True)

    dev_dataset = ReasonDataset(dev_encodings, all_list_labels_dev)
    test_dataset = ReasonDataset(test_encodings, all_list_labels_eval)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions[:,
                [nli_model.config.label2id["contradiction"], nli_model.config.label2id["entailment"]]].argmax(-1)
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
    dever.dev()
    print("eval")
    dever.evaluate(ignore_keys=["encoder_last_hidden_state"])


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
    all_reasons_initial = list_reasons[0]
    list_verb_f1 = []

    for reasons_pred, reasons_gt, all_reasons, verb in zip(list_predicted, list_gt, list_reasons, list_verbs):
        if verb != verb_initial:
            list_all = list_gt_labels + list_p_labels
            y_all = MultiLabelBinarizer(classes=range(len(all_reasons_initial))).fit_transform(list_all)
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
    y_all = MultiLabelBinarizer(classes=range(len(all_reasons_initial))).fit_transform(list_all)
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


def majority_class_baseline(file_in1, file_in2, file_out):
    with open(file_in1) as json_file:
        dict_gt_text_label_dev = json.load(json_file)

    with open(file_in2) as json_file:
        dict_gt_text_label_test = json.load(json_file)

    dict_results = {"gt": {}, "predicted": {}}

    for verb in dict_gt_text_label_dev:
        list_gt_labels_dev = [l[1] for l in dict_gt_text_label_dev[verb]["answers"]]
        all_labels = [item for sublist in list_gt_labels_dev for item in sublist]
        counter = Counter(all_labels).most_common()
        max_nb_times = counter[0][1]
        majority_class_list = []
        for (reason, nb_times) in counter:
            if nb_times == max_nb_times:
                majority_class_list.append(reason)

        list_gt_text_label_test = dict_gt_text_label_test[verb]["answers"]
        for [transcript, annotated_labels, _] in list_gt_text_label_test:
            if str((verb, transcript)) not in dict_results["gt"].keys():
                dict_results["gt"][str((verb, transcript))] = []
            dict_results["gt"][str((verb, transcript))].append(annotated_labels)

            if str((verb, transcript)) not in dict_results["predicted"].keys():
                dict_results["predicted"][str((verb, transcript))] = []
            dict_results["predicted"][str((verb, transcript))].append(majority_class_list)

    with open(file_out, 'w+') as fp:
        json.dump(dict_results, fp)

    return dict_results


def main():
    # method = "majority"
    method = "cosine"
    # method = "NLI"

    # use for multimodal and video models
    # file_out_dev = "data/test.json"
    # file_out_test = "data/dev.json"

    # same data as dev.json and test.json, only formatted differently and have only text info
    file_out_dev = "data/dict_text_dev.json"
    file_out_test = "data/dict_text_test.json"

    if method == "majority":
        majority_class_baseline(file_in1=file_out_dev, file_in2=file_out_test,
                                file_out="data/output/dict_majority_results.json")
        compute_metrics(file_in1="data/output/dict_majority_results.json", file_in2=file_out_test,
                        print_per_verb=True)

    elif method == "cosine":
        model = SentenceTransformer(
            'stsb-roberta-base')  # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity

        threshold = 0.1  # tuned on dev
        similarity_CN_transcript(threshold, model, is_SRL=True, file_in=file_out_test,
                                 file_out="data/output/dict_cosine_results.json")
        compute_metrics(file_in1="data/output/dict_cosine_results.json", file_in2=file_out_test,
                        print_per_verb=False)

    elif method == "NLI":  # TODO run on GPU
        # finetune = True
        finetune = False

        from transformers import pipeline
        # nli_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        nli_pipeline = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        nli_model = nli_pipeline.model
        tokenizer = nli_pipeline.tokenizer

        if finetune:
            file_out = "data/output/dict_NLI_finetune_results.json"
            NLI_finetune(nli_model, tokenizer, file_in_dev=file_out_dev, file_in_test=file_out_test,
                         file_out=file_out)
        else:
            # threshold = 0.8  # for transcript
            threshold = 0.1  # for video
            file_out = "data/output/dict_NLI_results.json"
            NLI(threshold, nli_model, tokenizer, file_in=file_out_test, file_out=file_out)
            compute_metrics(file_in1=file_out, file_in2=file_out_test, print_per_verb=True)


if __name__ == '__main__':
    main()
