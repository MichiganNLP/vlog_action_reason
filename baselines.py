from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging # if error - change runtime and try again
import json
import numpy as np


def read_annotations(file_output):
    with open('data/annotation_output/' + file_output) as json_file:
        dict_web_annotations = json.load(json_file)
    list_GT_text_label = []
    list_verbs = []
    for key in dict_web_annotations:
        verb, video, text = key.split(", ")
        list_verbs.append(verb)
        list_labels = dict_web_annotations[key]["oana"]
        list_GT_text_label.append([text, list_labels])

    #TODO: dict_concept_net TO dict_concept_net_clustered
    with open('data/dict_concept_net.json') as json_file:
        dict_concept_net = json.load(json_file)
    list_reasons = []
    for verb in list(set(list_verbs)):
        conceptnet_labels = dict_concept_net[verb]
        for label in conceptnet_labels:
            list_reasons.append(" ".join(label.split("_")))

    return list_verbs, list_GT_text_label, list_reasons


def SRL():
    return


def similarity_CN_transcript(list_GT_text_label, conceptnet_labels, is_SRL):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('stsb-roberta-base')  # models: https://www.sbert.net/docs/pretrained_models.html#semantic-textual-similarity

    list_transcripts = []
    for [text, _] in list_GT_text_label:
        list_transcripts.append(text)

    #Compute embedding for both lists
    list_emb_transcripts = model.encode(list_transcripts, convert_to_tensor=True)
    list_emb_reasons = model.encode(conceptnet_labels, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(list_emb_transcripts, list_emb_reasons)

    # Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(list_emb_transcripts)):
        for j in range(len(list_emb_reasons)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    for pair in pairs[0:10]:
        i, j = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(list_transcripts[i], conceptnet_labels[j], pair['score']))

    print(cosine_scores.shape)
    print(len(list_transcripts))
    print(len(conceptnet_labels))
    #TODO: finish - add threshold
    #TODO: Note, in the above approach we use a brute-force approach to find the highest scoring pairs,
    # which has a quadratic complexity. For long lists of sentences, this might be infeasible.
    # If you want find the highest scoring pairs in a long list of sentences, have a look at Paraphrase Mining.
    # https://www.sbert.net/docs/usage/semantic_textual_similarity.html


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

def NLI(list_GT_text_label, conceptnet_labels):
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    sequence_to_classify = [l[0] for l in list_GT_text_label][:10]
    candidate_labels = conceptnet_labels
    print(len(sequence_to_classify))
    list_dicts = classifier(sequence_to_classify, candidate_labels, multi_class=True)
    print(list_dicts)

    threshold = 0.5
    list_predicted_output = []
    for d in list_dicts:
        scores = d["scores"]
        labels = d["labels"]
        text = d["sequence"]
        list_labels = []
        for s, l in zip(scores, labels):
            if s > threshold:
                list_labels.append(l)
        list_predicted_output.append([text, list_labels])
        print([text, list_labels])
    return list_predicted_output

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

def calculate_metrics(list_GT_text_label, list_predicted_output, conceptnet_labels):
    from sklearn.metrics import label_ranking_average_precision_score, hamming_loss, accuracy_score, jaccard_score

    one_hot_list_text_label_GT = label_to_onehot(list_GT_text_label, conceptnet_labels)
    one_hot_list_text_label_P = label_to_onehot(list_predicted_output, conceptnet_labels)

    list_gt_labels = []
    list_p_labels = []
    for [text, label] in one_hot_list_text_label_GT:
        list_gt_labels.append(label)
    for [text, label] in one_hot_list_text_label_P:
        list_p_labels.append(label)

    y_true = np.array(list_gt_labels)
    y_pred = np.array(list_p_labels)
    # label_ranking_average_precision_score(y_true, y_score)

    print("label_ranking_average_precision_score:", label_ranking_average_precision_score(y_true, y_pred))
    print("accuracy_score:", accuracy_score(y_true, y_pred))
    print("jaccard_score:", jaccard_score(y_true, y_pred, average='samples'))
    print("Hamming_loss: (smaller is better)", hamming_loss(y_true, y_pred))


def main():
    list_verbs, list_GT_text_label, conceptnet_labels = read_annotations(file_output="dict_web_annotations_for_agreement.json")

    # similarity_CN_transcript(list_GT_text_label, conceptnet_labels, is_SRL=False)
    # NLI(list_GT_text_label, conceptnet_labels)
    Roberta_multilabel(list_GT_text_label, conceptnet_labels)


if __name__ == '__main__':
    main()
