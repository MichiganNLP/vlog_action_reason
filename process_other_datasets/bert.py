# import json
# import time
# import torch #commented to not load tensorflow
# import numpy as np
# from tqdm import tqdm
# from transformers.tokenization_auto import AutoTokenizer
# from transformers.modeling_bert import BertModel
#
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
#
# def get_bert_finetuned_embeddings(model, tokenizer, action):
#     marked_text = "[CLS] " + action + " [SEP]"
#     tokenized_text = tokenizer.tokenize(marked_text)
#
#     indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#     segments_ids = [1] * len(tokenized_text)
#
#     # Convert inputs to PyTorch tensors
#     tokens_tensor = torch.tensor([indexed_tokens])
#     segments_tensors = torch.tensor([segments_ids])
#
#     # Predict hidden states features for each layer
#     with torch.no_grad():
#         model_output = model(tokens_tensor, segments_tensors)
#         last_hidden_states = model_output[0]  # The last hidden-state is the first element of the output tuple
#
#     # [CLS] token representation would be: sentence_embedding = last_hidden_states[:, 0, :]
#
#     sentence_embedding = torch.mean(last_hidden_states, 1)  # average token embeddings
#
#     return sentence_embedding.cpu().numpy()
#
# def create_bert_embeddings(list_all_actions, path_output):
#     tokenizer_name = 'bert-base-uncased'
#
#     # pretrained_model_name = 'data/epoch_29/'
#     pretrained_model_name = tokenizer_name
#
#     start = time.time()
#     # Load pre-trained model tokenizer (vocabulary)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     # Load pre-trained model (weights)
#     # model = PreTrainedModel.from_pretrained(pretrained_model_name)
#     model = BertModel.from_pretrained(pretrained_model_name)
#     # Put the model in "evaluation" mode, meaning feed-forward operation.
#     model.eval()
#     end = time.time()
#     print("Load BERT model took " + str(end - start))
#     dict_action_embeddings = {}
#     print("Running BERT ... ")
#     for action in tqdm(list_all_actions):
#         emb_action = get_bert_finetuned_embeddings(model, tokenizer, action)
#         # emb_action = finetune_bert(model,tokenizer, action)
#         dict_action_embeddings[action] = emb_action.reshape(-1)
#
#     with open(path_output, 'w+') as outfile:
#     # with open('data/embeddings/dict_action_embeddings_Bert_COIN.json', 'w+') as outfile:
#         json.dump(dict_action_embeddings, outfile, cls=NumpyEncoder)
#     return dict_action_embeddings