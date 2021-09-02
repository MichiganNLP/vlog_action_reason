import ssl
import json

import ast
from matplotlib import pyplot as plt
import seaborn as sns
from statistics import mean

from compute_stats import compute_stats
from data_prep_annotation import read_AMT_output, agreement_AMT_output

ssl._create_default_https_context = ssl._create_unverified_context
import plotly.graph_objects as go
import pandas as pd


def make_input_sunburst(list_actions):
    with open("data/dict_concept_net_clustered_manual.json") as json_file:
    # with open("data/dict_concept_net_clustered_manual_modif.json") as json_file:
        dict_concept_net_clustered = json.load(json_file)

    index_root = 1
    labels = []
    ids = []
    parents = []
    index = 1
    # for action in dict_concept_net_clustered:
    for action in list_actions:
        labels.append(action)
        parents.append('')
        ids.append(index)
        reasons = dict_concept_net_clustered[action]
        for reason in reasons:
            index += 1
            labels.append(reason)
            parents.append(index_root)
            ids.append(index)
        index += 1
        index_root += (len(reasons) + 1)

    print(ids)
    print(labels)
    print(parents)

    df = pd.DataFrame({'ids': ids, 'labels': labels, 'parents': parents})
    df.to_csv('data/analysis/action_reasons2.csv', index=False)

def sunburst_plot2():
    df2 = pd.read_csv('data/analysis/action_reasons2.csv')

    fig = go.Figure()

    fig.add_trace(go.Sunburst(
        ids=df2.ids,
        labels=df2.labels,
        parents=df2.parents,
        domain=dict(column=1),
        maxdepth=-1,
        insidetextorientation='auto'
        # opacity=1
    ))

    fig.update_layout(
        grid=dict(columns=1, rows=1),
        margin=dict(t=0, l=0, r=0, b=0)
    )
    fig.update_layout(uniformtext=dict(minsize=6, mode='show'))
    # fig.update_traces(textfont=dict(size=[20]))
    fig.write_image("data/analysis/img/fig1.pdf")

    # fig.show()

def sunburst_plot(pathname, file_out):
    df2 = pd.read_csv(pathname)
    fig = go.Figure()
    # fig.add_trace(go.Sunburst(
    #     ids=df1.ids,
    #     labels=df1.labels,
    #     parents=df1.parents,
    #     domain=dict(column=0)
    # ))
    values = [1 for i in df2.ids]
    color_sequence = ["" for i in df2.ids]
    # values[-1] = 3
    # color_sequence[-1] = "white"
    fig.add_trace(go.Sunburst(
        ids=df2.ids,
        labels=df2.labels,
        parents=df2.parents,
        values=values,
        domain=dict(column=0),
        maxdepth=-1,
        insidetextorientation='auto',
        branchvalues="remainder",
        marker=dict(colors=color_sequence)
        # opacity=1
    ))
    fig.update_layout(
        grid=dict(columns=1, rows=1),
        margin=dict(t=0, l=0, r=0, b=0),
        # uniformtext=dict(minsize=5, mode='show'), #for entire
        uniformtext=dict(minsize=8,  mode='show'),
        # plot_bgcolor="#C4C4C4",
        font=dict(
            size=20,
            color="black"
    )
    )
    # fig.update_traces(textfont=dict(size=[50]))
    fig.write_image(file_out)
    # fig.show()


# scores can be agreement scores or nb videos
def plot_action_distrib(list_actions, list_scores, title, save_name, ylabel):
    new_list_action = []
    for verb in list_actions:
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
        new_list_action.append(verb)
    df = pd.DataFrame({'action': new_list_action, 'score': list_scores})
    # plt.figure(figsize=(15, 12))
    plt.figure(figsize=(17, 17))
    ax = sns.barplot(x="action", y="score", data=df, palette=sns.color_palette("husl", len(list_actions)))
    ax.axhline(mean(list_scores))

    # ax.axes.set_title("Difference in top 3 emotions between inspiring and not inspiring posts: All_FB_Comments",fontsize=20)
    # ax.set_ylabel("action", fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
    ax.set_ylabel(ylabel, fontsize=25)
    # ax.set_title(title, fontsize=20)
    plt.savefig(save_name)


def main():
    with open("data/AMT/output/for_spam_detect/final_output/trial.json") as json_file:
        annotations_clean = json.load(json_file)

    file_out3 = "data/AMT/output/for_spam_detect/final_output/trial_initial.json"
    # I removed 2 corrupted videos from trial.json
    dict_verb_label, list_confidence, list_whys = read_AMT_output(
        file_in1="data/AMT/output/for_spam_detect/edited_no_spam/all_batches.csv",
        file_out=file_out3)

    new_dict_verb_label = {}
    for key in annotations_clean:
        (video, verb) = ast.literal_eval(key)
        labels = annotations_clean[key]["labels"]
        if verb not in new_dict_verb_label:
            new_dict_verb_label[verb] = {"labels": [], "GT": dict_verb_label[verb]["GT"]}
        for label in labels:
            list_label = ast.literal_eval(label)
            new_dict_verb_label[verb]["labels"].append(list_label)

    list_verb_agreement = agreement_AMT_output(new_dict_verb_label, list_confidence, list_whys)

    list_actions = sorted([a for [a, s] in list_verb_agreement])
    print(len(list_actions))
    print(list_actions)
    list_actions2 = ['buy', 'clean', 'cook', 'drink', 'drive', 'eat', 'fall', 'help', 'learn',
                     'listen', 'paint', 'play', 'read', 'relax', 'sell', 'shop', 'sleep',
                     'switch', 'thank', 'travel', 'walk', 'work', 'write']
    # make_input_sunburst(list_actions2)
    # sunburst_plot('data/analysis/action_reasons2_small.csv', file_out="data/analysis/img/tmp2.pdf")
    # # sunburst_plot('data/analysis/action_reasons2_small.csv', file_out="data/analysis/img/tmp2.jpg")
    # sunburst_plot('data/analysis/action_reasons2.csv', file_out="data/analysis/img/all_actions.pdf")

    plot_distrib = True
    if plot_distrib:
        agreement = False
        video = False
        reasons = False
        results = True
        clusters = False
        if agreement:
            list_scores = [s for [a, s] in list_verb_agreement]
            title = "Fleiss Kappa agreement score per action"
            save_name = "data/analysis/img/distrib_agreement_per_action.pdf"
            # save_name = "data/analysis/img/distrib_agreement_per_action.jpg"
            ylabel = "score"
            plot_action_distrib(list_actions, list_scores, title, save_name, ylabel)
        if video:
            counter = compute_stats(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")
            list_scores = [counter[a] for a in list_actions]
            title = "Number of videos per action"
            save_name = "data/analysis/img/distrib_videos_per_action.pdf"
            # save_name = "data/analysis/img/distrib_videos_per_action.jpg"
            ylabel = "#videos"
            plot_action_distrib(list_actions, list_scores, title, save_name, ylabel)
        if reasons:
            counter = compute_stats(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")
            list_scores = [len(dict_verb_label[action]["GT"]) - 1 for action in list_actions] # minus "no reason answer"
            title = "Number of reasons per action"
            save_name = "data/analysis/img/distrib_reasons_per_action.pdf"
            # save_name = "data/analysis/img/distrib_reasons_per_action.jpg"
            ylabel = "#reasons"
            plot_action_distrib(list_actions, list_scores, title, save_name, ylabel)
        if results:
            d = {
                  "clean": 0.3763757050037384,
                  "drink": 0.35014334321022034,
                  "work": 0.4499551057815552,
                  "write": 0.3353785276412964,
                  "thank": 0.5662088394165039,
                  "buy": 0.47021564841270447,
                  "sleep": 0.4378787577152252,
                  "eat": 0.4787960648536682,
                  "help": 0.28349804878234863,
                  "learn": 0.3823215067386627,
                  "play": 0.34709495306015015,
                  "listen": 0.44819313287734985,
                  "cook": 0.5179694890975952,
                  "walk": 0.36657682061195374,
                  "read": 0.42611831426620483,
                  "fall": 0.34730851650238037,
                  "remember": 0.3860411047935486,
                  "travel": 0.41893941164016724,
                  "shop": 0.5411959886550903,
                  "sell": 0.41181322932243347,
                  "drive": 0.355869323015213,
                  "switch": 0.2889033257961273,
                  "relax": 0.5423280000686646,
                  "paint": 0.33763498067855835
                }
            actions = sorted(d.keys())
            print(actions)
            list_scores = [d[a] for a in actions]
            title = "F1 score per action"
            save_name = "data/analysis/img/distrib_F1_per_action.pdf"
            # save_name = "data/analysis/img/distrib_agreement_per_action.jpg"
            ylabel = "score"
            plot_action_distrib(list_actions, list_scores, title, save_name, ylabel)

        if clusters:
            file_in = "data/analysis/clusters.json"
            with open(file_in) as json_file:
                clusters = json.load(json_file)


if __name__ == '__main__':
    main()