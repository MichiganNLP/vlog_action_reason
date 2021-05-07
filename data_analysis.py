import ssl
import json

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
    values[-1] = 5
    color_sequence[-1] = "white"
    fig.add_trace(go.Sunburst(
        ids=df2.ids,
        labels=df2.labels,
        parents=df2.parents,
        values=values,
        domain=dict(column=1),
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
        uniformtext=dict(minsize=25),
    )
    # fig.update_traces(textfont=dict(size=[50]))
    fig.write_image(file_out)
    # fig.show()


# scores can be agreement scores or nb videos
def plot_action_distrib(list_actions, list_scores, title, save_name, ylabel):
    df = pd.DataFrame({'action': list_actions, 'score': list_scores})
    # plt.figure(figsize=(15, 12))
    plt.figure(figsize=(15, 11))
    ax = sns.barplot(x="action", y="score", data=df, palette=sns.color_palette("husl", len(list_actions)))
    # ax.axhline(mean(list_scores))

    # ax.axes.set_title("Difference in top 3 emotions between inspiring and not inspiring posts: All_FB_Comments",fontsize=20)
    # ax.set_ylabel("action", fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    plt.savefig(save_name)


def main():
    file_out3 = "data/AMT/output/for_spam_detect/final_output/trial.json"
    dict_verb_label, list_confidence, list_whys = read_AMT_output(
        file_in1="data/AMT/output/for_spam_detect/edited_no_spam/all_batches.csv",
        file_out=file_out3)
    list_verb_agreement = agreement_AMT_output(dict_verb_label, list_confidence, list_whys)

    list_actions = sorted([a for [a, s] in list_verb_agreement])
    print(len(list_actions))
    print(list_actions)
    list_actions2 = ['buy', 'clean', 'cook', 'drink', 'drive', 'eat', 'fall', 'help', 'learn',
                     'listen', 'paint', 'play', 'read', 'relax', 'sell', 'shop', 'sleep',
                     'switch', 'thank', 'travel', 'walk', 'work', 'write']
    make_input_sunburst(list_actions2)
    sunburst_plot('data/analysis/action_reasons2_small.csv', file_out="data/analysis/img/tmp2.pdf")
    # sunburst_plot('data/analysis/action_reasons2.csv', file_out="data/analysis/img/all_actions.pdf")

    plot_distrib = False
    if plot_distrib:
        agreement = False
        if agreement:
            list_scores = [s for [a, s] in list_verb_agreement]
            title = "Fleiss Kappa agreement score per action"
            save_name = "data/analysis/img/distrib_agreement_per_action.pdf"
            ylabel = "score"
        else:
            counter = compute_stats(file_in="data/AMT/output/for_spam_detect/final_output/trial.json")
            list_scores = [counter[a] for a in list_actions]
            title = "Number of videos per action"
            save_name = "data/analysis/img/distrib_videos_per_action.pdf"
            ylabel = "#videos"

        plot_action_distrib(list_actions, list_scores, title, save_name, ylabel)


if __name__ == '__main__':
    main()