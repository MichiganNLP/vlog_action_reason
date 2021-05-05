import ssl
import json
ssl._create_default_https_context = ssl._create_unverified_context
import plotly.graph_objects as go
import pandas as pd


def make_input_sunburst():
    with open("data/dict_concept_net_clustered_manual.json") as json_file:
        dict_concept_net_clustered = json.load(json_file)

    index_root = 1
    labels = []
    ids = []
    parents = []
    index = 1
    for action in dict_concept_net_clustered:
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

def sunburst_plot(pathname):
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
        uniformtext=dict(minsize=12, mode='show'),
    )
    # fig.update_traces(textfont=dict(size=[50]))
    fig.write_image("data/analysis/img/tmp2.pdf")
    # fig.show()


def main():
    # make_input_sunburst()
    sunburst_plot('data/analysis/action_reasons2_small.csv')


if __name__ == '__main__':
    main()