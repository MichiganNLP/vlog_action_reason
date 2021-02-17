import glob

def read_annotations():
    path = "/home/user/Action_Recog/OTHER/CrossTask/crosstask_release/annotations/"
    list_csv = glob.glob(path + "*.csv")
    for video in list_csv:
        print(video.split(".csv")[0].split("/")[-1])

        task = video.split(".csv")[0].split("/")[-1].split("_")[0]
        video_id = video.split(".csv")[0].split("/")[-1].split("_")[1:]
        if len(video_id) > 1:
            video_id = "_".join(video_id)
        else:
            video_id = video_id[0]
        print(video, task, video_id)



def main():
    read_annotations()

if __name__ == '__main__':
    main()