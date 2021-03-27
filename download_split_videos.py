import glob
import json
import os
import datetime
import time
import cv2
import shutil
import numpy as np
import pandas as pd

def split_video_by_time(video_id, time_start, time_end, verb):
    duration = time_end - time_start
    print(time_start)
    print(time_end)
    print(duration)
    time_start = str(datetime.timedelta(seconds=time_start))
    time_end = str(datetime.timedelta(seconds=time_end))
    duration = str(datetime.timedelta(seconds=duration))
    print(time_start)
    print(time_end)
    command_split_video = 'ffmpeg -ss ' + time_start + ' -i ' + 'data/videos/' + video_id + '.mp4 ' + '-to ' + duration + \
                    ' -c copy data/videos/splits/' + verb + "/" + video_id + '_' + time_start + '_' + time_end + '.mp4'
    os.system(command_split_video)
    print(command_split_video)


def download_video(video_id):
    url = "https://www.youtube.com/watch?v=" + video_id
    command_save_video = 'youtube-dl --no-check-certificate -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -v -o ' \
                         + "data/videos/" + video_id + " " + url
    os.system(command_save_video)


def filter_split_by_motion(PATH_miniclips, PATH_problematic_videos, PARAM_CORR2D_COEFF= 0.8):
    print("filtering videos by motion")
    if not os.path.exists(PATH_problematic_videos):
        os.makedirs(PATH_problematic_videos)

    list_video_names_removed = []
    list_videos = sorted(glob.glob(PATH_miniclips + "*.mp4"), key=os.path.getmtime)

    for video in list_videos:
        vidcap = cv2.VideoCapture(video)

        if (vidcap.isOpened() == False):
            continue
            # vidcap.open(video)

        corr_list = []
        video_name = video.split("/")[-1]
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_nb_1 in range(0, length - 100, 100):
            vidcap.set(1, frame_nb_1)
            success, image = vidcap.read()
            if success == False:
                continue
            # image1 = cv2.resize(image, (100, 50))
            gray_image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # for frame_nb_2 in range(frame_nb_1 + 100, length, 100):
            frame_nb_2 = frame_nb_1 + 100
            vidcap.set(1, frame_nb_2)
            success, image = vidcap.read()
            if success == False:
                continue
            # image2 = cv2.resize(image, (100, 50))
            gray_image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corr2_matrix = np.corrcoef(gray_image_1.reshape(-1), gray_image_2.reshape(-1))
            corr2 = corr2_matrix[0][1]
            corr_list.append(corr2)

        if np.median(corr_list) >= PARAM_CORR2D_COEFF:
            # move video in another folder
            list_video_names_removed.append(video_name)
            shutil.move(video, PATH_problematic_videos + video_name)

    return list_video_names_removed

def download_from_dict():
    # with open('data/dict_sentences_per_verb2.json') as json_file:
    with open('data/dict_sentences_per_verb_REASONS.json') as json_file:
        # with open('data/dict_sentences_per_verb_reasons.json') as json_file:
        data = json.load(json_file)

    # verb = "drink this"
    # list_verbs = ["read", "eat", "drink", "write", "clean"]
    ftr = [3600, 60, 1]
    list_all_video_names_removed = []
    list_actions = data.keys()
    # list_actions = ["clean", "read", "write", "sleep", "eat", "travel", "shop", "sew", "run", "listen"]
    # list_actions = ["travel", "shop", "sew", "run", "listen"]
    for verb in list_actions:
        # for s in data[verb][:1000]: #TODO: MAYBE REMOVE LIMIT
        for reason in data[verb].keys():
            if data[verb][reason]:
                for s in data[verb][reason][:10]:  # TODO: MAYBE REMOVE LIMIT
                    # print(verb.split())
                    x = time.strptime(s["time_s"].split('.')[0], '%H:%M:%S')
                    time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
                    x = time.strptime(s["time_e"].split('.')[0], '%H:%M:%S')
                    time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

                    # time_s = round(s["time_s"])
                    # time_e = round(s["time_e"])
                    video_id = s["video"]
                    verb = "_".join(verb.split())
                    if not os.path.exists('data/videos/' + video_id + ".mp4"):
                        download_video(video_id)
                    if not os.path.exists('data/videos/splits/' + verb):
                        os.makedirs('data/videos/splits/' + verb)
                    split_video_by_time(video_id, time_s, time_e, verb)

        list_video_names_removed = filter_split_by_motion(PATH_miniclips="data/videos/splits/" + verb + "/",
                                                          PATH_problematic_videos="data/videos/splits/" + verb + "/filtered_out/",
                                                          PARAM_CORR2D_COEFF=0.9)
        # 0.8 by default
        for video_name in list_video_names_removed:
            list_all_video_names_removed.append(video_name)

    df = pd.DataFrame({'videos_to_remove': list_all_video_names_removed})
    df.to_csv('data/videos_to_remove.csv')

def download_from_AMT_input(file_in):
    list_all_video_names_removed = []

    df = pd.read_csv(file_in)
    for video_url, verb, reasons in zip(df["video_url"], df["action"], df["reasons"]):
        video_id, time_s, time_e = video_url.split("https://github.com/OanaIgnat/miniclips/blob/master/")[1].split("||")
        time_e = time_e.split(".mp4?raw=true")[0]

        x = time.strptime(time_s.split('.')[0], '%H:%M:%S')
        time_s = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
        x = time.strptime(time_e.split('.')[0], '%H:%M:%S')
        time_e = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

        if not os.path.exists('data/videos/' + video_id + ".mp4"):
            download_video(video_id)
        if not os.path.exists('data/videos/splits/' + verb):
            os.makedirs('data/videos/splits/' + verb)
        split_video_by_time(video_id, time_s, time_e, verb)

    for verb in list(set(df["action"])):
        list_video_names_removed = filter_split_by_motion(PATH_miniclips="data/videos/splits/" + verb + "/",
                                                          PATH_problematic_videos="data/videos/splits/" + verb + "/filtered_out/",
                                                          PARAM_CORR2D_COEFF=0.8)
    # 0.8 by default
    for video_name in list_video_names_removed:
        list_all_video_names_removed.append(video_name)

    df = pd.DataFrame({'videos_to_remove': list_all_video_names_removed})
    df.to_csv('data/AMT/videos_to_remove.csv')

def main():
    download_from_AMT_input(file_in="data/AMT/trial1.csv")


if __name__ == '__main__':
    main()
