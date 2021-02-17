import json
import os
import datetime
import time

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
    command_save_video = 'youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -v -o ' + "data/videos/" + video_id + " " + url
    os.system(command_save_video)


def main():

    # with open('data/dict_sentences_per_verb.json') as json_file:
    with open('data/dict_sentences_per_verb_rachel.json') as json_file:
        data = json.load(json_file)

    # verb = "drink this"
    # list_verbs = ["read", "eat", "drink", "write", "clean"]
    ftr = [3600, 60, 1]
    for verb in data.keys():
        for s in data[verb][:15]:
            if verb != "read":
                continue
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



if __name__ == '__main__':
    main()
