import json


def results_object_detection():
    with open('data/video_baselines/detectron_results.json') as json_file:
        detectron_results = json.load(json_file)

    for result in detectron_results:
        video = result["video"]
        detection = result["detection"]
        all_pred_classes = []
        for frame_pred in detection:
            for pred, score in zip(frame_pred["pred_classes"], frame_pred["scores"]):
                if score > 0.8:
                    all_pred_classes.append(pred)

        print(video, set(all_pred_classes))



def main():
    results_object_detection()

if __name__ == '__main__':
    main()
