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

def get_all_captions(captions, video):
    list_captions = []
    for dict in captions[video]:
        if dict["sentence"] not in captions:
            list_captions.append(dict["sentence"])
    return ". ".join(list_captions)

def results_captions(file_in1, file_in2, file_out):
    with open(file_in1) as json_file:
        captions = json.load(json_file)
    with open(file_in2) as json_file:
        pipeline_trial = json.load(json_file)

    dict_pipeline_captions = {}
    for video in captions.keys():
        for video_verb in pipeline_trial.keys():
            if video in video_verb:
                if video_verb not in dict_pipeline_captions:
                    dict_pipeline_captions[video_verb] = []
                    reasons_gt = pipeline_trial[video_verb][1]
                    reasons_pred = pipeline_trial[video_verb][2]
                    caption_str = get_all_captions(captions, video)
                    dict_pipeline_captions[video_verb].append(caption_str)
                    dict_pipeline_captions[video_verb].append(reasons_gt)
                    dict_pipeline_captions[video_verb].append(reasons_pred)
                    break
                else:
                    print("Error with " + video_verb)

    with open(file_out, 'w+') as fp:
        json.dump(dict_pipeline_captions, fp)

def get_all_objects(objects, video):
    list_objects = []
    for dict in objects[video]:
        for object in dict["pred_classes"]:
            if object not in list_objects:
                list_objects.append(object)
    return ". ".join(list_objects)

def results_objects(file_in1, file_in2, file_out):
    with open(file_in1) as json_file:
        objects = json.load(json_file)
    with open(file_in2) as json_file:
        pipeline_trial = json.load(json_file)

    dict_pipeline_objects = {}
    for video in objects.keys():
        for video_verb in pipeline_trial.keys():
            if video in video_verb:
                if video_verb not in dict_pipeline_objects:
                    dict_pipeline_objects[video_verb] = []
                    reasons_gt = pipeline_trial[video_verb][1]
                    reasons_pred = pipeline_trial[video_verb][2]
                    object_str = get_all_objects(objects, video)
                    dict_pipeline_objects[video_verb].append(object_str)
                    dict_pipeline_objects[video_verb].append(reasons_gt)
                    dict_pipeline_objects[video_verb].append(reasons_pred)
                    break
                else:
                    print("Error with " + video_verb)

    with open(file_out, 'w+') as fp:
        json.dump(dict_pipeline_objects, fp)

def get_all_objects_captions(captions, objects, video):
    list_objects = []
    list_captions = []
    for dict in objects[video]:
        for object in dict["pred_classes"]:
            if object not in list_objects:
                list_objects.append(object)
    for dict in captions[video]:
        if dict["sentence"] not in captions:
            list_captions.append(dict["sentence"])

    return ". ".join(list_captions) + ". " + ". ".join(list_objects)

def results_objects_captions(file_in1, file_in2, file_in3, file_out):
    with open(file_in1) as json_file:
        captions = json.load(json_file)
    with open(file_in2) as json_file:
        objects = json.load(json_file)
    with open(file_in3) as json_file:
        pipeline_trial = json.load(json_file)

    dict_pipeline_objects = {}
    for video in objects.keys():
        for video_verb in pipeline_trial.keys():
            if video in video_verb:
                if video_verb not in dict_pipeline_objects:
                    dict_pipeline_objects[video_verb] = []
                    reasons_gt = pipeline_trial[video_verb][1]
                    reasons_pred = pipeline_trial[video_verb][2]
                    object_str = get_all_objects_captions(captions, objects, video)
                    dict_pipeline_objects[video_verb].append(object_str)
                    dict_pipeline_objects[video_verb].append(reasons_gt)
                    dict_pipeline_objects[video_verb].append(reasons_pred)
                    break
                else:
                    print("Error with " + video_verb)

    with open(file_out, 'w+') as fp:
        json.dump(dict_pipeline_objects, fp)

def main():
    ## results_object_detection()
    # results_captions(file_in1='data/video_baselines/captions.json',
    #                  file_in2="data/AMT/output/for_spam_detect/final_output/pipeline_trial.json",
    #                  file_out='data/video_baselines/pipeline_captions.json')

    # results_objects(file_in1='data/video_baselines/object_detections.json',
    #                  file_in2="data/AMT/output/for_spam_detect/final_output/pipeline_trial.json",
    #                  file_out='data/video_baselines/pipeline_objects.json')

    results_objects_captions(file_in1='data/video_baselines/captions.json',
                             file_in2='data/video_baselines/object_detections.json',
                    file_in3="data/AMT/output/for_spam_detect/final_output/pipeline_trial.json",
                    file_out='data/video_baselines/pipeline_objects_captions.json')

if __name__ == '__main__':
    main()
