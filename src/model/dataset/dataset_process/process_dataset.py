"""
    This code is to process the labeled dataset to the format that can be used by the model.
    One example of the dataset is as follow:
        [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "completed_by": 1,
                        "result": [
                            {
                                "value": {
                                    "start": 1,
                                    "end": 12,
                                    "text": "APTX-Fansub",
                                    "labels": [
                                        "Team"
                                    ]
                                },
                                "id": "EI-7rKAGWv",
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            },
                            {
                                "value": {
                                    "start": 14,
                                    "end": 29,
                                    "text": "Detective Conan",
                                    "labels": [
                                        "Series Name"
                                    ]
                                },
                                "id": "unjq3Viess",
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            },
                            {
                                "value": {
                                    "start": 32,
                                    "end": 36,
                                    "text": "1101",
                                    "labels": [
                                        "Episode Number"
                                    ]
                                },
                                "id": "uKFvdjunqN",
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            },
                            {
                                "value": {
                                    "start": 37,
                                    "end": 40,
                                    "text": "FHD",
                                    "labels": [
                                        "Resolution"
                                    ]
                                },
                                "id": "ER1BnSyfAz",
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            },
                            {
                                "value": {
                                    "start": 52,
                                    "end": 55,
                                    "text": "mp4",
                                    "labels": [
                                        "Format"
                                    ]
                                },
                                "id": "m5mT_EogLY",
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            }
                        ],
                        "was_cancelled": false,
                        "ground_truth": false,
                        "created_at": "2023-10-27T01:56:00.559926Z",
                        "updated_at": "2023-10-27T01:56:00.559963Z",
                        "draft_created_at": "2023-10-27T01:53:46.826285Z",
                        "lead_time": 117.793,
                        "prediction": {},
                        "result_count": 0,
                        "unique_id": "8403ba3b-8051-4258-9238-85a598233162",
                        "import_id": null,
                        "last_action": null,
                        "task": 1,
                        "project": 1,
                        "updated_by": 1,
                        "parent_prediction": null,
                        "parent_annotation": null,
                        "last_created_by": null
                    }
                ],
                "file_upload": "0b3b1080-titles.txt",
                "drafts": [],
                "predictions": [],
                "data": {
                    "text": "[APTX-Fansub] Detective Conan - 1101 FHD [4E00AC60].mp4"
                },
                "meta": {},
                "created_at": "2023-10-27T01:53:23.033687Z",
                "updated_at": "2023-10-27T01:56:00.661246Z",
                "inner_id": 1,
                "total_annotations": 1,
                "cancelled_annotations": 0,
                "total_predictions": 0,
                "comment_count": 0,
                "unresolved_comment_count": 0,
                "last_comment_updated_at": null,
                "project": 1,
                "updated_by": 1,
                "comment_authors": []
            },
        ]
    
        Our processed dataset will be SPAN format as below:
            {"label": [{"type": "ORG", "ent": "市委", "pos": [10, 11]}, {"type": "PER", "ent": "张敬涛", "pos": [14, 16]}], "text": "去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出三问：『假如上访群众是我们的父母姐妹，你会用什么样的感情对待他们？"}
            {"label": [{"type": "PER", "ent": "金大中", "pos": [5, 7]}], "text": "今年2月,金大中新政府成立后,社会舆论要求惩治对金融危机负有重大责任者。"}
            {"label": [], "text": "与此同时，作者同一题材的长篇侦破小说《鱼孽》也出版发行。"}
            The label is a list of dict, each dict contains the type of the entity, the entity itself and the position of the entity in the text.
            the type of the entity is the "labels" in the original dataset.
            the entity itself is the "text" in the original dataset.
            the position of the entity is the "start" and "end" in the original dataset.
            the text is the "text" in the original dataset.

"""

import json
from tqdm import tqdm

USE_LOWER_CASE=True

def load_json_file(file_path):
    """
        load the json file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_SPAN_text_file(file_path):
    """
        load json data, extract the text and annotations, save to SPAN format text file
    """
    # load
    print("loading json file...")
    data = load_json_file(file_path)
    # assert property
    print("asserting the property of the loaded json data...")
    assert_property(data)
    # extract and save
    print("extracting and saving...")
    # split the file into train, dev, test set by the ratio of 8:1:1
    train_num = int(len(data) * 0.8)
    dev_num = int(len(data) * 0.1)
    test_num = len(data) - train_num - dev_num
    # for train data
    train_file = open(file_path[:-5] + "_SPAN_train.txt", "w", encoding="utf-8")
    dev_file = open(file_path[:-5] + "_SPAN_dev.txt", "w", encoding="utf-8")
    test_file = open(file_path[:-5] + "_SPAN_test.txt", "w", encoding="utf-8")
    for idx, item in enumerate(tqdm(data)):
        # get the text and the annotations
        text = item["data"]["text"] # the text
        # if use lower case
        if USE_LOWER_CASE:
            text = text.lower()
        annotations = item["annotations"][0]["result"] # the annotations
        labels = []
        for annotation in annotations:
            label = annotation["value"]["labels"][0]
            ent = annotation["value"]["text"]
            start = annotation["value"]["start"]
            end = annotation["value"]["end"]+1
            # if use lower case
            if USE_LOWER_CASE:
                ent = ent.lower()
            labels.append({"type": label, "ent": ent, "pos": [start, end]})
        if idx < train_num:
            train_file.write(json.dumps({"label": labels, "text": text}, ensure_ascii=False) + "\n")
        elif idx < train_num + dev_num:
            dev_file.write(json.dumps({"label": labels, "text": text}, ensure_ascii=False) + "\n")
        else:
            test_file.write(json.dumps({"label": labels, "text": text}, ensure_ascii=False) + "\n")
    train_file.close()
    dev_file.close()
    test_file.close()
    print("Done!")

def assert_property(data):
    """
        Assert the property of the loaded json data, if not satisfied, raise error.
        Property:
            1. all label only have one label

        Summary:
            1. data entry number
            2. annotations entry number
            3. annotations label number
            4. all annotation label's list
    """
    entry_num = len(data)
    all_label = {}
    annotation_num = 0
    for item in tqdm(data):
        annotations = item["annotations"]
        annotation_num += len(annotations)
        for annotation in annotations:
            result = annotation["result"]
            for r in result:
                assert len(r["value"]["labels"]) == 1, "The label number of one annotation is not 1."
                label = r["value"]["labels"][0]
                if label not in all_label:
                    all_label[label] = 1
                else:
                    all_label[label] += 1
    print("data entry number: ", entry_num)
    print("annotations entry number: ", annotation_num)
    print("annotations label number: ", len(all_label))
    print("all annotation label's list: ", all_label)

if __name__ == "__main__":
    file_path = "./src/model/dataset/lll_dataset_v1/labeled_data.json"
    save_SPAN_text_file(file_path)

    
    