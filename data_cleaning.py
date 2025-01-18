import os, json

from pywin.framework.app import contributors


def get_all_file_names():
    file_names = []
    for filename in os.listdir("result/"):
        filepath = os.path.join("result/", filename)
        if os.path.isfile(filepath):  # 判断是否是文件
            file_names.append(filename)
    return file_names


def file_cleaning(file_names):
    for file_name in file_names:
        try:
            # 打开文件
            with open(os.path.join("result/", file_name), "r") as file:
                json_data = json.load(file)

                # 获取唯一的 URL key
                if len(json_data.keys()) != 1:
                    print(f"Unexpected keys in {file_name}: {json_data.keys()}")
                    continue
                url_key = list(json_data.keys())[0]  # 获取第一个 key

                # 提取对应的数据
                json_data = json_data[url_key]
                languages = json_data.get("languages", {})
                commits = json_data.get("commits", [])
                stars = json_data.get("stars", 0)
                forks = json_data.get("forks", 0)
                watchers = json_data.get("watchers", 0)
                topics = json_data.get("topics", [])
                contributors = json_data.get("contributors", 0)

                # 做data cleaning
                # TODO

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue


file_cleaning(["0voice-interview_internal_reference.json"])
