import multiprocessing
import time
import requests
import re
import json
import os
import shutil
from json import JSONDecodeError

# GitHub API Token
GITHUB_TOKEN = "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"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# 获取 PR 数量
def get_pull_requests(url: str):
    url += "/pulls?state=all&per_page=1"
    while True:
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                links = response.headers.get('Link')
                if links:
                    match = re.search(r'page=(\d+)>; rel="last"', links)
                    return int(match.group(1)) if match else 1
                else:
                    return len(response.json())
            elif response.status_code in [403, 429]:
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
                wait_time = max(0, reset_time - int(time.time()))
                print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                time.sleep(wait_time + 5)
            else:
                print(f"请求失败，状态码: {response.status_code}")
                return 0
        except requests.exceptions.RequestException as e:
            print(f"网络请求异常: {e}")
            time.sleep(5)

def pull_requests(filename):
    os.makedirs("finished_data", exist_ok=True)
    if os.path.exists(f"results/{filename}"):
        print(f"Skipping {filename}")
        shutil.move(f"data/{filename}", f"finished_data/{filename}")
        return

    if filename.endswith(".json"):
        file_path = os.path.join("data", filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data:
                print(f"跳过空文件：{filename}")
                return

            url = list(data.keys())[0]
            pr_number = get_pull_requests(url)
            data["pull_requests"] = pr_number

            with open(os.path.join("results", filename), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            shutil.move(file_path, f"finished_data/{filename}")
            print(f"处理完成：{filename}, PR 数量：{pr_number}")

        except JSONDecodeError:
            print(f"JSON 格式错误，跳过文件：{filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错：{e}")

def process_files_parallel(filenames):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(pull_requests, filenames)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    filenames = [f for f in os.listdir("data/") if f.endswith(".json")]
    process_files_parallel(filenames)
