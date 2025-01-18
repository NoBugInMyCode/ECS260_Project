import json, time
import multiprocessing
from time import sleep

import requests

GITHUB_TOKEN = "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"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

def get_repo_info(url, subfield=""):
    url += subfield
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403 or response.status_code == 429:  # Rate limit exceeded
        reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
        current_time = int(time.time())
        wait_time = reset_time - current_time

        if wait_time > 0:
            print(f"Rate limit reached. Waiting for {wait_time} seconds...")
            time.sleep(wait_time + 5)

        # 重试请求
        return get_repo_info(url, subfield=subfield)
    else:
        print(f"Failed to fetch data: {response.status_code}")


def data_getter(file_name: str):
    with open(file_name, "r") as f:
        # 遍历文件中的每一行，直到文件结束
        for line in f:
            try:
                # Remove \n at the end
                url = line.strip()
                print(f"Processing: {url}")

                # Get general info (forks, watchers, stars, etc.)
                general_info = get_repo_info(url)
                forks = general_info["forks"]
                watchers = general_info["watchers"]
                description = general_info["description"]
                subscribers = general_info["subscribers_count"]
                full_name = general_info["full_name"]
                full_name = full_name.replace("/", "-")
                stars = general_info["stargazers_count"]
                creation_date = general_info["created_at"]
                topics = general_info["topics"]
                # Get what language it uses
                languages = get_repo_info(url, "/languages")

                # Get most recent 30 commits of this repo
                commits = []
                commits_info = get_repo_info(url, "/commits")
                for commit in commits_info:
                    commits.append(commit["commit"]["author"]["date"])

                # Get contributors
                contributors = get_repo_info(url, "/contributors")
                with open(f"result/{full_name}.json", "w") as json_file:
                    json.dump({
                        url: {
                            "forks": forks,
                            "watchers": watchers,
                            "stars": stars,
                            "languages": languages,
                            "commits": commits,
                            "creation_date": creation_date,
                            "contributors": len(contributors),
                            "topics": topics,
                            "subscribers": subscribers
                        }
                    }, json_file, indent=4)
                    print(f"Finish Writing {full_name}.json")
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue



def process_file_parallel(file_list):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(data_getter, file_list)
    return



if __name__ == "__main__":
    file_list = []
    for i in range(1, 10):
        file_list.append(f"jobs/{i}.txt")
    process_file_parallel(file_list)

