import requests

# 可选：添加你的 Personal Access Token (PAT)
HEADERS = {
    "Authorization": "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"
}

# 发送 API 请求
response = requests.get("https://api.github.com/user", headers=HEADERS)

if response.status_code == 200:
    remaining = response.headers.get("X-RateLimit-Remaining", "Unknown")
    limit = response.headers.get("X-RateLimit-Limit", "Unknown")
    reset_time = response.headers.get("X-RateLimit-Reset", "Unknown")
    print(f"Remaining requests: {remaining}/{limit}")
    print(f"Rate limit resets at: {reset_time}")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
