"""Developer utility to inspect public dataset metadata without remote code execution."""
import json
import urllib.request

for name in ["rajpurkar/squad", "qwant/squad_fr", "lincoln/newsquadfr", "hsseinmz/arcd"]:
    try:
        with urllib.request.urlopen("https://huggingface.co/api/datasets/" + name, timeout=30) as response:
            data = json.load(response)
        print(name, json.dumps({key: data.get(key) for key in ["sha", "cardData", "siblings"]}, ensure_ascii=False)[:18000], flush=True)
    except Exception as exc:
        print(name, str(exc), flush=True)
