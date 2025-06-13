import requests, concurrent.futures, json

URL = "http://localhost:8000/v2/models/tinyllama/infer"
Q   = "Що таке машинне навчання?"

def infer(q):
    payload = {
        "inputs": [
            {"name":"input__0","shape":[1],"datatype":"STRING","data":[q]}
        ]
    }
    r = requests.post(URL, json=payload, timeout=30)
    return json.loads(r.text)["outputs"][0]["data"][0]

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        fut = [ex.submit(infer, Q) for _ in range(20)]
        for f in fut:
            print(f.result()[:120], "…")
