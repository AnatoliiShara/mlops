import subprocess, time, requests, json, pathlib, os

def _start_server():
    return subprocess.Popen(
        [
            "docker","run","--rm","-p8000:8000","-p8001:8001",
            "-v", f"{pathlib.Path(__file__).parent.parent}/models:/models",
            "nvcr.io/nvidia/tritonserver:23.10-py3",
            "tritonserver","--model-repository=/models"
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )

def _infer():
    url = "http://localhost:8000/v2/models/tinyllama/infer"
    payload = {"inputs":[{"name":"input__0","shape":[1],"datatype":"STRING",
                          "data":["Привіт, світе!"]}]}
    r = requests.post(url, json=payload, timeout=60)
    return "Привіт" in json.loads(r.text)["outputs"][0]["data"][0]

def test_batch():
    proc = _start_server()
    try:
        time.sleep(15)              # дати Triton завантажитись
        assert _infer()
    finally:
        proc.terminate()
