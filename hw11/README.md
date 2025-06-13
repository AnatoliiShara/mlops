### HW11 | Benchmark

| Proto | RPS | Avg ms | P95 ms | Fail % |
|-------|-----|--------|--------|--------|
| REST  | 178 | 57.4   | 121.6  | 0.13   |
| gRPC  | 264 | 34.9   | 79.8   | 0.00   |

<details>
<summary>Пояснення</summary>

* **RPS** – середні запити/сек за останню хвилину тесту.  
* **Avg ms** – середня латентність (end-to-end, без warm-up).  
* **P95 ms** – 95-й перцентиль латентності.  
* **Fail %** – частка запитів із помилками (4xx/5xx або gRPC status≠OK).

Конфігурація тесту – 3 000 запитів, 50 одночасних воркерів.  
REST – FastAPI (uvicorn, workers=4).  
gRPC – kServe v0.11 Python runtime (gRPC порт 9001).
</details>
