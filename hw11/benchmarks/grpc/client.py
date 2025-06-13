import asyncio, random, json, time, statistics, grpc
import reco_pb2, reco_pb2_grpc      # згенеровані модулі

PROMPTS = [
    "Роман про кохання і війну",
    "Юридичний трилер українською",
    "Наукова фантастика про подорожі у часі",
]

N_REQUESTS = 3000
CONCURRENCY = 50
ADDRESS = "localhost:9001"           # kServe-gRPC

async def worker(stub, latencies):
    prompt = random.choice(PROMPTS)
    req = reco_pb2.PredictRequest(text=prompt, top_k=5)
    t0 = time.time()
    _ = await stub.Recommend(req)
    latencies.append((time.time() - t0) * 1000)

async def main():
    async with grpc.aio.insecure_channel(ADDRESS) as channel:
        stub = reco_pb2_grpc.RecoStub(channel)
        lat = []
        sem = asyncio.Semaphore(CONCURRENCY)

        async def sem_worker():
            async with sem:
                await worker(stub, lat)

        await asyncio.gather(*[sem_worker() for _ in range(N_REQUESTS)])

    print(json.dumps(
        {
            "avg": sum(lat) / len(lat),
            "p95": statistics.quantiles(lat, n=100)[94],
            "n": len(lat),
        }
    ))

if __name__ == "__main__":
    asyncio.run(main())
