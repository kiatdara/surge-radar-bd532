"""Latency benchmark of the running API (Section 2.3.3 / 3.6 scalability).
Run with the server up:  python -m src.benchmark --n 200
"""
import argparse, json, statistics, time
import requests

from . import config as C


def main(base="http://localhost:8000", n=200):
    pid = requests.get(f"{base}/api/v1/products/top?k=1").json()["items"][0]["product_id"]
    body = requests.get(f"{base}/api/v1/samples/{pid}").json()
    res = {}
    for name, explain in [("predict_with_explanation", True), ("predict_no_explanation", False)]:
        body["options"]["explain"] = explain
        rt, srv = [], []
        for _ in range(n):
            t0 = time.perf_counter()
            r = requests.post(f"{base}/api/v1/predict", json=body).json()
            rt.append((time.perf_counter() - t0) * 1e3); srv.append(r["latency_ms"]["total"])
        q = lambda a, p: sorted(a)[int(p * (len(a) - 1))]
        res[name] = {"n": n, "round_trip_p50_ms": round(statistics.median(rt), 1),
                     "round_trip_p95_ms": round(q(rt, .95), 1),
                     "server_p50_ms": round(statistics.median(srv), 1),
                     "server_p95_ms": round(q(srv, .95), 1)}
    t0 = time.perf_counter()
    for _ in range(20):
        requests.get(f"{base}/api/v1/products/top?k=10")
    res["top10_mean_ms"] = round((time.perf_counter() - t0) / 20 * 1e3, 1)
    print(json.dumps(res, indent=2))
    json.dump(res, open(C.REPORT_DIR / "latency.json", "w"), indent=2)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--n", type=int, default=200)
    a.add_argument("--base", default="http://localhost:8000"); x = a.parse_args()
    main(x.base, x.n)
