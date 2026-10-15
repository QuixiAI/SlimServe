"""N6 serving gate: start the video service over HTTP in this process, fire N concurrent requests, then
one more, and check: all complete, they ran one at a time, every mp4 probes correctly, and resident memory
is flat between requests. Usage: PYTHONPATH=<worktree> python n6_serve_check.py PIPELINE [N] [SIZE] [SECONDS]
Run through gpu_run.py (--need-gb 95)."""
import json, subprocess, sys, threading, time, urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from slimserve.video import server
from paths import SCRATCH
pipeline = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
size = sys.argv[3] if len(sys.argv) > 3 else "768x512"; seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 2
cfg = {"pipeline": pipeline, "width": 1536, "height": 1024, "num_frames": 121, "fps": 24.0, "max_video_tokens": 24576}
out = Path(SCRATCH) / "n6/videos"
svc = server.VideoService(cfg, "LTX-2.5", out)
httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(svc)); port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"
def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()
t0 = time.time()
while True:
    code, body = call("GET", "/health"); h = json.loads(body)
    if h["status"] != "loading": break
    time.sleep(1)
print(f"[n6] ready in {time.time()-t0:.1f} s: {h}", flush=True); assert h["status"] == "ok", h
resident = h["memory_gib"]["active"]
code, body = call("POST", "/v1/videos", {"prompt": "x", "size": "3072x2048"}); print("[n6] oversize request ->", code, json.loads(body)["error"]["message"][:90]); assert code == 400
prompts = ["A red fox trotting through a snowy pine forest at dawn, birds chirping", "Waves rolling onto a black sand beach at sunset, wind and surf",
           "A steam locomotive crossing a stone viaduct in autumn, whistle blowing", "Rain on a neon-lit street at night, distant traffic"]
results = {}
def client(i):
    t = time.time(); code, body = call("POST", "/v1/videos", {"prompt": prompts[i % 4], "size": size, "seconds": seconds, "seed": 100 + i, "wait": True})
    results[i] = (code, json.loads(body), time.time() - t)
threads = [threading.Thread(target=client, args=(i,)) for i in range(n)]
t0 = time.time(); [t.start() for t in threads]
time.sleep(5); print("[n6] mid-run health:", json.loads(call("GET", "/health")[1]), flush=True)
[t.join() for t in threads]; wall = time.time() - t0
ok = True
for i in sorted(results):
    code, job, dt = results[i]
    good = code == 200 and job["status"] == "completed"
    probe = ""
    if good:
        c, mp4 = call("GET", f"/v1/videos/{job['id']}/content"); p = out / f"check_{i}.mp4"; p.write_bytes(mp4)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,nb_frames,width,height", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.replace("\n", " ")
    ok &= good
    print(f"[n6] request {i}: http {code} {job.get('status')} waited {dt:.1f} s gen {job.get('seconds_elapsed')} s peak {job.get('timings', {}).get('peak_gib')} GiB | {probe} {job.get('error', '')}", flush=True)
h = json.loads(call("GET", "/health")[1]); print(f"[n6] {n} concurrent requests in {wall:.1f} s; health {h}", flush=True)
client(n); code, job, dt = results[n]; h2 = json.loads(call("GET", "/health")[1])
print(f"[n6] follow-up request: {job['status']} {job.get('seconds_elapsed')} s; health {h2}")
flat = abs(h2["memory_gib"]["active"] - h["memory_gib"]["active"]) < 1.0
print(f"[n6] resident at ready {resident} GiB, after burst {h['memory_gib']['active']}, after follow-up {h2['memory_gib']['active']} -> flat={flat}")
print("[n6] PASS" if ok and flat and job["status"] == "completed" and h2["failed"] == 0 else "[n6] FAIL")
svc.stop(); httpd.shutdown()
