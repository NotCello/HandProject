import cv2, time, argparse, csv, os
import numpy as np
import psutil
cv2.setNumThreads(1)
def pct(arr, p):
    if not arr: return None
    return float(np.percentile(np.array(arr, dtype=np.float64), p))

def now_ms(): return time.time()*1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--save_csv", default="run_metrics.csv")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.device_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))  # helps 720p/30 on many UVC cams

    if not cap.isOpened():
        print("ERROR: cannot open webcam index", args.device_index)
        return

    read_ms, pre_ms, det_ms, post_ms, e2e_ms, inter_ms = [], [], [], [], [], []
    dropped = 0; frames = 0
    start = time.time(); last = None
    proc = psutil.Process(os.getpid())
    out = open(args.save_csv, "w", newline="")
    w = csv.writer(out)
    w.writerow(["t_sec","frame","read_ms","pre_ms","det_ms","post_ms","e2e_ms","interframe_ms","cpu%","rss_mb","ok"])

    while True:
        if (time.time()-start) > args.duration:
            break

        loop0 = now_ms()
        t0 = now_ms(); ok, frame = cap.read(); t1 = now_ms()
        r = t1 - t0
        if not ok or frame is None:
            dropped += 1
            time.sleep(0.002)
            cpu = psutil.cpu_percent(interval=None); rss = proc.memory_info().rss/(1024*1024)
            w.writerow([time.time()-start, frames, r, None, None, None, None, None, cpu, rss, 0])
            continue

        frames += 1
        tnow = time.time()
        if last is not None:
            inter_ms.append((tnow - last)*1000.0)
        last = tnow

        # Preprocess
        t2 = now_ms()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pre = cv2.GaussianBlur(gray, (5,5), 1.2)
        t3 = now_ms()

        # Detect (placeholder)
        t4 = now_ms()
        edges = cv2.Canny(pre, 80, 160)
        cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t5 = now_ms()

        # Postprocess
        t6 = now_ms()
        _ = [c for c in cnts if cv2.contourArea(c) > 100.0]
        t7 = now_ms()

        pre_t = t3 - t2
        det_t = t5 - t4
        post_t = t7 - t6
        e2e_t = now_ms() - loop0

        read_ms.append(r); pre_ms.append(pre_t); det_ms.append(det_t); post_ms.append(post_t); e2e_ms.append(e2e_t)

        cpu = psutil.cpu_percent(interval=None); rss = proc.memory_info().rss/(1024*1024)
        w.writerow([time.time()-start, frames, r, pre_t, det_t, post_t, e2e_t, inter_ms[-1] if inter_ms else None, cpu, rss, 1])

        if args.display and frames % 15 == 0:
            p50 = pct(e2e_ms,50); p95 = pct(e2e_ms,95); p99 = pct(e2e_ms,99)
            vis = frame.copy()
            cv2.putText(vis, f"FPS~{len(e2e_ms)/max(1e-6,(time.time()-start)):.1f} | e2e p50/p95/p99: {p50:.1f}/{p95:.1f}/{p99:.1f} ms | drops:{dropped}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("instrumented", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    if args.display: cv2.destroyAllWindows()
    out.close()

    def fmt(x): return "n/a" if x is None else f"{x:.1f}"
    print("\n=== RUN SUMMARY ===")
    print(f"Frames OK: {len(e2e_ms)} | Dropped: {dropped} | Duration: {time.time()-start:.1f}s | Avg FPS: {len(e2e_ms)/max(1e-6,(time.time()-start)):.2f}")
    print(f"Read   ms p50/p95/p99: {fmt(pct(read_ms,50))}/{fmt(pct(read_ms,95))}/{fmt(pct(read_ms,99))}")
    print(f"Pre    ms p50/p95/p99: {fmt(pct(pre_ms,50))}/{fmt(pct(pre_ms,95))}/{fmt(pct(pre_ms,99))}")
    print(f"Detect ms p50/p95/p99: {fmt(pct(det_ms,50))}/{fmt(pct(det_ms,95))}/{fmt(pct(det_ms,99))}")
    print(f"Post   ms p50/p95/p99: {fmt(pct(post_ms,50))}/{fmt(pct(post_ms,95))}/{fmt(pct(post_ms,99))}")
    print(f"E2E    ms p50/p95/p99: {fmt(pct(e2e_ms,50))}/{fmt(pct(e2e_ms,95))}/{fmt(pct(e2e_ms,99))}")
    print(f"Inter-frame ms p50/p95/p99: {fmt(pct(inter_ms,50))}/{fmt(pct(inter_ms,95))}/{fmt(pct(inter_ms,99))}")
    print(f"CSV saved: {os.path.abspath(args.save_csv)}\n")

if __name__ == "__main__":
    main()
