# Time one patch embedding and one transformer block on the GPU, float16 autocast against float32.
# Weights are untrained: timing does not depend on them. Output: outputs/results/gpu_precision_check.csv
import time, torch, timm, pandas as pd
m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).cuda().eval()
def run(fn, n=40):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return sorted(ts)[n // 2] * 1000
rows = []
with torch.no_grad():
    for ac in (True, False):
        for b in (1, 2, 4, 8, 16, 32, 64, 128):
            x, t = torch.randn(b, 3, 224, 224).cuda(), torch.randn(b, 197, 192).cuda()
            with torch.autocast("cuda", enabled=ac):
                conv, blk = run(lambda: m.patch_embed(x)), run(lambda: m.blocks[0](t))
            rows.append({"precision": "float16 autocast" if ac else "float32", "batch": b, "embed_ms": conv, "block_ms": blk, "embed_ms_per_image": conv / b, "block_ms_per_image": blk / b})
d = pd.DataFrame(rows).round(3); d.to_csv("outputs/results/gpu_precision_check.csv", index=False); print(d.to_string())
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
