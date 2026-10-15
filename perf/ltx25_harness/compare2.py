import sys, numpy as np
ref = np.load(sys.argv[1]); 
for c in sys.argv[2:]:
    d = np.load(c)
    for kind in ("video", "audio"):
        a = ref[kind].astype(np.float64).ravel(); b = d[kind].astype(np.float64).ravel()
        print(f"{c:40s} {kind:5s} cos={a@b/(np.linalg.norm(a)*np.linalg.norm(b)):.6f} rel_l2={np.linalg.norm(a-b)/np.linalg.norm(a):.5f}")
