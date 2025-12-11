import os
import numpy as np
import json
import torch
import torch.nn.functional as F
from convert import get_aligned_face_repo, preprocess_bgr_112_from_aligned

def build_employee_db(args, model, device):
    if not args.employees_dir or not os.path.isdir(args.employees_dir):
        raise ValueError(f"Employees directory not found: {args.employees_dir}")
    
    db = {}
    for name in os.listdir(args.employees_dir):
        dir_ = os.path.join(args.employees_dir, name)
        if not os.path.isdir(dir_): continue
        paths = [os.path.join(dir_, f) for f in os.listdir(dir_) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not paths:
            db[name] = []
            continue
        
        embs = []
        for p in paths:
            aligned = get_aligned_face_repo(image_path=p, conf_thresh=args.retina_conf)
            if aligned is None: continue
            inp = preprocess_bgr_112_from_aligned(aligned, args.swap_color_channel, device=device)
            with torch.no_grad():
                out = model(inp)
                emb = F.normalize(out, dim=1).cpu().numpy().flatten()
            embs.append(emb)
        
        if embs:
            avg = np.mean(embs, axis=0).tolist()
            db[name] = avg
        
    with open(args.db_path, 'w') as f:
        json.dump(db, f)