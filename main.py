import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import config
from backbones import get_model
from convert import batch_preprocess_images
from data import get_inference_dataloader
from db_builder import build_employee_db
from video_processor import process_video

def main():
    args = config.get_args()
    device = torch.device(args.device)
    
    model = get_model(args.arch)
    model.to(device)
    model.eval()
    
    ckpt_path = args.checkpoint_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd: sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    print(f"Loaded model {args.arch} from {ckpt_path} on {device}")
    
    if args.build_db:
        build_employee_db(args, model, device)
    elif args.video_path:
        process_video(args, model, device)
    else:
        # Image inference (keep as-is)
        dl = get_inference_dataloader(args.input_dir, args.batch_size, args.num_workers)
        print(f"Found {len(dl.dataset)} images in {args.input_dir}")
        
        st = time.time()
        all_e = []
        all_p = []
        for bp in dl:
            inp, sp = batch_preprocess_images(bp, args.swap_color_channel, device)
            if inp is None: continue
            with torch.no_grad():
                o = model(inp)
                e = F.normalize(o, dim=1).cpu().numpy()
            all_e.extend(e)
            all_p.extend(sp)
        
        el = time.time() - st
        print(f"Inference done. Time: {el:.2f}s for {len(all_p)} images")
        
        for e, p in zip(all_e, all_p):
            fn = os.path.splitext(os.path.basename(p))[0] + '.npy'
            np.save(os.path.join(args.output_dir, fn), e)
        print(f"Saved {len(all_p)} embs to {args.output_dir}")

if __name__ == '__main__':
    main()