import cv2
import numpy as np
from PIL import Image
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
app = FaceAnalysis(det_name='retinaface_r50_v1', providers=providers)
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

def get_aligned_face_repo(image_path=None, rgb_pil_image=None, conf_thresh=0.5):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = cv2.cvtColor(np.array(rgb_pil_image), cv2.COLOR_RGB2BGR)
    if img is None: return None
    faces = app.get(img)
    if not faces: return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    if face.det_score < conf_thresh: return None
    aligned = face_align.norm_crop(img, landmark=face.kps, image_size=112)
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

def preprocess_bgr_112_from_aligned(aligned_pil, swap_color_channel, normalize=True, device='cpu'):
    img = np.array(aligned_pil.convert('RGB')).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5 if normalize else img
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return torch.from_numpy(img).to(device)

def batch_preprocess_images(batch_paths, swap_color_channel, device):
    imgs = []
    spaths = []
    for p in batch_paths:
        aligned = get_aligned_face_repo(image_path=p, conf_thresh=0.5)
        if aligned is None: continue
        inp = preprocess_bgr_112_from_aligned(aligned, swap_color_channel, device=device)
        imgs.append(inp)
        spaths.append(p)
    if not imgs:
        return None, None
    return torch.cat(imgs, dim=0), spaths



