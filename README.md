### Hướng dẫn chạy thử
### 1. Hướng dẫn cài checkpoint
mkdir -p ./checkpoints
gdown --id <ID_từ_HF> -O ./checkpoints/LVFace-B_Glint360K.pt # Điều chỉnh version cho phù hợp
# or
# wget https://huggingface.co/bytedance/LVFace/resolve/main/LVFace-B_Glint360K.pt -O .#     checkpoints/LVFace-B_Glint360K.pt

### 1. Trên ảnh tĩnh
python main.py --arch vit_b \
    --checkpoint_path ./checkpoints/LVFace-B_Glint360K.pt \
    --input_dir ./input_images \
    --output_dir ./results/ \
    --batch_size 32
### 2. Xây dựng Employee DB
python main.py --build_db \
    --arch vit_b \
    --checkpoint_path ./checkpoints/LVFace-B_Glint360K.pt \
    --employees_dir ./employees \
    --db_path ./employee_db.json \
    --device cuda \
    --retina_conf 0.5
### 3. Chạy trên Video
python main.py --video_path ./input_video.mp4 \
    --arch vit_b \
    --checkpoint_path ./checkpoints/LVFace-B_Glint360K.pt \
    --db_path ./employee_db.json \
    --output_video_path ./output_video.mp4 \
    --device cuda \
    --conf_threshold 0.5 \
    --cos_threshold 0.6 \
    --reid_threshold 0.8 \
    --skip_interval 5 \
    --retina_conf 0.5