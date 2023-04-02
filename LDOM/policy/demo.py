import os
import sys
import torch

sys.path.append('/root/fairseq/examples/MMPT')

from mmpt.models import MMPTModel


model, tokenizer, aligner = MMPTModel.from_pretrained(
    "/root/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")

model.eval()

def pooled_text(text):
    # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
    video_frames = torch.randn(1, 2, 30, 224, 224, 3)
    caps, cmasks = aligner._build_text_seq(tokenizer(text, add_special_tokens=False)["input_ids"])
    
    caps, cmasks = caps[None, :], cmasks[None, :]
    
    with torch.no_grad():
        output = model(video_frames, caps, cmasks, return_score=False)
    return output['pooled_text']

# vim /root/fairseq/examples/MMPT/videoclip.py
# vim /root/fairseq/examples/MMPT/mmpt/models/mmfusion.py 42
# '/root/fairseq/examples/MMPT/runs/retri/videoclip/checkpoint_best.pt'
# vim /root/fairseq/examples/MMPT/mmpt/processors/models/s3dg.py 205
# /root/fairseq/examples/MMPT/pretrained_models/s3d_dict.npy
# vim /root/fairseq/examples/MMPT/mmpt/models/mmfusion.py 47
# /root/fairseq/examples/MMPT/pretrained_models/s3d_howto100m.pth