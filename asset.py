import os
PROJECT_ROOT = os.path.abspath('../input/diffusion/pytorch/models/1/') 
# print(PROJECT_ROOT)
annotator_ckpts_path = os.path.join(PROJECT_ROOT, 'openpose/ckpts')
# body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
# hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
# face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

parsing_atr_model = os.path.join(PROJECT_ROOT, 'humanparsing/parsing_atr.onnx')
parsing_lip_model = os.path.join(PROJECT_ROOT, 'humanparsing/parsing_lip.onnx')

VIT_PATH = os.path.join(PROJECT_ROOT, 'clip-vit-large-patch14')
VAE_PATH = os.path.join(PROJECT_ROOT, 'ootd')
UNET_PATH = os.path.join(PROJECT_ROOT, 'ootd/ootd_hd/checkpoint-36000')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ootd')