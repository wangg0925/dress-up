from pathlib import Path
import sys
from PIL import Image
from mask import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
# print(PROJECT_ROOT)

import os
OUTPUT = os.path.abspath('output') 
new_dir = Path(OUTPUT)
if not new_dir.exists():
    try:
        new_dir.mkdir()
    except PermissionError:
        print(f"没有权限创建目录 {new_dir}。")
else:
    pass

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

from datetime import datetime

# model_type - "hd" or "dc"
# category - 0:upperbody; 1:lowerbody; 2:dress
category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

def clear_folder():
    if os.path.exists(OUTPUT):
        for item in os.listdir(OUTPUT):
            item_path = os.path.join(OUTPUT, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                except Exception as e:
                    print(f"删除文件 {item_path} 时出错: {e}")
    else:
        pass

def run(model_path, cloth_path, out_path, gpu_id=0, high_gpu_id=0, model_type="hd", category=0, image_scale=2.0, n_steps=20, n_samples=4, seed=-1):
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"start time: {formatted_now}")

    clear_folder()

    openpose_model = OpenPose(gpu_id)
    parsing_model = Parsing(gpu_id)

    if model_type == "hd":
        model = OOTDiffusionHD(high_gpu_id)
    elif model_type == "dc":
        model = OOTDiffusionDC(high_gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")
    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))

    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    # masked_vton_img.save('./output/mask_' + model_type + '.jpg')
    print('start to generate images...')
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )
    print ('images generated!')

    images[-1].save(OUTPUT + '/out_' + out_path + '.png')

    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"end time: {formatted_now}")