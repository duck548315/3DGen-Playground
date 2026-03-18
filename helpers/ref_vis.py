import torch
import numpy as np
from matplotlib import pyplot as plt
# Enable interactive mode
# %matplotlib ipympl
# %matplotlib inline
import shutil
import tarfile
import json
import tempfile
import sys
import os
from datetime import datetime, timezone
import math
import gc
# sys.path.append("../")
sys.path.append("submodules/gaussian-splatting")
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from hydra import initialize, compose
import random
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf
from source.trainer import EDGSTrainer
from gradio_demo import process_input

from source.utils_preprocess import read_video_frames, preprocess_frames, select_optimal_frames, save_frames_to_scene_dir, run_colmap_on_scene
from source.visualization import generate_circular_camera_path, save_numpy_frames_as_mp4, generate_fully_smooth_cameras_with_tsp, put_text_on_image


with initialize(config_path="configs", version_base="1.1"):
    cfg = compose(config_name="train")
# print(OmegaConf.to_yaml(cfg))

"""# 3. Init input parameters

## 3.1 Optionally preprocess video
"""

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def preprocess_input(input_path, num_ref_views, obj_name, max_size=512):
    tmpdirname = os.path.join('/vision/group/occnerf/gaussianatlas/gaussianverse/colmap/', obj_name)
    scene_dir = tmpdirname # os.path.join(tmpdirname, "scene")

    os.makedirs(scene_dir, exist_ok=True)

    #### load frames 
    with tarfile.open(input_path, 'r:gz') as tar:
        # List all members
        names = []
        for member in tar.getmembers():
            if '.png' in member.name:
                names.append(int(member.name.split('/')[-2]))


        max_length = max(names)
        frames = []
        Ks, Rs, ts = [], [], []
        for frame_idx in range(max_length):
            

            # Read a PNG file
            png_member = tar.getmember('campos_512_v1/%05d/%05d.png' % (frame_idx, frame_idx))
            png_data = tar.extractfile(png_member).read()  # bytes
            # You can now use PIL.Image to load it
            image = Image.open(BytesIO(png_data))
            frames.append(np.array(image)[:,:,:4]) # rgb a!!

            json_member = tar.getmember('campos_512_v1/%05d/%05d.json' % (frame_idx, frame_idx))
            json_data = tar.extractfile(json_member).read().decode('utf-8')
            meta = json.loads(json_data)

            c2w = np.eye(4)
            c2w[:3, 0] = np.array(meta['x'])
            c2w[:3, 1] = np.array(meta['y'])
            c2w[:3, 2] = np.array(meta['z'])
            c2w[:3, 3] = np.array(meta['origin'])
            #print(meta.keys())
            fovx = meta['x_fov']
            fovy = meta['y_fov']

            focal = fov2focal(fovx, 512)

            K = np.eye(3)
            K[0,0] = K[1,1] = focal
            K[0,2] = K[1,2] = 256
            Ks.append(K)


            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)


            R = w2c[:3,:3]  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            Rs.append(R)
            ts.append(T)

    save_frames_to_scene_dir(frames, scene_dir)
    selected_frames = process_input(input_path, num_ref_views, scene_dir, frames, max_size)
    try:
        run_colmap_on_scene(scene_dir, Ks, Rs, ts)
    except Exception as e:
        print(e)
        return None, None

    return selected_frames, scene_dir


def cleanup_memory():
    """Clean up memory after each iteration to prevent OOM"""
    # Clear matplotlib figures
    plt.close('all')
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()


with open('todo.txt', 'r') as f:
    obj_names = f.readlines()


chunk = 10000
fitting_idx = int(sys.argv[1])

print('fitting idx', fitting_idx)


obj_names = obj_names[fitting_idx * chunk : (fitting_idx + 1) * chunk]

############### DEBUG ###############
# obj_names = ['1918/9586761.tar.gz']

print('TOTAL TODO:', len(obj_names))

save_path = "/vision/group/occnerf/gaussianatlas/gaussianverse/edgs/"
data_path = '/vision/group/occnerf/gaussianatlas/gaussianverse/gobjaverse'
for obj_idx, obj_name in enumerate(obj_names):
    print('############# PROCESSING ############', obj_idx, chunk)
    
    # Periodic memory cleanup every 100 iterations
    if obj_idx > 0 and obj_idx % 100 == 0:
        print(f"Performing periodic memory cleanup at iteration {obj_idx}")
        cleanup_memory()
    obj_name = obj_name.strip()
    PATH_TO_VIDEO = os.path.join(data_path, obj_name) #
    num_ref_views = 16 # how many frames you want to extract from video and colmap
    if not os.path.exists(os.path.join('/vision/group/occnerf/gaussianatlas/gaussianverse/colmap/', obj_name[:-7], 'sparse', '0', 'points3D.bin')):
        print('COLMAP NOT FITTED', obj_name[:-7])
        continue
        
        # images, scene_dir = preprocess_input(PATH_TO_VIDEO, num_ref_views, obj_name[:-7])
        # if images is None:
        #     print('ERROR', obj_name[:-7])
        #     continue

    # Update the config with your settings
    model_path = os.path.join(save_path, obj_name[:-7])

    if os.path.exists(os.path.join(model_path, 'point_cloud.ply')):

        ####check date
        mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(model_path, 'point_cloud.ply')), tz=timezone.utc)
        year = datetime.now(tz=timezone.utc).year
        july_1 = datetime(year, 7, 1, tzinfo=timezone.utc)
        #print(mtime, july_1, mtime >= july_1)
        #continue
        if mtime >= july_1:
            print(obj_name[:-7], 'fitted! SKIP!!!!!')
            continue

    os.makedirs(model_path, exist_ok=True)

    source_path = os.path.join("/vision/group/occnerf/gaussianatlas/gaussianverse/colmap" , obj_name[:-7])
    # os.makedirs(source_path, exist_ok=True)

    cfg.wandb.name="EDGS.demo.scene"
    cfg.wandb.mode="disabled" # "online"
    cfg.gs.dataset.model_path=model_path #"/vision/group/occnerf/gaussianatlas/gaussianverse/edgs/"  # "change this to your path to the processed scene"
    cfg.gs.dataset.source_path=source_path# "change this to your path"
    # Optionally for video processed
    # cfg.gs.dataset.source_path="../assets/video_colmaped/"
    cfg.gs.dataset.images="images"
    cfg.gs.opt.TEST_CAM_IDX_TO_LOG=12
    cfg.train.gs_epochs=30000
    cfg.gs.opt.opacity_reset_interval=1_000_000
    cfg.gs.opt.densify_until_iter=10_000
    cfg.train.no_densify=False
    cfg.init_wC.matches_per_ref=15_000
    cfg.init_wC.nns_per_ref=3
    cfg.init_wC.num_refs=25 # half of the frames # 180
    cfg.init_wC.roma_model="outdoors"
    cfg.init_wC.MAX_POINTS = 128*128


    """# 4. Initilize model and logger"""

    trainer = None
    gs = None
    try:
        # _ = wandb.init(entity=cfg.wandb.entity,
        #                 project=cfg.wandb.project,
        #                 config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        #                 name = cfg.wandb.name,
        #                 mode = cfg.wandb.mode)
        omegaconf.OmegaConf.resolve(cfg)
        set_seed(cfg.seed)
        # Init output folder
        print("Output folder: {}".format(cfg.gs.dataset.model_path))
        os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
        # Init gs model
        gs = hydra.utils.instantiate(cfg.gs)
    except:
        print(obj_name[:-7], 'COLMAP ERROR! SKIP!!!!!')
        if trainer is not None:
            trainer.cleanup()
            del trainer
        if gs is not None:
            del gs
        cleanup_memory()
        continue
    
    trainer = EDGSTrainer(GS=gs,
                        training_config=cfg.gs.opt,
                        device=cfg.device,
                        log_wandb=False)

    """# 5. Init with matchings"""

    trainer.timer.start()
    trainer.init_with_corr(cfg.init_wC)
    trainer.timer.pause()


    """# 6.Optimize scene
    Optimize first briefly for 5k steps and visualize results. We also disable saving of pretrained models. Train function can be changed for any other method
    """

    trainer.saving_iterations = []

    cfg.train.gs_epochs=10_000
    try:
        trainer.train(cfg.train)
    except Exception as e:
        print(e)
        print(obj_name[:-7], 'FITTING ERROR! SKIP!!!!!')
        # Clean up trainer and model before continuing
        if trainer is not None:
            trainer.cleanup()
            del trainer
        if gs is not None:
            del gs
        cleanup_memory()
        continue

    """### Visualize same viewpoints"""
    vis_path = '/vision/group/occnerf/gaussianatlas/gaussianverse/gobjaverse_render'
    os.makedirs(os.path.join(vis_path, obj_name[:-7]), exist_ok=True)
    viewpoint_cams_to_viz = random.sample(trainer.GS.scene.getTrainCameras(), 4)
    with torch.no_grad():
        for cam_idx, viewpoint_cam in enumerate(viewpoint_cams_to_viz):
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]

            image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
            image_gt_np = viewpoint_cam.original_image.clone().detach().cpu().numpy().transpose(1, 2, 0)

            # Clip values to be in the range [0, 1]
            image_np = np.clip(image_np*255, 0, 255).astype(np.uint8)
            image_gt_np = np.clip(image_gt_np*255, 0, 255).astype(np.uint8)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
            ax[0].imshow(image_gt_np)
            ax[0].axis("off")
            ax[1].imshow(image_np)
            ax[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path, obj_name[:-7], 'debug_optimzed_%d.png' % cam_idx))
            plt.close(fig)  # Close figure immediately after saving
            
            # Clear variables to free memory
            del render_pkg, image, image_np, image_gt_np, fig, ax

    """### Save model"""

    with torch.no_grad():
        trainer.save_model()

    """# 7. Continue training until we reach total 30K training steps"""


    # cfg.train.gs_epochs=25_000
    # trainer.train(cfg.train)

    # """### Save model"""

    # with torch.no_grad():
    #     trainer.save_model()
    
    # Clean up memory after each iteration
    if trainer is not None:
        trainer.cleanup()
        del trainer
    if gs is not None:
        del gs
    cleanup_memory()