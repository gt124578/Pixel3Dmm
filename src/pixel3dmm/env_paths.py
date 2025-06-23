import os
import json
from pathlib import Path
from environs import Env
from huggingface_hub import hf_hub_download, snapshot_download

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/pixel3dmm/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("PIXEL3DMM_"):
    CODE_BASE = env("CODE_BASE")
    PREPROCESSED_DATA = env("PREPROCESSED_DATA")
    TRACKING_OUTPUT = env("TRACKING_OUTPUT")

base = snapshot_download(
            repo_id="alexnasa/pixel3dmm",     # your model repo
            repo_type="model",                # model vs dataset
        )

FLAME_ASSET       = os.path.join(base, "generic_model.pkl")
MICA_TAR_ASSET    = os.path.join(base, "mica.tar")
PIPNET_LOCAL_ASSET= os.path.join(base, "epoch59.pth")
CKPT_N_PRED       = os.path.join(base, "normals.ckpt")
CKPT_UV_PRED      = os.path.join(base, "uv.ckpt")
ANT_DIR           = os.path.join(base, "insightface")
BUFFALO_DIR       = os.path.join(base, "insightface")

head_template = f'{CODE_BASE}/assets/head_template.obj'
head_template_color = f'{CODE_BASE}/assets/head_template_color.obj'
head_template_ply = f'{CODE_BASE}/assets/test_rigid.ply'
VALID_VERTICES_WIDE_REGION = f'{CODE_BASE}/assets/uv_valid_verty_noEyes_debug.npy'
VALID_VERTS_UV_MESH = f'{CODE_BASE}/assets/uv_valid_verty.npy'
VERTEX_WEIGHT_MASK = f'{CODE_BASE}/assets/flame_vertex_weights.npy'
MIRROR_INDEX = f'{CODE_BASE}/assets/flame_mirror_index.npy'
EYE_MASK = f'{CODE_BASE}/assets/uv_mask_eyes.png'
FLAME_UV_COORDS = f'{CODE_BASE}/assets/flame_uv_coords.npy'
VALID_VERTS_NARROW = f'{CODE_BASE}/assets/uv_valid_verty_noEyes.npy'
VALID_VERTS = f'{CODE_BASE}/assets/uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy'
FLAME_MASK_ASSET = f'{CODE_BASE}/src/pixel3dmm/preprocessing/MICA/data/'

