# app.py
# Modified version of the original script for local execution (Original: https://huggingface.co/spaces/alexnasa/pixel3dmm/blob/main/app.py)
# Make sure you have installed all dependencies in your Conda environment beforehand.

import os
import subprocess
import tempfile
import uuid
import glob
import shutil
import gradio as gr
from PIL import Image

# --- STEP 1: Environment Setup and Imports ---

# Disable Dynamo, as it can cause issues with some PyTorch versions
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
torch._dynamo.disable()


# Define environment variables required by the Pixel3DMM project.
# These point to sub-folders within the project.
# We create them to ensure they exist.
print("Setting up environment variables...")
os.environ["PIXEL3DMM_CODE_BASE"] = os.getcwd()
os.environ["PIXEL3DMM_PREPROCESSED_DATA"] = os.path.join(os.getcwd(), "preprocessed_data")
os.environ["PIXEL3DMM_TRACKING_OUTPUT"] = os.path.join(os.getcwd(), "tracking_output")
os.makedirs(os.environ["PIXEL3DMM_PREPROCESSED_DATA"], exist_ok=True)
os.makedirs(os.environ["PIXEL3DMM_TRACKING_OUTPUT"], exist_ok=True)


# Import project modules. These should work if the installations
# `pip install -e .` and `pip install -e src/pixel3dmm/preprocessing/facer` were successful.
print("Importing project modules...")
from pixel3dmm import env_paths
from omegaconf import OmegaConf
from pixel3dmm.network_inference import normals_n_uvs
from pixel3dmm.run_facer_segmentation import segment
print("Imports successful.")


# --- STEP 2: Main Function Definitions ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Cache for heavy models to avoid reloading them on every click
_model_cache = {}

# Utility functions to find files
def first_file_from_dir(directory, ext):
    files = glob.glob(os.path.join(directory, f"*.{ext}"))
    return sorted(files)[0] if files else None

def first_image_from_dir(directory):
    patterns = ["*.jpg", "*.png", "*.jpeg"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(directory, p)))
    return sorted(files)[0] if files else None


# Pipeline Step 1: Image Preprocessing
def preprocess_image(image_array, session_id):
    print(f"[{session_id}] Step 1: Preprocessing...")
    if image_array is None:
        return "❌ Please choose an image first.", None
    
    # Create a unique directory for this session
    base_dir = os.path.join(os.environ["PIXEL3DMM_PREPROCESSED_DATA"], session_id)
    os.makedirs(base_dir, exist_ok=True)
    saved_image_path = os.path.join(base_dir, f"{session_id}.png")

    Image.fromarray(image_array).save(saved_image_path)

    # Load 'facer' models if not already cached
    if "face_detector" not in _model_cache:
        import facer
        print("Loading facer models (detector and parser)...")
        _model_cache['face_detector'] = facer.face_detector('retinaface/mobilenet', device=DEVICE)
        _model_cache['face_parser'] = facer.face_parser('farl/celebm/448', device=DEVICE)
        print("Facer models loaded.")
            
    # Run project scripts (replaces `sh` calls)
    # Note: `capture_output=True` can be useful for debugging if the script fails
    print("Running scripts/run_preprocessing.py...")
    subprocess.run(
        ["python", "scripts/run_preprocessing.py", "--video_or_images_path", saved_image_path],
        check=True
    )
    print("Running segmentation...")
    segment(f'{session_id}', _model_cache['face_detector'], _model_cache['face_parser'])
    
    crop_dir = os.path.join(base_dir, "cropped")
    image = first_image_from_dir(crop_dir)
    print("Preprocessing finished.")
    return "✅ Step 1 complete. Ready for Normals.", image


# Step 2: Normals Inference
def step2_normals(session_id):
    print(f"[{session_id}] Step 2: Computing normals...")
    from pixel3dmm.lightning.p3dmm_system import system as p3dmm_system
    
    # Load normals model if not in cache
    if "normals_model" not in _model_cache:
        print("Loading normals prediction model...")
        model = p3dmm_system.load_from_checkpoint(f"{env_paths.CKPT_N_PRED}", strict=False)
        _model_cache["normals_model"] = model.eval().to(DEVICE)
        print("Normals model loaded.")
    
    base_conf = OmegaConf.load("configs/base.yaml")
    base_conf.video_name = f'{session_id}'
    normals_n_uvs(base_conf, _model_cache["normals_model"])

    normals_dir = os.path.join(os.environ["PIXEL3DMM_PREPROCESSED_DATA"], session_id, "p3dmm", "normals")
    image = first_image_from_dir(normals_dir)
    print("Normals computation finished.")
    return "✅ Step 2 complete. Ready for UV Map.", image


# Step 3: UV Map Inference
def step3_uv_map(session_id):
    print(f"[{session_id}] Step 3: Computing UV map...")
    from pixel3dmm.lightning.p3dmm_system import system as p3dmm_system

    if "uv_model" not in _model_cache:
        print("Loading UV prediction model...")
        model = p3dmm_system.load_from_checkpoint(f"{env_paths.CKPT_UV_PRED}", strict=False)
        _model_cache["uv_model"] = model.eval().to(DEVICE)
        print("UV model loaded.")

    base_conf = OmegaConf.load("configs/base.yaml")
    base_conf.video_name = f'{session_id}'
    base_conf.model.prediction_type = "uv_map"
    normals_n_uvs(base_conf, _model_cache["uv_model"])

    uv_dir = os.path.join(os.environ["PIXEL3DMM_PREPROCESSED_DATA"], session_id, "p3dmm", "uv_map")
    image = first_image_from_dir(uv_dir)
    print("UV map computation finished.")
    return "✅ Step 3 complete. Ready for Tracking.", image


# Step 4: Tracking
def step4_track(session_id):
    print(f"[{session_id}] Step 4: Starting Tracking...")
    from pixel3dmm.tracking.tracker import Tracker
    from pixel3dmm.tracking.flame.FLAME import FLAME
    from pixel3dmm.tracking.renderer_nvdiffrast import NVDRenderer
    from pytorch3d.io import load_obj

    tracking_conf = OmegaConf.load("configs/tracking.yaml")
    tracking_conf.video_name = f'{session_id}' # Assign video/session name

    if "flame_model" not in _model_cache:
        print("Loading FLAME model for tracking...")
        flame = FLAME(tracking_conf).to(DEVICE)
        _model_cache["flame_model"] = flame
        
        mesh_file = env_paths.head_template
        _model_cache["diff_renderer"] = NVDRenderer(
            image_size=tracking_conf.size, 
            obj_filename=mesh_file,
            no_sh=False,
            white_bg=True
        ).to(DEVICE)
        print("FLAME model loaded.")
   
    tracker = Tracker(tracking_conf, _model_cache["flame_model"], _model_cache["diff_renderer"])
    tracker.run()

    tracking_dir = os.path.join(os.environ["PIXEL3DMM_TRACKING_OUTPUT"], session_id, "frames")
    image = first_image_from_dir(tracking_dir)
    print("Tracking finished.")
    return "✅ Step 4 complete. Pipeline finished!", image


# Main function that chains all steps together
def generate_results_and_mesh(image, session_id=None):
    if image is None:
        return "❌ Please choose an image first.", None, None, None, None, None

    if session_id is None:
        session_id = uuid.uuid4().hex
         
    status1, crop_img = preprocess_image(image, session_id)
    if "❌" in status1:
      return status1, None, None, None, None, None
      
    status2, normals_img = step2_normals(session_id)
    status3, uv_img = step3_uv_map(session_id)
    status4, track_img = step4_track(session_id)
    
    # Locate the 3D mesh file (.glb is the modern format)
    mesh_dir = os.path.join(os.environ["PIXEL3DMM_TRACKING_OUTPUT"], session_id, "mesh")
    mesh_file = first_file_from_dir(mesh_dir, "glb")

    final_status = "\n".join([status1, status2, status3, status4])
    return final_status, crop_img, normals_img, uv_img, track_img, mesh_file


# Function to start a new session with a unique ID
def start_session(request: gr.Request):
    return uuid.uuid4().hex


# --- STEP 3: Create the Gradio Interface ---

print("Creating Gradio interface...")
with gr.Blocks(css="#col-container {max-width: 1024px; margin: 0 auto;}") as demo:
    
    session_state = gr.State()
    demo.load(start_session, outputs=[session_state])

    gr.HTML("""
        <div style="text-align: center;">
            <h1>Pixel3DMM - 3D Face Reconstruction (Local)</h1>
            <p>Upload an image to generate a 3D model of the face.</p>
        </div>""")

    with gr.Row(elem_id="col-container"):
        with gr.Column(scale=1):
            image_in = gr.Image(label="Input Image", type="numpy", height=512)
            run_btn = gr.Button("Start Reconstruction", variant="primary")
            status = gr.Textbox(label="Status", lines=5, interactive=False, value="Ready to start.")
        
        with gr.Column(scale=2):
            mesh_file = gr.Model3D(label="3D Model Preview", height=512)
            with gr.Row():
                crop_img = gr.Image(label="1. Preprocessing", height=128)
                normals_img = gr.Image(label="2. Normals", height=128)
                uv_img = gr.Image(label="3. UV Map", height=128)
                track_img = gr.Image(label="4. Tracking", height=128)

    # Example images
    gr.Examples(
        examples=[
            os.path.join("example_images", "jennifer_lawrence.png"),
            os.path.join("example_images", "brendan_fraser.png"),
            os.path.join("example_images", "jim_carrey.png"),
        ],
        inputs=[image_in]
    )

    # Link the button to the main function
    run_btn.click(
        fn=generate_results_and_mesh,
        inputs=[image_in, session_state],
        outputs=[status, crop_img, normals_img, uv_img, track_img, mesh_file]
    )

# --- STEP 4: Launch the local web server ---

print("Launching Gradio server...")
# `share=True` creates a temporary public link. Remove it if you only want local access.
demo.queue().launch(share=True)
