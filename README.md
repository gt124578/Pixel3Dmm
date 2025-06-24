# Pixel3DMM [Image mode] (Windows-compatible Fork & Demo)

<div style="text-align: center">
<img src="media/banner.gif" />
</div>

**This is a fork of the official [Orficial Pixel3DMM repository]((https://simongiebenhain.github.io/pixel3dmm/)) modified to run on Windows.**

It includes necessary code patches, a detailed installation guide, and a fixed dependency list.

### ✨ Live Demo on Hugging Face Spaces! ✨

Try Pixel3DMM directly in your browser without any installation:
**[➡️ Live Demo on Hugging Face Spaces by alexnasa](https://huggingface.co/spaces/alexnasa/pixel3dmm)**
(This github fork is based on this Hugging Face repository to create a local implementation)
<br>

---

*Original Project Links:*
[**Paper**](https://arxiv.org/abs/2505.00615) | [**Video**](https://www.youtube.com/watch?v=BwxwEXJwUDc) | [**Original Project Page**](https://simongiebenhain.github.io/pixel3dmm/)

*Authors of this project:*
[Simon Giebenhain](https://simongiebenhain.github.io/), 
[Tobias Kirschstein](https://niessnerlab.org/members/tobias_kirschstein/profile.html), 
[Martin Rünz](https://www.martinruenz.de/), 
[Lourdes Agaptio](https://scholar.google.com/citations?user=IRMX4-4AAAAJ&hl=en) and 
[Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html)  

---

## 1. Installation on Windows

This guide details the steps to set up the project on a local Windows machine.

### Prerequisites

1.  **Git:** Required to clone the repository. [Download here](https://git-scm.com/).
2.  **Conda:** Anaconda or Miniconda is used for environment management. [Download Miniconda here](https://docs.conda.io/en/latest/miniconda.html).
3.  **Microsoft C++ Build Tools:** Essential for compiling C++/Cython code.
    -   Download the installer from the [Visual Studio website](https://visualstudio.microsoft.com/fr/downloads/).
    -   Run the installer and in the "Workloads" tab, check the box for **"Desktop development with C++"**.
4.  **(Optional) NVIDIA GPU & Drivers:** ensure you have an NVIDIA GPU with up-to-date drivers.

### Step-by-Step Installation

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/gt124578/Pixel3Dmm.git
    cd Pixel3DMM
    ```

2.  **Download MICA and epoch59.pth (Manual Step due to Licensing)**
    
    The MICA component used in this project has a restrictive license that prevents its redistribution in this fork. You must download it manually.
    -   Go to the official MICA repository: [https://github.com/Zielon/MICA](https://github.com/Zielon/MICA).
    -   Download the project as a ZIP file.
    -   Unzip the archive.
    -   Copy the **entire content** of the `MICA-main` folder into the `src\pixel3dmm\preprocessing\MICA` folder of this project.
    -   Download epoch59.pth in this [Hugging Face Repo](https://huggingface.co/alexnasa/pixel3dmm/tree/main) and put this one into the `src\pixel3dmm\preprocessing\PIPNet\snapshots\WFLW\pip_32_16_60_r18_l2_l1_10_1_nb10` folder
    
    > By downloading and using MICA, you agree to its specific license terms.

3.  **Create and activate the Conda environment:**
    ```bash
    conda create -n p3dmm python=3.9 -y
    conda activate p3dmm

    pip --default-timeout=1000 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    conda install nvidia/label/cuda-11.8.0::cuda-nvcc nvidia/label/cuda-11.8.0::cuda-cccl nvidia/label/cuda-11.8.0::cuda-cudart nvidia/label/cuda-11.8.0::cuda-cudart-dev nvidia/label/cuda-11.8.0::libcusparse nvidia/label/cuda-11.8.0::libcusparse-dev nvidia/label/cuda-11.8.0::libcublas nvidia/label/cuda-11.8.0::libcublas-dev nvidia/label/cuda-11.8.0::libcurand nvidia/label/cuda-11.8.0::libcurand-dev nvidia/label/cuda-11.8.0::libcusolver nvidia/label/cuda-11.8.0::libcusolver-dev
    ```

    ```bash
    conda env config vars set TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
    conda deactivate 
    conda activate p3dmm
    ```

4.  **Install Python dependencies:**
    
    This fork provides a `requirements-windows.txt` file with tested, compatible versions of all packages.
    ```bash
    #Install PyTorch3D
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    
    #Install Nvdiffrast
    pip install git+https://github.com/NVlabs/nvdiffrast.git
    
    #Install Requirements 
    pip install -r requirements-windows.txt
    ```
    This will install specific versions of PyTorch (with CUDA support), `numpy==1.23.5`, `gradio==3.50.2`, and other critical libraries.

5.  **Install local packages:**
    
    Install Pixel3DMM and its sub-packages in "editable" mode.
    ```bash
    # Install the main Pixel3DMM package
    pip install --no-deps -e .

    # Install the 'facer' sub-package
    pip install --no-deps -e src/pixel3dmm/preprocessing/facer/
    ```

6.  **Compile Cython extensions:**
    
    This final step compiles a necessary part of the code for face detection.
    ```bash
    pip install Cython
    cd ./src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/utils/            
    python build.py build_ext --inplace
    cd ../../../../../../
    pip install -r requirements.txt
    pip install gradio==3.50.2 trimesh
    ```

You are now ready to run the application!

## 2. Running the Gradio Demo

This fork comes with a user-friendly web interface. To launch it, run:
```bash
python app.py
```
Open the local URL (e.g., `http://127.0.0.1:7860`) in your web browser, upload an image, and click "Start Reconstruction".

Several models will be downloaded if you need to download them manually here is the repo with all the necessary models: [Hugging Face Repo Model](https://huggingface.co/alexnasa/pixel3dmm/tree/main)


---

## 3. Citation and Acknowledgements

This project would not be possible without the foundational work of the original authors. If you use this code in your research, please cite the respective papers.

**Pixel3DMM**
```bibtex
@misc{giebenhain2025pixel3dmm,
    title={Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction},
    author={Simon Giebenhain and Tobias Kirschstein and  Martin R{\"{u}}nz and Lourdes Agapito and Matthias Nie{\ss}ner},
    year={2025},
    url={https://arxiv.org/abs/2505.00615},
}
```

**MICA**
```bibtex
@inproceedings{zielonka2023mica,
    title={MICA: A Malleable Model for Monocular 3D Human Face Reconstruction},
    author={Wojciech Zielonka and Timo Bolkart and Justus Thies},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={4169--4180},
    year={2023}
}
```

**FLAME**
```bibtex
@article{FLAME:SiggraphAsia2017,
    title = {Learning a model of facial shape and expression from {4D} scans},
    author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {36},
    number = {6},
    year = {2017},
    url = {https://doi.org/10.1145/3130800.3130813}
}
```

**PIPNet**
```bibtex
@inproceedings{jin2021pipnet,
  title={Pipnet: A pixel-in-pixel net for 68-point landmark detection},
  author={Jin, Haibo and Culture, Sheng and Li, Jun-Hai and Song, Dong-Gyu},
  booktitle={International Conference on Advanced Hybrid Information Processing},
  pages={25--36},
  year={2021},
  organization={Springer}
}
```



## 4. License Information and Attribution

Please respect their individual licenses and cite them in any resulting publications.

-   **Pixel3DMM:** Licensed under [CC BY-NC 4.0](https://github.com/SimonGiebenhain/pixel3dmm/blob/master/LICENSE).
    -   *Giebenhain, Simon, et al. "Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction." (2025).*
-   **MICA:** Licensed for [non-commercial scientific research](https://github.com/Zielon/MICA/tree/master?tab=License-1-ov-file).
    -   *Zielonka, Wojciech, et al. "MICA: A Malleable Model for Monocular 3D Human Face Reconstruction." (2023).*
-   **FLAME:** Licensed under [CC BY 4.0](https://flame.is.tue.mpg.de/modellicense.html).
    -   *Li, Tianye, et al. "Learning a model of facial shape and expression from 4D scans." (2017).*
-   **PIPNet:** Licensed under [Unknown](https://github.com/M-Nauta/PIPNet).
    -   *Jin, Haibo, et al. "PIPNet: A Pixel-in-Pixel Net for 68-Point Landmark Detection." (2021).*



This fork is provided under the same license as the original Pixel3DMM project: [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

Please be aware that dependencies such as MICA, FLAME, and PIPNet are subject to their own licenses. By using this software, you agree to comply with all of them.


