import subprocess, sys, os
import tyro
from pixel3dmm import env_paths

def run_and_check(cmd, cwd=None):
    print(f"> {' '.join(cmd)}  (in {cwd or os.getcwd()})")
    # stream logs live
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
    )
    return result

def main(video_or_images_path: str):
    vid_name = (
        os.path.basename(video_or_images_path)
        if os.path.isdir(video_or_images_path)
        else os.path.splitext(os.path.basename(video_or_images_path))[0]
    )

    SCRIPTS = os.path.join(env_paths.CODE_BASE, "scripts")
    MICA    = os.path.join(env_paths.CODE_BASE, "src", "pixel3dmm", "preprocessing", "MICA")

    try:
        run_and_check(
            [sys.executable, "-u", "run_cropping.py", "--video_or_images_path", video_or_images_path],
            cwd=SCRIPTS,
        )
        run_and_check(
            [sys.executable, "-u", "demo.py", "-video_name", vid_name],
            cwd=MICA,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e.cmd!r} exited with {e.returncode}", file=sys.stderr)
        print("---- STDOUT ----", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("---- STDERR ----", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    tyro.cli(main)
