"""
Patch Dragon's slurm.py for NCSA Delta:
- Replaces --ntasks={nnodes} with --ntasks-per-node=1
- Adds --overlap flag
- Backs up original on first run, restores from backup on every run (idempotent)
- Deletes __pycache__ to force recompilation
"""

import os
import shutil


def patch_slurm_file():
    try:
        import dragon.launcher.wlm.slurm as slurm_module
        slurm_path = slurm_module.__file__
    except ImportError:
        print("[ERROR] Could not find Dragon slurm module.")
        return

    backup_path = slurm_path + ".bak"

    if not os.path.exists(backup_path):
        shutil.copy(slurm_path, backup_path)
        print(f"[INFO] Backup created at {backup_path}")
    else:
        shutil.copy(backup_path, slurm_path)
        print(f"[INFO] Restored original from {backup_path}")

    with open(slurm_path, "r") as f:
        content = f.read()

    OLD_SRUN_CMD = (
        'SRUN_COMMAND_LINE = "srun --nodes={nnodes} --ntasks={nnodes} --cpu_bind=none -u -l -W 0"'
    )
    NEW_SRUN_CMD = (
        'SRUN_COMMAND_LINE = (\n'
        '        "srun "\n'
        '        "--nodes={nnodes} "\n'
        '        "--ntasks-per-node=1 "\n'
        '        "--cpu_bind=none "\n'
        '        "--overlap "\n'
        '        "-u -l -W 0"\n'
        '    )'
    )

    OLD_BE_ARGS = (
        '        slurm_launch_be_args = [\n'
        '            "srun",\n'
        '            f"--nodes={args_map[\'nnodes\']}",\n'
        '            f"--ntasks={args_map[\'nnodes\']}",\n'
        '            "--cpu_bind=none",\n'
        '            f"--nodelist={args_map[\'nodelist\']}",\n'
        '        ]'
    )
    NEW_BE_ARGS = (
        '        slurm_launch_be_args = [\n'
        '            "srun",\n'
        '            f"--nodes={args_map[\'nnodes\']}",\n'
        '            "--ntasks-per-node=1",\n'
        '            "--cpu_bind=none",\n'
        '            "--overlap",\n'
        '        ]'
    )

    if OLD_SRUN_CMD not in content:
        print("[WARN] SRUN_COMMAND_LINE not found in original — already patched or file changed?")
    else:
        content = content.replace(OLD_SRUN_CMD, NEW_SRUN_CMD)

    if OLD_BE_ARGS not in content:
        print("[WARN] _get_wlm_launch_be_args body not found — already patched or file changed?")
    else:
        content = content.replace(OLD_BE_ARGS, NEW_BE_ARGS)

    with open(slurm_path, "w") as f:
        f.write(content)

    print(f"[SUCCESS] Dragon slurm.py patched at {slurm_path}")

    pycache_dir = os.path.join(os.path.dirname(slurm_path), "__pycache__")
    if os.path.exists(pycache_dir):
        shutil.rmtree(pycache_dir)
        print(f"[INFO] Removed __pycache__ at {pycache_dir}")


if __name__ == "__main__":
    patch_slurm_file()
