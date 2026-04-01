#!/bin/bash
# One-time Dragon/RHAPSODY environment setup for NCSA Delta.
# Run once per venv creation from a login or interactive node.

set -euo pipefail

module load cray-mpich-abi
source ~/ve/rhapsody-cray/bin/activate

echo "=== Configuring Dragon for Cray libfabric ==="
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

echo "=== Patching Dragon SLURM launcher ==="
python utils/slurm_patch.py

echo "=== Installing benchmark package ==="
pip install -e ".[dev]"

echo "=== Setup complete ==="
