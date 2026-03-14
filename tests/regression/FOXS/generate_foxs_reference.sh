#!/bin/bash
# Generate FOXS reference profile for 5PTI regression test
# Uses the imp_env conda environment with IMP installed
#
# Requires: conda create -n imp_env -c salilab imp
#
# Usage: ./generate_foxs_reference.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDB_FILE="${SCRIPT_DIR}/fixtures/5PTI_frame0.pdb"
OUTPUT_DAT="${SCRIPT_DIR}/fixtures/5PTI_frame0_foxs_reference.dat"

if [ ! -f "$PDB_FILE" ]; then
    echo "Error: PDB file not found at $PDB_FILE"
    exit 1
fi

echo "Generating FOXS reference profile..."
echo "Input PDB:  $PDB_FILE"
echo "Output DAT: $OUTPUT_DAT"

# Run FoXS with standard parameters:
# -h: include explicit hydrogens
# Default: profile_size=500, max_q=0.50, c1=1.0, c2=0.0
conda run -n imp_env foxs -h "$PDB_FILE" 2>&1 | grep -v "^$"

# Move the generated .dat file to the expected location
if [ -f "${PDB_FILE}.dat" ]; then
    mv "${PDB_FILE}.dat" "$OUTPUT_DAT"
    echo "Reference profile saved to: $OUTPUT_DAT"
else
    echo "Error: FoXS did not generate expected output file ${PDB_FILE}.dat"
    exit 1
fi

echo "Done!"
