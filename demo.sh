#!/usr/bin/env bash
# ==============================================================================
# CVM Optimizing Compiler Backend — End-to-End Demo Script
# ==============================================================================
set -e

# Detect python command
if command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "Error: Python interpreter not found." >&2
    exit 1
fi

echo "================================================================================"
echo " CVM Optimizing Compiler Backend — Showcase Demonstration"
echo "================================================================================"

# Execute the comprehensive Python demo harness
$PY demo.py

echo ""
echo "================================================================================"
echo " Running CLI Showcase Examples"
echo "================================================================================"

echo ""
echo "[1] AI Pass Selector Explainability Report (examples/matmul.ir):"
$PY main.py examples/matmul.ir --explain

echo ""
echo "[2] Disassembled Bytecode with Virtual Registers (examples/copy_chain.ir):"
$PY main.py examples/copy_chain.ir --dis

echo ""
echo "[3] Full Regression Benchmark Suite:"
$PY benchmark.py --regression examples/

echo ""
echo "Demo finished successfully."
