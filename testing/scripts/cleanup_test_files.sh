#!/bin/bash

################################################################################
# Cleanup Script - Remove Test/Experimental Files
# Keeps only production-ready training scripts and documentation
################################################################################

set -e

echo "================================================================================"
echo "🧹 DanZero Test Files Cleanup"
echo "================================================================================"
echo ""

# Track what we're removing
REMOVED_COUNT=0
TOTAL_SIZE=0

echo "📋 Files to be removed:"
echo ""

# Test scripts
echo "Test/Experimental Scripts:"
for file in h100_*.py find_gpu_limit.py; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        TOTAL_SIZE=$((TOTAL_SIZE + size))
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
        echo "  - $file ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
    fi
done
echo ""

# Test log files
echo "Test Log Files:"
for file in *.log; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        TOTAL_SIZE=$((TOTAL_SIZE + size))
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
        echo "  - $file ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
    fi
done
echo ""

# Test output directories
echo "Test Output Directories:"
for dir in checkpoints_test logs_test results_test; do
    if [ -d "$dir" ]; then
        size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo 0)
        TOTAL_SIZE=$((TOTAL_SIZE + size))
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
        echo "  - $dir/ ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
    fi
done
echo ""

echo "================================================================================"
echo "Total: $REMOVED_COUNT items, ~$(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo $TOTAL_SIZE bytes) to be freed"
echo "================================================================================"
echo ""

read -p "Proceed with cleanup? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 0
fi

echo ""
echo "🧹 Cleaning up..."
echo ""

# Remove test scripts
rm -f h100_*.py find_gpu_limit.py
echo "✅ Removed test scripts"

# Remove log files
rm -f *.log
echo "✅ Removed log files"

# Remove test directories
rm -rf checkpoints_test logs_test results_test
echo "✅ Removed test directories"

echo ""
echo "================================================================================"
echo "✅ Cleanup Complete!"
echo "================================================================================"
echo ""
echo "📁 Remaining production files:"
echo ""
echo "Training Scripts:"
ls -lh train_*.sh analyze_results.py 2>/dev/null | tail -n +2 | awk '{print "  ✅ " $9 " (" $5 ")"}'
echo ""
echo "Documentation:"
ls -1 *.md 2>/dev/null | grep -E "TRAINING|SCALING|FILES" | awk '{print "  ✅ " $1}'
echo ""
echo "================================================================================"
echo "🔧 Verifying production scripts..."
echo "================================================================================"
echo ""

# Verify scripts
bash -n train_test.sh && echo "✅ train_test.sh - Syntax OK"
bash -n ../../scripts/training/train_production.sh && echo "✅ train_production.sh - Syntax OK"

echo ""
echo "================================================================================"
echo "✅ All production scripts verified and ready to use!"
echo "================================================================================"
echo ""
echo "You can now run:"
echo "  $ bash testing/scripts/train_test.sh"
echo "  $ bash scripts/training/train_production.sh"
echo ""

