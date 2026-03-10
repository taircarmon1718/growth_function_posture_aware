#!/bin/bash
cd /Users/taircarmon/Desktop/growth_function_posture_aware/figures
echo "Starting skeleton_length_figure.py..." > run_log.txt
/opt/anaconda3/envs/env-for-ml/bin/python3 skeleton_length_figure.py >> run_log.txt 2>&1
EXIT_CODE=$?
echo "" >> run_log.txt
echo "Exit code: $EXIT_CODE" >> run_log.txt
if [ -f "skeleton_length_figure.png" ]; then
    echo "SUCCESS: Figure generated" >> run_log.txt
    ls -lh skeleton_length_figure.png >> run_log.txt
else
    echo "ERROR: Figure not found" >> run_log.txt
fi
cat run_log.txt

