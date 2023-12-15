#!/bin/bash
for task in "box-inpaint" "random-inpaint" "gaussian-blur" "motion-blur" "super-resolution" "gamma" "sobel";
do
    python main.py --task_name $task
done