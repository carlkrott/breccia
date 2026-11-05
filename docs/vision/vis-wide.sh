#!/bin/bash
# Non-square JPEG (2560x1440): Breccia serve vs llama.cpp server ref, same 1024-2048 token budget.
cd /var/tmp/breccia-coord
IMGS=/home/korphaus/Pictures/variety-copied-wallpaper-8d16c88555ef4d72ad93d322239cbe4a.jpg bash vis-serve.sh
bash vision/ref-wide.sh
echo WIDE DONE
