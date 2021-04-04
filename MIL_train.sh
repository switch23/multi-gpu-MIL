#!/bin/sh

#$ -cwd
#$ -V -S /bin/bash
#$ -N mil
#$ -q gpu.q@server00

# >>> conda init >>>
__conda_setup="$(CONDA_REPORT_ERRORS=false '$HOME/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="$PATH:$HOME/anaconda3/bin"
    fi
fi
unset __conda_setup
# <<< conda init <<<

conda activate openslide
time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python MIL_train.py 123 4
