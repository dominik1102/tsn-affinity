#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J seg-train
## Liczba alokowanych węzłów
#SBATCH -N 1
#SBATCH -n 8
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem=900GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=48:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plglaoisi25-gpu-a100 
## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
## Konfiguracja GPU
#SBATCH --gres=gpu:1
 
 
## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR


srun /bin/hostname
export PYTHONPATH=$PYTHONPATH://net/people/plgrid/plgdomin088/clbench_dt_full_build
ml GCCcore/13.2.0 Python/3.11.5 
source $SCRATCH/.venv/bin/activate

##source $HOME/venvs/open-mmlab/bin/activate
##export PYTHONPATH=$HOME/workspace/cyfrovet/mmdetection
##cd $HOME/clbench_dt_full_build

cd $HOME/clbench_dt_full_build
##./tools/dist_train.sh \
 ##   cyfrovet/mask_rcnn_x101_64x4d_fpn_1x_cells_complete.py \
 ##   8 \
  ##  --work-dir ./runs


python bin/train_atari_expert.py   --spec specs_atari.json   --episodes-per-task 200   --max-len 1000   --total-steps-expert 5000000  --out-dir data/atari_expert   --device cuda   --expert-action-prob 1
##python3 main.py experiment=mimiciv_icd10/llama gpu=1


