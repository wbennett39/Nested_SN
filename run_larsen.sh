#!/bin/bash
#$ -M wbennet2@nd.edu
#$ -m abe
#$ -N larsen_problem
#$ -q long


module load python
python3 Larsen_run.py

