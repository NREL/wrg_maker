#!/bin/bash
#SBATCH --ntasks=36 # I.e., one node
#SBATCH --time=100:00:00  # Jobs for big domains take a while
#SBATCH --account=boempac
#SBATCH --output=job_output_filename.%j.out  # %j will be replaced with the job ID
#SBATCH --partition=long

site_name="example_1"
h5_file=/path/to/h5_file/example_file.h5
wrg_file=/path/to/wrg_file/output.wrg
hub_height=160 # Hub height to produce WRG
resolution=2   # Needs to match resolution of WRF data
max_workers=36
buffer=0.05    # Cuts off the outer edges of WRF domain so we can construct a perfect X/Y grid

wrg_maker -v --h5_file=${h5_file} --wrg_file=${wrg_file} --hub_height=${hub_height} --resolution=${resolution} --max_workers=${max_workers} --buffer=${buffer}
