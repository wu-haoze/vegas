#!/usr/bin/env bash

# Incomplete analysis
mkdir F_incomplete F+B1_incomplete F+BC_incomplete
for file in ipqs_sp_8x/*
do
    filename=$(basename ${file})
    ../verifier --milp --incomplete --num-workers=16 --input-query $file --summary-file=F_incomplete/"$filename".summary --timeout 3600
    ../verifier --milp --incomplete --backward --num-workers=16 --input-query $file --summary-file=F+B1_incomplete/"$filename".summary --timeout 3600
    ../verifier --milp  --incomplete --backward --converge --num-workers=16 --input-query $file --summary-file=F+BC_incomplete/"$filename".summary --timeout 3600
done

for file in ipqs_mnist/*
do
    filename=$(basename ${file})
    ../verifier --milp --incomplete --num-workers=16 --input-query $file --summary-file=F_incomplete/"$filename".summary --timeout 3600
    ../verifier --milp --incomplete --backward --num-workers=16 --input-query $file --summary-file=F+B1_incomplete/"$filename".summary --timeout 3600
    ../verifier --milp  --incomplete --backward --converge --num-workers=16 --input-query $file --summary-file=F+BC_incomplete/"$filename".summary --timeout 3600
done
