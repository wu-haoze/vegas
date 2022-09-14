#!/usr/bin/env bash


# Complete analysis
mkdir F F+B1 F+BC A+F+BC
for file in oopsla22/ipqs_sp_20x/*
do
    filename=$(basename ${file})
    ../verifier --milp --num-workers=16 --input-query $file --summary-file=F/"$filename".summary --timeout 3600
    ../verifier --milp --backward --num-workers=16 --input-query $file --summary-file=F+B1/"$filename".summary --timeout 3600
    ../verifier --milp --backward --converge --num-workers=16 --input-query $file --summary-file=F+BC/"$filename".summary --timeout 3600
    ../verifier --milp --backward --converge --relax --num-workers=16 --input-query $file --summary-file=A+F+BC/"$filename".summary --timeout 3600
done

