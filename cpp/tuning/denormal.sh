#!/bin/sh

N=0.0; for ((i=0; i<50; ++i)); do N=${N}0; j=${N}1; echo $j $i; time -p ./a.out $j; done
