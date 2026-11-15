#!/bin/bash

LSAN_OPTIONS=suppressions=asan.supp:print_suppressions=0 mpirun -n $1 $2
