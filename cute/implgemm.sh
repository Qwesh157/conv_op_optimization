#!/bin/sh
make clean;
make;
./implgemm 8 16 16 16 16 3 3;