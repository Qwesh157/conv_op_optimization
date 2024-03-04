rm -f ./*.o;
nvcc main.cu -o convbwdweight -lcudnn;
echo "##### Start test #####";
echo " N  H  W  C R S  K";
for case_k in {1..2}
do for case_c in {1..2}
    do for case_size in {1..2}
        do 
        C=$[ $case_c * 32 ]
        H=$[ $case_size * 64 ]
        W=$[ $case_size * 64 ]
        K=$[ $case_k * 128 ]
        ./convbwdweight 8 ${C} ${H} ${W} ${K} 3 3 1 1 0 0;
        done;
    done;
done;
echo "##### Test finish! #####";