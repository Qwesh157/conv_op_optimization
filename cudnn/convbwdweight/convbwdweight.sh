rm -f ./*.o;
nvcc main.cu -o convbwdweight -lcudnn -I./include;
echo "##### Start test #####";
echo " N  H  W  C R S  K";
./convbwdweight 2 128 16 16 128 3 3 1 1 0 0;
./convbwdweight 2 128 16 16 128 3 3 1 1 1 1;
./convbwdweight 2 128 16 16 128 3 3 1 1 2 2;
echo "##### Test finish! #####";