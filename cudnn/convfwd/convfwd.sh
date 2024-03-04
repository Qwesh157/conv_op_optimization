rm -f ./*.o;
nvcc main.cu -o convfwd -lcudnn -I./include;
echo "##### Start test #####";
echo " N  H  W  C R S  K";
./convfwd 2 128 16 16 128 3 3 1 1 0 0;
./convfwd 2 128 16 16 128 3 3 1 1 1 1;
./convfwd 2 128 16 16 128 3 3 1 1 2 2;
echo "##### Test finish! #####";