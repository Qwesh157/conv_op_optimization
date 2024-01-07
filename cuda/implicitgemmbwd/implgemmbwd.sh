make clean
make
echo "##### Start test #####";
echo " N  H  W  C R S  K";
./implgemmbwd 2 128 16 16 128 3 3 1 1 0 0;
./implgemmbwd 2 128 16 16 128 3 3 1 1 1 1;
./implgemmbwd 2 128 16 16 128 3 3 1 1 2 2;
echo "##### Test finish! #####";
