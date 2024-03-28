##  How to run

###  Task 1  

```shell
python ./QP_KKT.py
```

###  Task 2

```
cd low-dim-QP/
mkdir build
cd build 
cmake .. && make
./sdqp_example
```

###  Task 3

```
cd mpc-car
catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=Yes
source devel/setup.zsh
roslaunch mpc_car simulation.launch
```

###  Result

***See Report.md for more details***