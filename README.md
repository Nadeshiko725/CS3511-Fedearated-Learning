### CS3511 Federated Learning Project

This repository is for the solo project of CS3511 2024 Spring. There are three stages in this experiment. Before you starting running it, please make sure you have installed the libs I mentioned in my report, and the structure of my files are in `tree.txt`

#### Stage 1

run the `stage12_multi.py` by:

```powershell
$ python stage12_multi.py
```

Then you are asked to input all the parameters like:

![image-20240512233335809](C:/Users/daijunhao/AppData/Roaming/Typora/typora-user-images/image-20240512233335809.png)

If the stage1 trainning processes starts successfully, you will see:

<img src="C:/Users/daijunhao/AppData/Roaming/Typora/typora-user-images/image-20240512233646933.png" alt="image-20240512233646933" style="zoom: 80%;" />

#### Stage 2

run the `stage12_multi.py` by:

```powershell
$ python stage12_multi.py
```

Then you are asked to input all the parameters like:

<img src="C:/Users/daijunhao/AppData/Roaming/Typora/typora-user-images/image-20240512233842718.png" alt="image-20240512233842718" style="zoom: 80%;" />

If the stage2 trainning processes starts successfully, you will see:

<img src="C:/Users/daijunhao/AppData/Roaming/Typora/typora-user-images/image-20240512233827493.png" alt="image-20240512233827493" style="zoom:80%;" />

#### Stage 3

run the `stage3.py` by

```powershell
$ python stage3.py
```

the inputs are as same as Stage2, so I won't show it again.

#### Files

All the result files are named with their parameters such as client numbers, mode, and communication methods. All the `.pth` files are storaged in `./models`,  and all the log files are storaged in `./client_log/stage*` .