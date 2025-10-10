/*==================================================================================
*    Copyright (C) 2024 Chengdu University of Technology.
*    Copyright (C) 2024 Zifei Li.
*    
*    Filename：segydemo
*    Author：Zifei Li
*    Institute：Chengdu University of Technology
*    Email：202005050218@stu.cdut.edu.cn
*    Data：2023/03/29/
*    Function：
*    
*    This program is free software: you can redistribute it and/or modify it 
*    under the terms of the GNU General Public License as published by the Free
*    Software Foundation, either version 3 of the License, or an later version.
*=================================================================================*/
读取、重写segy、二进制格式地震数据
## Data size
数据来源为东方杯地球物理软件开发大赛的3d数据，其xline与inline两方向关键字分别为CDP和Inline,每个维度的大小可以自行运行代码查看。
## MATLAB
matlab代码使用segyread函数直接简单的将数据读取为总道头信息、每道道头信息以及数据体，通过查找对应关键字可以抽取出我们的目标数据，包含两部分：
1. 读取segy并根据道头直接抽取道集切面，并将数据体保存为二进制.dat格式
2. 根据数据的大小，通过指针跳跃式抽取对应的道集切面
## python
python使用指针从.dat数据中抽取对应的道集，与matlab读取方法的第二部分相同。并保存为npy格式
使用python读取segy的方法在github上有非常之多，每个代码的具体方法需要自己去查阅，但是总体思想和其他语言读写是一样的
## Madagascar
主文件夹下的SC脚本为Madagascar代码 抽取数据、保存数据、三维绘图,最终三维绘图成果位于Fig中
segy_write下的SC脚本为读取原数据为二进制的道集切面，再将道集切面重新写成segy数据的代码


链接: https://pan.baidu.com/s/1BSz7oNiASx3n04jYIqsrXg?pwd=meux 提取码: meux 
--来自百度网盘超级会员v4的分享
