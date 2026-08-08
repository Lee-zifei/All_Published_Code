# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：seis.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/05/20/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def snr(output, target):
	eps = 1e-5
	results = 0
	batch_size = target.size(0)
	for i in range(batch_size):
		loss1 = sum(sum((target[i, 0, :, :] - output[i, 0, :, :]) ** 2)) + eps
		loss2 = sum(sum(target[i, 0, :, :] ** 2)) + eps
		results += 10 * torch.log(loss2 / loss1)
	return results / batch_size