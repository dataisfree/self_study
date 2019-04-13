# _*_ coding: utf-8 _*_

'''
原文： 云栖社区
title： (python)零起步数学+神经网络入门
disclaimer： 供学习之用
目的: 基于python来了解用于构建具有各种层神经网络(完全连接，卷积等)的小型库中的机器学习和代码。
假设: 对神经网络已经有所了解，重点在于说明如何正确实现
'''


'''
整个框架:
step1: 将数据输入神经网络
step2: 在得到输出之前，数据从一层流向下一层
step3: 一旦得到输出，就可以计算出一个标量误差
Step4: 可以通过相对于参数本身减去误差的导数来调整给定参数(权重或偏差)
step5: 遍历整个过程
'''

