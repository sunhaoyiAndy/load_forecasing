import sys
import os
import time


class ConsoleOutput():
    """
    重定向输出到文件
    先调用start方法 接受两个参量，然后调用stop停止重定向
    """
    def __init__(self, name='console_output'):
        self.name = name
        self.standard_output = None

    def start(self):
        # 自定义目标文件夹和目标文件名
        dir = ".\\" + self.name
        if os.path.exists(dir):
            filepath = dir
        else:
            os.mkdir(dir)
            filepath = dir
        time_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        filename = time_name + ".txt"
        fullname = filepath + "\\" + filename
        # 备份默认的标准输出（输出值控制台）
        standard_output = sys.stdout
        # 将标准输出重定向至文件
        sys.stdout = open(fullname, "w+")
        return standard_output, fullname

    def stop(self, standard_output, fullname):
        sys.stdout.close()
        sys.stdout = standard_output
        print(open(fullname, "r").read())