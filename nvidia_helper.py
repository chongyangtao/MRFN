import subprocess
import csv


def get_available_gpu():
    child = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory', '--format=csv,noheader'],
        stdout=subprocess.PIPE)
    text = child.communicate()[0]
    reader = csv.reader(text.split('\n')[:-1])
    for line in reader:
        memory = int(line[1].strip()[:-4])  # MB
        compute_percentage = int(line[2].strip()[:-1])  # %
        memory_percentage = int(line[3].strip()[:-1])  # %
        print('index:{} memory:{} compute_percentage:{} memory_percentage:{}'.format(line[0], memory, compute_percentage, memory_percentage))
        if memory < 1000 and compute_percentage < 5 and memory_percentage < 5:
            print('\033[32m gpu ' + line[0] + ' is available \033[0m')
            return int(line[0])
    print('\033[31m can\'t find an available gpu, please change another server!!!! \033[0m')
    #exit(-1)
    return int(input('use which GPU?'))

if __name__ == '__main__':
    get_available_gpu()
