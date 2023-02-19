import ast
import numpy as np

observation_space = 3

coordinatedata = open('coordinatedata.txt', 'r')
motorscaling = open("motorscaling.txt", "w")

lines = coordinatedata.readlines()
temporary_data = []

for i in range(len(lines)):
    if lines[i] == ("-" + '\n'):
        if temporary_data != []:
            _right_side, right_side, _left_side, left_side = temporary_data[0], temporary_data[1], temporary_data[2], temporary_data[3]

            q_right = (right_side - _right_side)/90
            q_left = (left_side - _left_side)/90

            motorscaling.write("-" + '\n')
            motorscaling.write(str(np.concatenate((_right_side, _left_side))) + '\n')
            motorscaling.write(str(np.concatenate((q_right, q_left))) + '\n')

        temporary_data = []
    else:
        data = ast.literal_eval(lines[i].strip())
        state = np.array(data[0:3])
        temporary_data.append(state)

motorscaling = open("motorscaling.txt", "r")
lines = motorscaling.readlines()
print(len(lines))
