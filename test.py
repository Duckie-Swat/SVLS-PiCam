import numpy as np

num_list = [[263, 104, '9'], [219, 311, '6'], [649, 313, '9'], [70, 305, '7'], [350, 306, '0'], [151, 100, '5'], [516, 313, '4'], [592, 100, '1'], [482, 103, 'N']]

def get_license_plate(num_list):
    def min_Y(array_2d):
        res = float('inf')
        for a in array_2d:
            if a[1] < res:
                res = a[1]
        return res
    min_y = min_Y(num_list)    
    line1 = []
    line2 = []
    THRESH_HOLD = 100
    for a in num_list:
        if abs(a[1] - min_y) < THRESH_HOLD:
            line1.append(a)
        else:
            line2.append(a)

    line1 = sorted(line1, key=lambda e: e[0])
    line2 = sorted(line2, key=lambda e: e[0])

    if len(line2) == 0:  # if license plate has 1 line
        license_plate = "".join([str(ele[2]) for ele in line1])
    else:   # if license plate has 2 lines
        license_plate = "".join([str(ele[2]) for ele in line1]) + "-" + "".join([str(ele[2]) for ele in line2])
    return license_plate


print(get_license_plate(num_list))


