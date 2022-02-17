import os

file_names=os.listdir("/media/lab505/Toshiba/compare_test/combined-1K-selected/images")
for file_name in file_names:
    file=file_name.split(".png")[-2]
    file=os.path.join("/media/lab505/Toshiba/compare_test/combined-1K-selected/mask",file)
    os.mkdir(file)

