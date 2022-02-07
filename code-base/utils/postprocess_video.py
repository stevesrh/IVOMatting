import os

import cv2
import numpy as np

# 传入的都是灰度图
def compare_neighbor_frame(pre_frame,obj_frame,pos_frame,index=0.1):
    h,w = obj_frame.shape[0:2]
    # pre_frame/=255.
    # obj_frame/=255.
    # pos_frame/=255.
    for y in range(h):
        for x in range(w):
            print(pre_frame[y][x])
            print(obj_frame[y][x])
            print(pos_frame[y][x])
            # print("pre-pos",abs(pre_frame[y][x]-pos_frame[y][x]))
            # print("pre-obj",abs(pre_frame[y][x]-obj_frame[y][x]))
            # print("obj-pos",abs(obj_frame[y][x]-pos_frame[y][x]))

            if abs(int(pre_frame[y][x])-int(pos_frame[y][x]))<=index and abs(int(pre_frame[y][x])-int(obj_frame[y][x]))>index \
                    and abs(int(obj_frame[y][x])-int(pos_frame[y][x]))>index:
                obj_frame[y][x] = (int(pre_frame[y][x])+int(pos_frame[y][x]))/2
    return  obj_frame



if __name__ == '__main__':
    pre_alpha=cv2.imread("/Users/shenronghao/PycharmProjects/IVOMatting/code-base/posprocess_photo/00011.jpg",cv2.IMREAD_GRAYSCALE)
    print(pre_alpha.shape)
    obj_alpha=cv2.imread("/Users/shenronghao/PycharmProjects/IVOMatting/code-base/posprocess_photo/00012.jpg",cv2.IMREAD_GRAYSCALE)
    pos_alpha=cv2.imread("/Users/shenronghao/PycharmProjects/IVOMatting/code-base/posprocess_photo/00013.jpg",cv2.IMREAD_GRAYSCALE)
    # pre_alpha=np.array([np.arange(0,0.3,0.1),np.arange(0,0.6,0.2),np.arange(0,0.9,0.3),np.arange(0,0.3,0.1)])
    # obj_alpha=np.array([np.arange(0,0.3,0.1),np.arange(0,0.3,0.1),np.arange(0,0.6,0.2),np.arange(0,0.3,0.1)])
    # pos_alpha=np.array([np.arange(0,0.3,0.1),np.arange(0,0.6,0.2),np.arange(0,0.9,0.3),np.arange(0,0.3,0.1)])
    # print(pre_alpha)
    # print(obj_alpha)
    # print(pos_alpha)
    image=compare_neighbor_frame(pre_alpha,obj_alpha,pos_alpha,5)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join("/Users/shenronghao/PycharmProjects/IVOMatting/code-base/posprocess_photo", "result" + '.jpg'), image)
    # cv2.imshow('obj_alpha',obj_alpha)
    # cv2.waitKey(0)
    # print(pre_alpha)
    # print(obj_alpha)
    # print(pos_alpha)


