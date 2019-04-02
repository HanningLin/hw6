import cv2 as cv
import math
import numpy as np
from numpy import *
from scipy import * 
import random

class Pic:
    def __init__(self, name, path):
         self.name=name
         self.path=path
         self.img=cv.imread(path,0)
    def show(self):
        print("data is {}".format(self.img))
        cv.imshow("7.bmp",self.img)
        cv.waitKey(30)
    def cal_h(self,method, temp_distance, n, d0, height, width):
        if method == 'butterworth_lowpass':
            return 1/(1 + (temp_distance/d0)**(2*n))
        elif method == 'gaussian_lowpass':
            return np.exp(-((temp_distance/d0)**2)/2)
        elif method == 'butterworth_highpass':
            return 1/(1 + (d0/temp_distance)**(2*n))
        elif method == 'gaussian_highpass':
            return 1 - np.exp(-((temp_distance/d0)**2)/2)
        elif method == 'laplace_highpass':
            return 4*temp_distance**2/(height**2+width**2)
        elif method == 'unmask_highpass':
            return 0.5 + 0.75*(1 - np.exp(-((temp_distance/50)**2)/2))
        else:
            print("[ERORR]:Invalid filter:{}!".format(method))
            return 0
    def filter(self,method,n,d0,task_num):
        height, width = self.img.shape
        frequency = np.fft.fft2(self.img)
        transformed = np.fft.fftshift(frequency)
        power1 = sum(np.abs(sum(transformed ** 2)))
        for i in range(height):
            for j in range(width):
                temp_distance = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
                H = self.cal_h(method,temp_distance=temp_distance, n=n, d0=d0, height=height, width=width)
                transformed[i][j] = transformed[i][j] * H
        power2 = sum(np.abs(sum(transformed ** 2)))
        freq_image = 20*np.log(np.abs(transformed) + 1)
        filted_image = np.abs(np.fft.ifft2(transformed))
        cv.imwrite('/home/hanninglin/Documents/CV/PROJECT/hw6/img/{}-{}_of_{}_n-{}_d0-{}_sr-{}.bmp'.format(task_num,method,self.name,n,d0,100*power2/power1),filted_image)
        print("[LOG]:{}-{}_of_{}_n:{}_d0:{}_sr:{}.bmp created sucessfully!".format(task_num,method,self.name,n,d0,100*power2/power1))
        print("!!!!!![RESULT]:spetral_ratio={}".format(100*power2/power1))

    def harmonic_filter(self,size, Q):
        image=self.img
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                numerator = 0
                denominator = 0
                for ii in range(size):
                    for jj in range(size):
                        temp_i = int(ii - (size - 1)/2)
                        temp_j = int(jj - (size - 1)/2)
                        if temp_i+i >= height or temp_i+i < 0 or temp_j+j >= width or temp_j+j < 0:
                            numerator += 0
                            denominator += 0
                        else:
                            if image[i+temp_i][j+temp_j] == 0:
                                numerator += 0
                                denominator += 0
                            else:
                                numerator += float(image[i+temp_i][j+temp_j])**(Q+1)
                                denominator += float(image[i+temp_i][j+temp_j])**Q
                # print(numerator, denominator)
                image[i][j] = numerator/denominator
        cv.imwrite("/home/hanninglin/Documents/CV/PROJECT/hw6/img/2_harmonic_filter_of_{}_size-{}_Q-{}.bmp".format(self.name,size,Q),image)
        print("[LOG]:Harmonic Done")

    def motion_blur(self,task,a,b,T):
        image=self.img
        height, width = image.shape
        frequency = np.fft.fft2(image)
        transformed = np.fft.fftshift(frequency)
        frequency = 20*np.log(np.abs(transformed))
        for i in range(height):
            for j in range(width):
                u = i - height/2
                v = j - width/2
                if (u * a + v * b) == 0:
                    H = T
                else:
                    H = T*np.sin(np.pi*(u*a+v*b))*np.exp(-1*np.pi*(u*a+v*b)*1j)
                    H = H/(np.pi*(u*a+v*b))
                transformed[i][j] = transformed[i][j] * H
        filted_image = np.abs(np.fft.ifft2(transformed))
        cv.imwrite("/home/hanninglin/Documents/CV/PROJECT/hw6/img/3_motion_blur.bmp",filted_image)
    
    def inverse_filtering(self, task, a, b, T, k):
        image=self.img
        height, width = image.shape
        frequency = np.fft.fft2(image)
        transformed = np.fft.fftshift(frequency)
        frequency = 20 * np.log(np.abs(transformed))
        for i in range(height):
           for j in range(width):
                u = i - height / 2
                v = j - width / 2
                if (u * a + v * b) == 0:
                    H = T
                else:
                    H = T * np.sin(np.pi * (u * a + v * b)) * np.exp(-1 * np.pi * (u * a + v * b) * 1j)
                    H = H / (np.pi * (u * a + v * b))
                    H = (np.abs(H)**2)/(H*(np.abs(H)**2 + k))
                transformed[i][j] = transformed[i][j] * H
        filted_image = np.abs(np.fft.ifft2(transformed))
        cv.imwrite("/home/hanninglin/Documents/CV/PROJECT/hw6/img/3_inverse.bmp",filted_image)
    
    def constrained_filtering(self, task, a, b, T, gamma):
        image=self.img
        height, width = image.shape

        # Generate P(u, v)
        laplace_model = np.zeros((height, width))
        i = int((height-1)/2)
        j = int((width-1)/2)
        laplace_model[i, j] = 4
        laplace_model[i - 1, j - 1] = -1
        laplace_model[i + 1, j - 1] = -1
        laplace_model[i - 1, j + 1] = -1
        laplace_model[i + 1, j + 1] = -1
        laplace_frequency = np.fft.fft2(laplace_model)
        laplace_transformed = np.fft.fftshift(laplace_frequency)

        # Do fft
        frequency = np.fft.fft2(image)
        transformed = np.fft.fftshift(frequency)

    # Filtering
        for i in range(height):
            for j in range(width):
                u = i - height / 2
                v = j - width / 2
                if (u * a + v * b) == 0:
                    H = T
                else:
                    H = T * np.sin(np.pi * (u * a + v * b)) * np.exp(-1 * np.pi * (u * a + v * b) * 1j)
                    H = H / (np.pi * (u * a + v * b))
                    H = H.conjugate()/(np.abs(H)**2 + gamma * (np.abs(laplace_transformed[i][j])**2))
                transformed[i][j] = transformed[i][j] * H
        filted_image = np.abs(np.fft.ifft2(transformed))
        cv.imwrite("/home/hanninglin/Documents/CV/PROJECT/hw6/img/3_onstrained_{}.bmp".format(gamma),filted_image)

    def addGNoise(self,method,mean=0,stddev=0):
        height, width = self.img.shape
        image=self.img
        if method == 'gaussian':
            for i in range(height):
                for j in range(width):
                    image[i][j] = image[i][j] + random.gauss(mean, stddev)
                    if image[i][j] < 0:
                        image[i][j] = 0
                    elif image[i][j] > 255:
                        image[i][j] = 255
        elif method == 'impulse_light':
            for i in range(height):
                for j in range(width):
                    temp = random.random()
                    if temp < 0.1:
                        image[i][j] = 255
                    else:
                        image[i][j] = 1.1*image[i][j]*(temp-0.1)
        elif method == 'impulse_dark':
            for i in range(height):
                for j in range(width):
                    temp = random.random()
                    if temp < 0.1:
                        image[i][j] = 0
                    else:
                        image[i][j] = 1.11*image[i][j]*(temp-0.1)
        else:
            image = image
        cv.imwrite("/home/hanninglin/Documents/CV/PROJECT/hw6/img/Noised_by_{}_mean-{}_stddev-{}.bmp".format(method,mean,stddev),image)








lena=Pic("lena","/home/hanninglin/Documents/CV/PROJECT/hw6/lena.bmp")

print("[LOG]:All object created successfully!\n")
print("#ANS:6-1\n")
print("------------------------------------------")
# 1.在测试图像上产生高斯噪声lena图-需能指定均值和方差；并用多种滤波器恢复图像，分析各自优缺点；
# lena.addGNoise('gaussian',30,10)
# lena_gaussian_noised=Pic("lena_gaussian_noised","/home/hanninglin/Documents/CV/PROJECT/hw6/img/Noised_by_gaussian_mean:30_stddev:10.bmp")
# lena_gaussian_noised.filter('butterworth_lowpass',2,50,1)
# lena_gaussian_noised.filter('gaussian_lowpass',2,50,1)
print("\n#ANS:6-2\n")
print("------------------------------------------")
# 2.在测试图像lena图加入椒盐噪声（椒和盐噪声密度均是0.1）；用学过的滤波器恢复图像；在使用反谐波分析Q大于0和小于0的作用；
# lena.addGNoise('impulse_light')
# lena.addGNoise('impulse_dark')
# lena_splight_noised=Pic("lena_splight_noised","/home/hanninglin/Documents/CV/PROJECT/hw6/img/Noised_by_impulse_light_mean:0_stddev:0.bmp")
# lena_spdark_noised=Pic("lena_spdark_noised","/home/hanninglin/Documents/CV/PROJECT/hw6/img/Noised_by_impulse_dark_mean:0_stddev:0.bmp")
# lena_splight_noised.filter('butterworth_lowpass',2,50,2)
# lena_splight_noised.filter('gaussian_lowpass',2,50,2)

# lena_spdark_noised.filter('butterworth_lowpass',2,50,2)
# lena_spdark_noised.filter('gaussian_lowpass',2,50,2)

# lena_splight_noised.harmonic_filter(3,1)
# lena_splight_noised.harmonic_filter(3,1.5)
# lena_splight_noised.harmonic_filter(3,-1.5)

# lena_spdark_noised.harmonic_filter(3,1.5)
# lena_spdark_noised.harmonic_filter(3,-1.5)

print("\n#ANS:6-3\n")
print("------------------------------------------")
# # 3.推导维纳滤波器并实现下边要求；
# # (a) 实现模糊滤波器如方程Eq. (5.6-11).
# # (b) 模糊lena图像：45度方向，T=1；
# # (c) 再模糊的lena图像中增加高斯噪声，均值= 0 ，方差=10 pixels 以产生模糊图像；
# # (d)分别利用方程 Eq. (5.8-6)和(5.9-4)，恢复图像；并分析算法的优缺点.test3.filter('laplace_highpass',0,0,3)
# lena.motion_blur( 6, 0.1, 0.1, 1)
# lena_motion_blurred=Pic("lena_motion_blurred","/home/hanninglin/Documents/CV/PROJECT/hw6/img/3_motion_blur.bmp")
# lena_motion_blurred.addGNoise('gaussian',0,10)
# lena_2=Pic("lena_2","/home/hanninglin/Documents/CV/PROJECT/hw6/img/Noised_by_gaussian_mean:0_stddev:10.bmp")
# lena_2.inverse_filtering( 6, 0.1, 0.1, 1, 0.015)
lean_3=Pic("lena_3","/home/hanninglin/Documents/CV/PROJECT/hw6/img/3_inverse.bmp")
lean_3.constrained_filtering(6, 0.1, 0.1, 1, 0.015)

