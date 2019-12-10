#!/usr/bin/env python
# coding: utf-8

# In[14]:


from PIL import Image
from matplotlib import rcParams
import os
import matplotlib
import pandas as pd
import mglearn
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = 200
matplotlib.use('Agg')
# rcParams['figure.figsize'] = 14, 12
# rcParams['font.size'] = 16
# rcParams['axes.labelsize'] = 14
# rcParams['xtick.labelsize'] = 13
# rcParams['ytick.labelsize'] = 13
# rcParams['legend.fontsize'] = 15

# import seaborn as sns
# sns.set_style("whitegrid")


# In[ ]:


class CreateData(object):
    def __init__(self):
        self.base_path = "D:informconnection"
        self.train_path = ""
        self.valid_path = ""

    def create_directory(self, base_path=os.getcwd()):
        self.train_path = self.base_path + "\\" + "train"
        self.valid_path = self.base_path + "\\" + "test"
        for fold in [self.train_path, self.valid_path]:
            for subf in ["tan", "sin_cos", "linear", 'exp']:
                os.makedirs(os.path.join(fold, subf))
        self.sin_cos()
        self.tan()
        self.linear_function()
        self.exp()
        return self
#     np.random(-1, 1)

    def tan(self):
        count = 0
        for begin in np.arange(0, 3, 0.1):
            for step in range(100, 1000, 100):
                x = np.linspace(0, 3*np.pi, step)
                y = np.tan(x)
                count += 1
                self.save_plot(x, y, self.train_path + "\\" + "tan" +
                               "\\" + "_tan_{}".format(count) + ".jpg", begin, True)

    def sin_cos(self):
        count = 0
        for multip in range(5, 10):
            for step in range(100, 1000, 100):
                for divide in [1, 2, 3]:
                    for noise in [0, 0.1, 0.2, 0.3]:
                        x = np.linspace(-multip * np.pi, multip * np.pi, step)
                        y = np.sin(x / divide) + [np.random.uniform(-noise, noise)
                                                  for _ in range(step)]
                        plt.plot(x, y)
                        count += 1
                        self.save_plot(x, y,
                                       self.train_path + "\\" + "sin_cos" +
                                       "\\" + "_sinCos_{}".format(count) +
                                       ".jpg")

    def linear_function(self):
        count = 0
        for noise in [0.1, 0.2, 0.3]:
            for koef in np.arange(-10, 10, 0.5):
                x = [i for i in np.arange(10 * abs(koef))]
                for i in range(2):
                    y = 1/koef * np.array(x) if i == 0 else koef * np.array(x)
                    y += [np.random.uniform(-noise, noise)
                          for _ in range(len(y))]
                    plt.figure(figsize=(15, 10))
                    fig = plt.gca()
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    if koef > 0:
                        plt.ylim(0, 10)
                        plt.xlim(0, 10)
                    else:
                        plt.ylim(-10, 0)
                        plt.xlim(0, 10)
                    fig.plot(x, y)
                    if koef:
                        count += 1
                        plt.savefig(self.train_path + "\\" + "linear" +
                                    "\\" + "_linear_{}".format(count) +
                                    ".jpg")
            for i in range(-9, 60):

                x_const = np.array([i for i in range(100)])
                y_const = np.array([10 for _ in range(100)]) + [np.random.uniform(-0.3, 0.3)
                                                                for _ in range(len(x_const))]
                plt.figure(figsize=(15, 10))
                fig = plt.gca()
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 100)
                plt.ylim(-i, 20)
                plt.plot(x_const, y_const)
                count += 1
                plt.savefig(self.train_path + "\\" + "linear" +
                            "\\" + "_linear_{}".format(count) +
                            ".jpg")
        self.__rotation_some_image('linear')

    def exp(self):
        count = 0
        for noise in range(210, 310, 10):
            for end in range(10, 30):
                plt.figure(figsize=(15, 10))
                fig = plt.gca()
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                x = np.linspace(2, 10, 100)
                y = np.exp(x) + [np.random.uniform(-noise, noise)
                                 for _ in range(len(x))]
                plt.plot(x, y)
                count += 1
                plt.xlim(2, end)
                plt.savefig(self.train_path + "\\" + "exp" +
                            "\\" + "_exp_{}".format(count) +
                            ".jpg")
        self.__rotation_some_image('exp', None)

    def __rotation_some_image(self, string, angle=90):
        path = self.train_path + '\\' + string + '\\'
        count = 0
        for end_of_string in os.listdir(path):
            count += 1
            image = Image.open(path + end_of_string)
            if angle:
                image = image.rotate(angle)
            else:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(path + string + '_rotate{}.jpg'.format(count))

    def save_plot(self, x, y, path, begin=None, flag=False):
        plt.figure(figsize=(15, 10))  # think about scale
        if flag:
            plt.xlim(begin, 3*np.pi)
        fig = plt.gca()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        fig.plot(x, y)
        plt.savefig(path)