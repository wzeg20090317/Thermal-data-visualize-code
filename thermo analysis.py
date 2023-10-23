import matplotlib.pyplot as plt
import contourpy
import pandas as pd
import numpy as np
import pathlib
import cv2
import os
import sys
sys.path.append("../contours/15V")


class Thermo_analysis():
    def __init__(self,TC_reading,time1,time2,freq1,freq2,
                 contour_temp_limit, time_profile,time_contour,time_video,
                 TC_loc='location.csv'):
        self.TC_loc=TC_loc
        self.TC_location=pd.read_csv(TC_loc,index_col=0)
        self.TC_reading=TC_reading      #pd.read_csv('15h_data_modified.csv')

        self.time_reading1=time1
        self.time_reading2 = time2
        self.frequency1 = freq1 #sampling freq of reading 1
        self.frequency2 = freq2  #sampling freq of reading 2
        self.time_profile=time_profile
        self.time_contour=time_contour
        self.target_temp=contour_temp_limit
        self.time_video=time_video

        self.temp_low_bound = -1
        self.file_path=pathlib.Path(__file__).parent.parent.resolve()

        self.profiles=['col0','col1','col2','col3','col4','col-1','col-2','col-3','col-4',
                       'row0','row1','row2','row3','row4','row-1','row-2','row-3','row-4']
        self.profile_temp_limits_2 = {
            'col0': 2, 'col1': 2, 'col2': 2, 'col3': 2, 'col4': 2, 'col-1':2, 'col-2': 2, 'col-3': 2,
            'col-4': 2,
            'row0': 2, 'row1': 2, 'row2': 2, 'row3': 2, 'row4': 2, 'row-1': 2, 'row-2': 2, 'row-3': 2,
            'row-4': 2
        }
        self.profile_temp_limits_30 = {
            # 'col0': 30, 'col1': 15, 'col2': 10, 'col3': 6, 'col4': 5, 'col-1': 15, 'col-2': 10, 'col-3': 6,
            # 'col-4': 5,
            # 'row0': 30, 'row1': 15, 'row2': 10, 'row3': 5, 'row4': 3, 'row-1': 15, 'row-2': 10, 'row-3': 5,
            # 'row-4': 3

            'col0': 30, 'col1': 20, 'col2': 15, 'col3': 15, 'col4': 8, 'col-1': 20, 'col-2': 15, 'col-3': 15,
            'col-4': 8,
            'row0': 30, 'row1': 20, 'row2': 15, 'row3': 10, 'row4': 8, 'row-1': 20, 'row-2': 15, 'row-3': 10,
            'row-4': 8
        }
        self.profile_temp_limits_35 = {

            'col0': 35, 'col1': 25, 'col2': 15, 'col3': 15, 'col4': 10, 'col-1': 25, 'col-2': 15, 'col-3': 15,
            'col-4': 10,
            'row0': 35, 'row1': 25, 'row2': 15, 'row3': 15, 'row4': 10, 'row-1': 25, 'row-2': 15, 'row-3': 15,
            'row-4': 10
        }
        self.profile_temp_limits_40 = {
            'col0': 45, 'col1': 32, 'col2': 32, 'col3': 30, 'col4': 30, 'col-1': 32, 'col-2': 32, 'col-3': 30,
            'col-4': 30,
            'row0': 45, 'row1': 42, 'row2': 40, 'row3': 36, 'row4': 34, 'row-1': 42, 'row-2': 40, 'row-3': 36,
            'row-4': 34

            # 'col0': 45, 'col1': 32, 'col2': 32, 'col3': 30, 'col4': 30, 'col-1': 32, 'col-2': 32, 'col-3': 30,
            # 'col-4': 30,
            # 'row0': 45, 'row1': 42, 'row2': 40, 'row3': 36, 'row4': 34, 'row-1': 42, 'row-2': 40, 'row-3': 36,
            # 'row-4': 34
        }

        self.profile_temp_limits_48 = {
            'col0': 48, 'col1': 25, 'col2': 20, 'col3': 10, 'col4': 6, 'col-1': 25, 'col-2': 20, 'col-3': 10,
            'col-4': 6,
            'row0': 48, 'row1': 25, 'row2': 20, 'row3': 10, 'row4': 8, 'row-1': 25, 'row-2': 20, 'row-3': 10,
            'row-4': 8

            # 'col0': 48, 'col1': 32, 'col2': 32, 'col3': 30, 'col4': 30, 'col-1': 32, 'col-2': 32, 'col-3': 30,
            # 'col-4': 30,
            # 'row0': 48, 'row1': 45, 'row2': 42, 'row3': 38, 'row4': 38, 'row-1': 45, 'row-2': 42, 'row-3': 38,
            # 'row-4': 38
        }
        self.profile_temp_limits_50={
            'col0':55, 'col1':40, 'col2':35, 'col3':30, 'col4':30, 'col-1':40, 'col-2':35, 'col-3':30, 'col-4':30,
            'row0':55, 'row1':40, 'row2':35, 'row3':30, 'row4':30, 'row-1':40, 'row-2':35, 'row-3':30, 'row-4':30
        }
        self.profile_temp_limits_55 = {
            'col0':55, 'col1':35, 'col2':25, 'col3':20, 'col4':15, 'col-1':35, 'col-2':25, 'col-3':20, 'col-4':15,
            'row0':55, 'row1':35, 'row2':25, 'row3':20, 'row4':15, 'row-1':35, 'row-2':25, 'row-3':20, 'row-4':15
        }
        self.profile_temp_limits_65 = {
            'col0': 65, 'col1': 55, 'col2': 45, 'col3': 35, 'col4': 32, 'col-1': 55, 'col-2': 45, 'col-3': 35,
            'col-4': 32,
            'row0': 65, 'row1': 60, 'row2': 45, 'row3': 35, 'row4': 32, 'row-1': 60, 'row-2': 45, 'row-3': 35,
            'row-4': 32
        }
        self.profile_temp_limits_70 = {
            'col0': 75, 'col1': 55, 'col2': 45, 'col3': 35, 'col4': 32, 'col-1': 55, 'col-2': 45, 'col-3': 35, 'col-4': 32,
            'row0': 75, 'row1': 60, 'row2': 45, 'row3': 35, 'row4': 32, 'row-1': 60, 'row-2': 45, 'row-3': 35, 'row-4': 32
        }

        self.profile_temp_limits_75 = {
            'col0': 75, 'col1': 55, 'col2': 45, 'col3': 35, 'col4': 32, 'col-1': 55, 'col-2': 45, 'col-3': 35,
            'col-4': 32,
            'row0': 75, 'row1': 60, 'row2': 45, 'row3': 35, 'row4': 32, 'row-1': 60, 'row-2': 45, 'row-3': 35,
            'row-4': 32
        }
        self.profile_temp_limits={2:self.profile_temp_limits_2,30:self.profile_temp_limits_30,35:self.profile_temp_limits_35,
                                  40:self.profile_temp_limits_40,48:self.profile_temp_limits_48,
                                  50:self.profile_temp_limits_50, 55:self.profile_temp_limits_55,
                                  65:self.profile_temp_limits_65,70:self.profile_temp_limits_70,
                                  75:self.profile_temp_limits_75}


    def plot_cross_section(self, location):
        t = self.time_profile
        TC_location=self.TC_location

        color_list = plt.cm.jet(np.linspace(0, 0.9, len(t)))
        legend_list=[]


        no_of_TC = list(TC_location[location].isnull()).count(False)
        list_TC = list(TC_location[location][:no_of_TC])
        x = TC_location.loc[list_TC]['x']  # x coords
        y = TC_location.loc[list_TC]['y']  # y coords

        data1 = self.TC_reading[list_TC]



        fig=plt.figure()  #figsize=(15, 1.8)  3, 12
        fig1=plt.subplot(111)
        for i in range(len(t)):
            if t[i]<self.time_reading1:
                zlist=data1.iloc[int(360 * t[i])]


            if location[0]== 'c':
                fig1.plot( zlist,y, linestyle='dashed', ms=3, color=color_list[i])
                legend_list.append('Time=%s h' % t[i])

            else:
                fig1.plot(x, zlist,  linestyle='dashed', ms=3, color=color_list[i])
                legend_list.append('Time=%s h' % t[i])

        if location[0] == 'c':
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Y location (in)')
            plt.xlim(self.temp_low_bound, self.profile_temp_limits[self.target_temp][location])
        else:
            plt.xlabel('X location (in)')
            plt.ylabel('Temperature (°C)')
            plt.ylim(self.temp_low_bound, self.profile_temp_limits[self.target_temp][location])


        plt.legend(legend_list,loc='upper center',bbox_to_anchor=(1.2, 0.8), fancybox=True)

        box = fig1.get_position()
        fig1.set_title(location)
        fig1.set_position([box.x0, box.y0, box.width * 0.8, box.height])


    def save_profile(self, location):  # here takes a string that represents the col or row of TC
        self.plot_cross_section(location)
        plt.savefig(str(self.file_path)+'/cross-section/%s.png' % location)

    def show_profile(self, location):  # here takes a string that represents the col or row of TC
        self.plot_cross_section(location)
        plt.show()

    def save_all_profile(self):  # here takes no inputs
        for i in self.profiles:  # profiles saved in properities
            self.save_profile(location=i)

    def plot_contour(self, t):

        start_time1 = 0
        start_time2 = self.time_reading1

        data1 = self.TC_reading

        colorbar_low = 0
        colorbar_high = int(self.target_temp)
        colorbar_range=colorbar_high-colorbar_low
        colorbar_tick=2
        if colorbar_range>30:
            colorbar_tick=5
        TC_location = self.TC_location


        #add 2 additional points to set up the upper boundary
        y0=TC_location['y'][0]
        x = [-18,18]+list(TC_location['x'])
        y =  [y0,y0]+list(TC_location['y'])


        if t < start_time2:
            z = list(data1.iloc[int(360* (t - start_time1))+3][1:65])
        if self.TC_loc =='location-10in.csv':
            z=z[5:]


        z=[z[0],z[0]]+z




        fig, (cntr_line) = plt.subplots(nrows=1)
        cntr_color = cntr_line.tricontourf(x, y, z, levels=np.linspace(self.temp_low_bound, self.target_temp, 91), cmap="Spectral_r")  # Spectral_rRdBu_r

        fig.colorbar(cntr_color, ax=cntr_line, spacing='proportional',label='Temprature °C',ticks=[i for i in range (colorbar_low,colorbar_high,colorbar_tick)])

        cntr_line.plot(x, y, 'ko', ms=2) # plot the location of the TC
        cntr_line.set(xlim=(-18, 18), ylim=(-20, 18))

        cntr_line.set_title('Temperature Contour-%s h' % str(t))

        '''this part is the text of temperature on the plot'''

        offset=0  # when the depth is -10 in, the configuration of the sample is different, we need a offset for plotting
        if self.TC_loc == 'location-10in.csv':
            offset=5

        for i in range(26-offset):
            cntr_line.text(x[i+2]-1, y[i+2]-0.5, str(round(z[i+2],1)),  fontsize=8,
                    verticalalignment='top')
        for i in range(27-offset,36-offset):
            cntr_line.text(x[i+2]-1, y[i+2]-0.5, str(round(z[i+2],1)),  fontsize=8,
                    verticalalignment='top')
        for i in range(37-offset,63-offset):
            cntr_line.text(x[i+2]-1, y[i+2]-0.5, str(round(z[i+2],1)),  fontsize=8,
                    verticalalignment='top')
        cntr_line.text(x[28-offset] + 0.5, y[28-offset] + 1, str(round(z[28-offset], 1)), fontsize=8,
                       verticalalignment='top')
        cntr_line.text(x[38-offset] - 2.1, y[38-offset] +1, str(round(z[38-offset], 1)), fontsize=8,
                       verticalalignment='top')
        cntr_line.text(x[65-offset] - 1, y[65-offset] + 1.5, str(round(z[65-offset], 1)), fontsize=8,
                       verticalalignment='top')


        plt.xlabel('X coord (in)')
        plt.ylabel('Y coord (in)')
        # plt.subplots_adjust(hspace=102.5)

    def save_contour(self, t):  # here takes a string that represents the col or row of TC
        self.plot_contour(t)

        plt.savefig(str(self.file_path)+'/contours/%sh.png' % t)

    def show_contour(self, t): #here takes a string that represents the col or row of TC
        self.plot_contour(t)
        plt.show()

    def save_all_contour(self): #here takes list of strings that represents the col or row of TC
        for i in self.time_contour:
            self.save_contour(i)

    def save_video(self):

        for i in range(self.time_video*10+1):
            self.plot_contour(i/10)

            plt.savefig(str(self.file_path) + '/contours/%s.png' % i)
            plt.close()

class png2video():
    def __init__(self,image_loc,video_name):
        self.image_loc=image_loc
        self.video_name=video_name

    def get_video(self):

        images = [img for img in os.listdir(self.image_loc) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.image_loc, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(self.video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))


        for image in range(1951):
            video.write(cv2.imread(os.path.join(self.image_loc, str(image)+".png")))

        cv2.destroyAllWindows()
        video.release()

masonry_test=Thermo_analysis(TC_reading=pd.read_csv('example_dataset1.csv'),
                     time_profile=[0.01,1,5, 10,20],
                 time_contour=[ 0,0.1, 0.5, 1, 2,3, 5, 10, 15, 20,25,30,40,50,60,70,80,90,100,110,120],
                             TC_loc='location.csv',
                    time1=200,time2=200,freq1=0.1,freq2=0.1,contour_temp_limit=52,time_video=100)

masonry_test.temp_low_bound = -1
# masonry_test.save_video()#
# masonry_test.save_all_profile()
# masonry_test.save_all_contour()
# masonry_test.save_contour(20)
# print(masonry_test.file_path)
masonry_test.show_contour(20)
# masonry_test.show_profile('col1')
# masonry_test.save_profile('row0')



# video_test=png2video(image_loc="../contours", video_name='5V.avi')
# video_test.get_video()