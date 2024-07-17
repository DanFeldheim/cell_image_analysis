#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:19:59 2024

This app imports cell microscopy images and analyzes them for various features such as colocalization. 
Image cropping, denoising, thresholding, cell counting, and colocalization using Mander's and Pearson coefficients are 
included.'

@author: danfeldheim
"""

# Imports
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
from pretty_notification_box import notification_box
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib import colors
import cv2 as cv
# from streamlit_cropperjs import st_cropperjs
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
st.set_option('deprecation.showfileUploaderEncoding', False)
from streamlit_cropper import st_cropper
import streamlit.components.v1 as components
import io
import scipy.stats
from scipy import stats
import statsmodels.api as sm
# from scipy.stats import spearmanr
import mpld3
# from mpld3 import plugins
# import plotly.express as px
import plotly.graph_objects as go



class Setup():
    """
    Instantiates the working directory and markdown styles. Creates a header and sidebar for the app and an upload button for data. 
    """
    
    # Set fonts and basic layout.
    def __init__(self):
        
        # Create a Working Directory
        self.working_directory = "/Users/danfeldheim/Documents/cell_image_pro_app/"
        
        st.markdown("""<style>
                        .small-font {
                        font-size:30px !important;
                        color:blue}
                        </style>
                        """, unsafe_allow_html=True)
                        
        st.markdown("""<style>
                        .purple-font-venti {
                        font-size:24px !important;
                        color:Purple}
                        </style>
                        """, unsafe_allow_html = True)
                        
        st.markdown("""<style>
                        .green-font-sm {
                        font-size:18px !important;
                        color:Green}
                        </style>
                        """, unsafe_allow_html = True)
                        
        st.markdown("""<style>
                        .green-font {
                        font-size:20px !important;
                        color:Green;
                        font-weight: bold;}
                        </style>
                        """, unsafe_allow_html = True)
       
        # Create a button style
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: rgb(0, 102, 102);
            color: white;
            
            font-size:20px;
            font-weight: bold;
            margin: auto;
            display: block;
        }
        div.stButton > button:hover {
        	background:linear-gradient(to bottom, rgb(0, 204, 204) 5%, rgb(0, 204, 204) 100%);
        	background-color:rgb(0, 204, 204);
        }
        div.stButton > button:active {
        	position:relative;
        	top:3px;
        }
        </style>""", unsafe_allow_html=True)
    
    # Write a header                            
    def header(self):
        st.markdown('<p class = "small-font">Velocity Sciences Image Processor</p>', unsafe_allow_html = True) 
        
    def sidebar(self): 
        with st.sidebar:
            self.brightness = st.slider("Brightness", min_value = -50, max_value = 50, value = 0)
            self.realtime_update = st.sidebar.checkbox(label="Update in Real Time", value = True)
            # Add a crop tool with a blue box. 
            self.box_color = st.sidebar.color_picker(label = "Box Color", value = '#0000FF')
            self.aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
            self.aspect_dict = {
                "1:1": (1, 1),
                "16:9": (16, 9),
                "4:3": (4, 3),
                "2:3": (2, 3),
                "Free": None
            }
            self.aspect_ratio = self.aspect_dict[self.aspect_choice]
            # Adjustable imaging smoothing parameters
            self.sigma_s = st.number_input("Sigma S", value = 10, min_value = 1, max_value = 30)
            self.sigma_r = st.number_input("Sigma R", value = 0.05, min_value = 0.01, max_value = 0.2)
            # Select box for color channels to remove from the image
            self.color = st.multiselect("Choose colors to remove.", ["Green", "Red", "Blue"])
            # Plot histogram and colocalization if desired
            self.color_histogram = st.checkbox(label = 'Plot color histogram')
            self.color_gaussian = st.checkbox(label = 'Gaussian Analysis')
            self.cell_count = st.checkbox(label = 'Live-Dead Count')
            self.colocalization = st.checkbox(label = 'Colocalization')
            if self.colocalization:
                self.bleed_correction = st.number_input("Bleed Correction", value = 0, min_value = 0, max_value = 50)
                self.correct_channel = st.radio('Select Channel to Correct', [':red[Red]', ':green[Green]', ':blue[Blue]'])
            
            
           
class Prep_Image(Setup):        
    """Class to perform image processing"""
    
    def __init__(self):
        
        super().__init__()
      
    # Create an upload button for data   
    def upload_image(self):
        
        st.write('')
        notification_box(icon='warning', title='Warning', textDisplay = 'The file uploaded must be in jpg, png, or jpeg format', styles = None, externalLink = None, url = None, key ='foo')
        
        self.image_file = st.file_uploader('Choose an image', type = ['jpg', 'png', 'jpeg'])
        
        if not self.image_file:
            st.stop()
        
        # Import image
        self.image = Image.open(self.image_file)
        self.image = self.image.convert('RGB')
        # st.write('image: ', self.image)
        
        st.markdown('<p class = "purple-font-venti">Select an area to process.</p>', unsafe_allow_html = True) 
        
    def crop_denoise(self):
       
        # Call st.cropper to instantiate the cropper box
        self.cropped_image = self.cropping(self.image)
        
        # Convert to numpy array
        self.image_array = np.array(self.cropped_image)
 
        # Enhance
        self.enhanced_img = self.enhance_details(self.image_array)
        # st.write('enhanced image: ', self.enhanced_img.shape)
        
        # Adjust brightness
        self.brightened_img = self.brighten(self.enhanced_img)
        
        # Denoise
        self.denoised_image = self.denoise(self.image_array)
        # st.write('image array: ', self.image_array.shape)
        # st.write('denoised image: ', self.denoised_image.shape)
        
        return self.image_array

    def display_images(self):
        
        # Display images
        col1, col2, col3, buff = st.columns([1,1,1,1])
        
        with col1:
            st.markdown('<p class = "green-font-sm">Original Image</p>', unsafe_allow_html = True) 
            
            # Display cropped image
            st.image(self.image_array)
            self.download_image(self.image_array, "Download Cropped Image")
         
        with col2:
            st.markdown('<p class = "green-font-sm">Smoothed Image</p>', unsafe_allow_html = True) 
            
            if self.color:
                # Delete one or 2 color channels from the image and display
                self.color_deletion = self.color_picker(self.brightened_img, self.color)
                st.image(self.color_deletion)
                self.download_image(self.color_deletion, "Download False-Colored Image")
                
            else:
                st.image(self.brightened_img)
                self.download_image(self.brightened_img, "Download Smoothed Image")
                
        with col3:
            st.markdown('<p class = "green-font-sm">Denoised Image</p>', unsafe_allow_html = True) 
            st.image(self.denoised_image)
            self.download_image(self.denoised_image, "Download Denoised Image")
            
    def enhance_details(self, image_file):
        
        enhanced_image = cv.detailEnhance(image_file, sigma_s = self.sigma_s, sigma_r = self.sigma_r)
        return enhanced_image

    def brighten(self, image):
        brightened_image = cv.convertScaleAbs(image, beta = self.brightness)
        return brightened_image
    
    def cropping(self, image):
        
        if not self.realtime_update:
            st.write("Double click to save crop")
            
        # Get a cropped image from the frontend
        cropped_img = st_cropper(image, realtime_update = self.realtime_update, box_color = self.box_color,
                                    aspect_ratio = self.aspect_ratio)
        
        # Manipulate cropped image at will
        _ = cropped_img.thumbnail((850,850))
        return cropped_img
        
    def color_picker(self, image_array, colors):
        """Function to remove 1 or 2 colors from the image."""
        try:
            for color in colors:
                if color == 'Red':
                    image_array[...,0] *= 0
                elif color == 'Green':
                    image_array[...,1] *= 0
                elif color == 'Blue':
                    image_array[...,2] *= 0
        except:
            pass
        
        return image_array
    
    def download_image(self, image, btn_text):
        
        pilImage = Image.fromarray(image)
        pilImage = pilImage.resize((600, 400))
        buffer = io.BytesIO()
        pilImage.save(buffer, format = "PNG")
        btn = st.download_button(
            label = btn_text,
            data = buffer,  
            file_name = "image.png",
            mime = "image/png",
        )
        
    def color_sum(self, image_array): 
        """Sums all the pixel intensities from each channel."""
        
        if self.sum:
            red = image_array[:, :, 0]
            green = image_array[:, :, 1]
            blue = image_array[:, :, 2]
            red_sum = np.sum(red)
            green_sum = np.sum(green)
            blue_sum = np.sum(blue)
        
        return (red_sum, green_sum, blue_sum)
    
        
    def denoise(self, image_array):
        """Denoises the image using opencv's color image non-local means denoiser: 
            https://www.javatpoint.com/fastnlmeansdenoising-in-python"""
        
        # Denoise the image
        denoised_image = cv.fastNlMeansDenoising(image_array, None, 15, 7, 21)  
        return denoised_image
        
    def thresh(self, red, green, blue):
        
        r_thr = cv.adaptiveThreshold(red, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        g_thr = cv.adaptiveThreshold(green, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        b_thr = cv.adaptiveThreshold(blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
       
        return r_thr, g_thr, b_thr
           
class Colocalization(Prep_Image):        
    """Class to perform colocalization analysis"""
     
    def __init__(self):
         
        super().__init__()      
        
    def histo(self, crop):
        
        if self.color_histogram:
            
            # Plot brightness histogram
            st.write('')
            st.write('')
            st.markdown('<p class = "purple-font-venti">RGB Histogram</p>', unsafe_allow_html = True) 
            buff, col2 = st.columns([1,15])
            with col2:
                self.histogram(self.image_array)
                
    def gaussian_test(self):
        
        if self.color_gaussian:
            
            split_image = self.split_channels(self.denoised_image)
            flat_img = list(self.flatten(split_image[0], split_image[1], split_image[2]))
            flat_img = [array.tolist() for array in flat_img]
            st.write('')
            st.write('')
            # Box plots of the rgb channels to check for normal distribution
            # If median line isn't in the middle or if the whiskers aren't the same size, the distribution isn't normal
            # and pearson's correlation test isn't valid
            st.markdown('<p class = "purple-font-venti">Normal Distribution Tests</p>', unsafe_allow_html = True) 
            st.markdown('<p class = "green-font">Box Plots</p>', unsafe_allow_html = True) 
            fig = plt.figure(figsize=(7, 2))
            flat_rgb_df = pd.DataFrame({'Red': flat_img[0], 'Green': flat_img[1], 'Blue': flat_img[2]})
            box = flat_rgb_df.boxplot(fontsize = 7, grid = False)
            plt.xlabel('Channel', fontsize = 8)
            plt.ylabel('Intensity', fontsize = 8)
            st.pyplot(fig)
            
            # Q-Q plots
            st.markdown('<p class = "green-font">Q-Q Plots</p>', unsafe_allow_html = True) 
            fig, axs = plt.subplots(1,3, figsize = (7, 2))
            stats.probplot(np.array(flat_img[0]), dist = 'norm', plot = axs[0])
            axs[0].set_title('Red Channel')
            stats.probplot(np.array(flat_img[1]), dist = 'norm', plot = axs[1])
            axs[1].set_title('Green Channel')
            stats.probplot(np.array(flat_img[2]), dist = 'norm', plot = axs[2])
            axs[2].set_title('Blue Channel')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Shapiro-Wilk test
            st.write('')
            st.markdown('<p class = "green-font">Shapiro-Wilk Test</p>', unsafe_allow_html = True) 
            shapiro_red = stats.shapiro(flat_img[0])
            shapiro_green = stats.shapiro(flat_img[1])
            shapiro_blue = stats.shapiro(flat_img[2])
           
            if shapiro_red.pvalue or shapiro_green.pvalue or shapiro_blue.pvalue <0.05:
                st.markdown(''':red[Warning: The Shapiro-Wilk test indicates that 
                            the intensities of at least one channel are not normally distributed.      
                            (p values < 0.05 mean the distribution is not normal.)  
                            In this case, the Pearson analysis of colocalization may not be as reliable as the Spearman analysis.]''')
                            
                col1, buff = st.columns([1,2])
                with col1:
                    # details_button = st.button('Show Data')
                    with st.expander('Show Data'):
                    
                        st.write(
                            'Red Channel W = ', str(round(shapiro_red.statistic, 2)), "\n\n"
                            # 'Red Channel p value = ', str(shapiro_red.pvalue), "\n\n"
                            'Red Channel p value = ', str('{:0.2e}'.format(shapiro_red.pvalue)), "\n\n"
                            'Green Channel W = ', str(round(shapiro_green.statistic, 2)), "\n\n"
                            'Green Channel p value = ', str('{:0.2e}'.format(shapiro_green.pvalue)), "\n\n"
                            # 'Green Channel p value = ', str(shapiro_green.pvalue), "\n\n"
                            'Blue Channel W = ', str(round(shapiro_blue.statistic, 2)), "\n\n"
                            'Blue Channel p value = ', str('{:0.2e}'.format(shapiro_blue.pvalue)), "\n\n"
                            # 'Blue Channel p value = ', str(shapiro_blue.pvalue), "\n\n"
                            )
             
            else:
                st.markdown(''':blue[Warning: All channels passsed the Shapiro-Wilk test.]''')
 
                
    def colocalizationAnalysis(self):
        
        try: 
            if self.colocalization:
                
                st.write('')
                st.write('')
                st.markdown('<p class = "purple-font-venti">Colocalization Analysis</p>', unsafe_allow_html = True) 
                # Split image into rgb channels
                
                split_image = self.split_channels(self.denoised_image)
                
                col1, col2, col3, buff = st.columns([1,1,1, 3])
                with col1:  
                    st.markdown(''':red[Red Channel]''')
                    st.image(split_image[0])
                with col2:
                    st.markdown(''':green[Green Channel]''')
                    st.image(split_image[1])
                with col3:
                    st.markdown(''':blue[Blue Channel]''')
                    st.image(split_image[2])
                    
                # Call colocalization function to analyze colocalized dyes
                st.write('')
                st.write('')
                flat_img = list(self.flatten(split_image[0], split_image[1], split_image[2]))
                flat_img = [array.tolist() for array in flat_img]
                # st.write('Before Correction: ', flat_img)
                
                # Subtract the bleed correction value from the desired channel
                if self.correct_channel == ':red[Red]':
                    for i in range(len(flat_img[0])):
                        if flat_img[0][i] <= self.bleed_correction:
                            flat_img[0][i] = flat_img[0][i] - self.bleed_correction
                elif self.correct_channel == ':green[Green]':
                    for i in range(len(flat_img[1])):
                        if flat_img[1][i] <= self.bleed_correction:
                            flat_img[1][i] = flat_img[1][i] - self.bleed_correction
                else:
                    for i in range(len(flat_img[2])):
                        if flat_img[2][i] <= self.bleed_correction:
                            flat_img[2][i] = flat_img[2][i] - self.bleed_correction
                            
                # Set all negative numbers to 0
                for l in flat_img:
                    for i in range(len(l)):
                        if l[i] < 0:
                            l[i] = 0
                # st.write('after Correction: ', flat_img)
                    
                
                # Plot each channel vs. the others and get pearson coeffs
                st.markdown('<p class = "green-font">Intensity Correlation Plots</p>', unsafe_allow_html = True) 
                coloc_analysis = list(self.pearson_coloc(flat_img[0], flat_img[1], flat_img[2]))
                coloc_analysis = [round(x,2) for x in coloc_analysis]

                # Change any p value <0.005 to "<0.005" for easier interpreation
                # Involves changing every other element in list up to the last p value (element 5 so use 6 in the slice)
                coloc_analysis[1:6:2] = ['<0.005' if x < 0.005 else x for x in coloc_analysis[1:6:2]]
                # st.write(coloc_analysis)
                
                g_r_conf = str(coloc_analysis[6]) + ' to ' + str(coloc_analysis[7])
                r_b_conf = str(coloc_analysis[8]) + ' to ' + str(coloc_analysis[9])
                b_g_conf = str(coloc_analysis[10]) + ' to ' + str(coloc_analysis[11])
                
                self.pearson_coeff = {'Statistic':['Correlation (-1 to 1)', 'p value', '95% CL'], 
                                      'Green-Red':[coloc_analysis[0], coloc_analysis[1], g_r_conf], 
                                      'Blue-Red':[coloc_analysis[2], coloc_analysis[3], r_b_conf], 
                                      'Green-Blue':[coloc_analysis[4], coloc_analysis[5], b_g_conf]}
                                      
                pearson_df = pd.DataFrame(self.pearson_coeff)
                # Create list that assigns colocalization meaning to each Pearson R value
                # This will be appended as a row with the entry in the first column = Conclusion
                pearson_interpretation = ["Conclusion"]
                # Subset the r coefficients away from the p values
                pearson_R = [coloc_analysis[0], coloc_analysis[2], coloc_analysis[4]]
                for coef in pearson_R:
                    if 0.75 <= coef <= 1.0: 
                        pearson_interpretation.append('Strongly Colocalized')
                    elif 0.50 <= coef <= 0.74:
                        pearson_interpretation.append('Weakly Colocalized')
                    elif -0.50 <= coef <= 0.49:
                        pearson_interpretation.append('Random Mixture')
                    else:
                        pearson_interpretation.append('Separated')
                       
                pearson_df.loc[len(pearson_df)] = pearson_interpretation
                self.create_table(pearson_df, 'Pearson Colocalization Results')
                     
                # Calculate Spearman's rank correlation and return as list of lists
                spearman_analysis = list(self.spearman(flat_img[0], flat_img[1], flat_img[2]))
                
                spearman_r = [spearman_analysis[0][0], spearman_analysis[1][0], spearman_analysis[2][0]]
                spearman_r = [round(x,2) for x in spearman_r]
                spearman_p = [spearman_analysis[0][1], spearman_analysis[1][1], spearman_analysis[2][1]]
                
                spearman_p = [round(x,2) for x in spearman_p]
                # Change any p value <0.005 to "<0.005" for easier interpreation
                spearman_p = ['<0.005' if x < 0.005 else x for x in spearman_p]
        
                self.spearman_coeff = {'Statistic':['Correlation (-1 to 1)', 'p value'], 'Green-Red':[spearman_r[0], spearman_p[0]], 
                                      'Blue-Red':[spearman_r[1], spearman_p[1]],
                                      'Green-Blue':[spearman_r[2], spearman_p[2]]}
                spearman_df = pd.DataFrame(self.spearman_coeff)
                
                # Create list that assigns colocalization meaning to each Pearson R value
                # This will be appended as a row with the entry in the first column = Conclusion
                spearman_interpretation = ["Conclusion"]
               
                for coef in spearman_r:
                    if 0.75 <= coef <= 1.0: 
                        spearman_interpretation.append('Strongly Colocalized')
                    elif 0.50 <= coef <= 0.74:
                        spearman_interpretation.append('Weakly Colocalized')
                    elif -0.50 <= coef <= 0.49:
                        spearman_interpretation.append('Random Mixture')
                    else:
                        spearman_interpretation.append('Separated')
                
                spearman_df.loc[len(spearman_df)] = spearman_interpretation
                self.create_table(spearman_df, 'Spearman Rank Colocalization Results')
                
                # Calculate Mander's M1 and M2 coefficients
                manders = self.manders(flat_img)
                self.manders_df = pd.DataFrame({'Manders Coeffs':['M1', 'M2'], 'Red-Green':round(manders[0],2), 
                                                'Red-Blue':round(manders[1]),'Green-Blue':round(manders[2])})
        
        
                fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1], header = dict(values = list(self.manders_df[['Manders Coeffs', 'Red-Green', 
                                                        'Red-Blue', 'Green-Blue']].columns),fill_color = '#FD8E72',
                                                        line_color='darkslategray', align = 'center', 
                                                        font = dict(color = 'blue', size = 18)), 
                                                        cells = dict(values = [self.manders_df['Manders Coeffs'], 
                                                        self.manders_df['Red-Green'], self.manders_df['Red-Blue'], 
                                                        self.manders_df['Green-Blue']],
                                                        fill_color = 'lavender', align = 'center', line_color='darkslategray',
                                                        height = 30)))
                
                fig.update_traces(cells_font = dict(size = 15))
                
                fig.update_layout(title = 'Manders Co-occurance Results', title_font = dict(size = 18, color = 'green',
                                  family = 'Arial'), title_x = 0.0, title_y = 1.0, height = 175,
                                    margin = dict(l = 0, r = 0, b = 0, t = 35))
                st.write('')
                st.write('')
                st.write(fig) 
            
        except UnboundLocalError:
            pass
        
        
    def histogram(self, image_array):
        
        # Save cropped image to working directory
        im = Image.fromarray(image_array)
        saveCrop = im.save(self.working_directory + 'temp_file.png')
        
        # Open image in opencv
        cvImage = cv.imread(self.working_directory + 'temp_file.png')
        cvImage = cv.cvtColor(cvImage, cv.COLOR_BGR2RGB)
        color = ('b','g','r')
        figure = plt.figure()
        for i,col in enumerate(color):
            hist = cv.calcHist([cvImage],[i],None,[256],[0,256])
            plt.plot(hist, color = col)
            plt.xlim([0,256])
            plt.xlabel('Brightness', fontsize = 18)
            plt.ylabel('# of Pixels', fontsize = 18)
            plt.tick_params(axis = 'x', labelsize = 14)
            plt.tick_params(axis = 'y', labelsize = 14)
            plt.title('Brightness vs. Pixel Count for R, G, B Channels', color = 'blue' , fontsize = 18, y = 0.9)
            plt.tight_layout()
            fig_html = mpld3.fig_to_html(figure)
        components.html(fig_html, height = 500)
    
    def split_channels(self, image_array):
        
        red = image_array[:, :, 0]   
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]
        
        return red, green, blue
    
    def pearson_coloc(self, red_flat, green_flat, blue_flat):
        """Analyzes the colocalization using Person's method."""
           
        g_r_pear = scipy.stats.pearsonr(red_flat, green_flat)
        r_b_pear =scipy.stats.pearsonr(red_flat, blue_flat)
        b_g_pear = scipy.stats.pearsonr(green_flat, blue_flat)
        
        g_r_CL = g_r_pear.confidence_interval(confidence_level = 0.95)
        r_b_CL = r_b_pear.confidence_interval(confidence_level = 0.95)
        b_g_CL = b_g_pear.confidence_interval(confidence_level = 0.95)
        
        pearson_results_list = [g_r_pear.statistic, g_r_pear.pvalue, r_b_pear.statistic, r_b_pear.pvalue, b_g_pear.statistic,
                                b_g_pear.pvalue, g_r_CL[0], g_r_CL[1], r_b_CL[0], r_b_CL[1], b_g_CL[0], b_g_CL[1]]
        
        # Plot R vs G vs B
        fig, axs = plt.subplots(1,3, figsize = (7, 2))
        axs[0].scatter(red_flat, green_flat, s = 1)
        axs[0].set_title('Green-Red')
        # axs[0].set_xlabel('Red Channel')
        # axs[0].set_ylabel('Green Channel')
        axs[1].scatter(red_flat, blue_flat, s = 1)
        axs[1].set_title('Blue-Red')
        # ax2.xlabel('Red Channel')
        # ax2.ylabel('Blue Channel')
        axs[2].scatter(blue_flat, green_flat, s = 1)
        axs[2].set_title('Green-Blue')
        # ax3.xlabel('Blue Channel')
        # ax3.ylabel('Green Channel')
        plt.tight_layout()
        st.pyplot(fig)
              
        # return g_r_pear.statistic, g_r_pear.pvalue, r_b_pear.statistic, r_b_pear.pvalue, b_g_pear.statistic, \
        #         b_g_pear.pvalue, g_r_CL, r_b_CL, b_g_CL
        
        return pearson_results_list
    
    def flatten(self, red, green, blue):
        
        red_flat = red.flatten()
        green_flat = green.flatten()
        blue_flat = blue.flatten()
        
        return red_flat, green_flat, blue_flat
    
    def spearman(self, red_flat, green_flat, blue_flat):
        
        g_r_spearman = scipy.stats.spearmanr(red_flat, green_flat)
        r_b_spearman =scipy.stats.spearmanr(red_flat, blue_flat)
        b_g_spearman = scipy.stats.spearmanr(green_flat, blue_flat)
        
        return g_r_spearman, r_b_spearman, b_g_spearman
    
    def manders(self, flat_img):
        
        # Create df of intensities for each channel
        rgb_df = pd.DataFrame({'Red':flat_img[0], 'Green':flat_img[1], 'Blue':flat_img[2]})
        
        rgb_df['rxg'] = rgb_df['Red'] * rgb_df['Green']
        rgb_df['rxb'] = rgb_df['Red'] * rgb_df['Blue']
        rgb_df['gxb'] = rgb_df['Green'] * rgb_df['Blue']
        # st.write(rgb_df)
        
        # Add up the intensities of each channel
        total_red, total_green, total_blue = rgb_df['Red'].sum(), rgb_df['Green'].sum(), rgb_df['Blue'].sum()
        # st.write(total_red, total_green, total_blue)
       
        rg_count = 0
        gr_count = 0
        
        rb_count = 0
        br_count = 0
        
        gb_count = 0
        bg_count = 0
        
        for row in rgb_df.itertuples():
            if row.rxg > 0:
                rg_count += row.Red
                gr_count += row.Green
            if row.rxb > 0:
                rb_count += row.Red
                br_count += row.Blue
            if row.gxb > 0:
                gb_count += row.Green
                bg_count += row.Blue    
    
        # st.write(rg_count, rb_count, gb_count)
        rg_M1 = rg_count/total_red
        rg_M2 = gr_count/total_green
        rb_M1 = rb_count/total_red
        rb_M2 = br_count/total_blue
        gb_M1 = gb_count/total_green
        gb_M2 = bg_count/total_blue
        
        return rg_M1, rg_M2, rb_M1, rb_M2, gb_M1, gb_M2
    
    def create_table(self, df, title):
        
        
        fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1], header = dict(values = list(df[['Statistic', 'Green-Red', 
                                                'Blue-Red', 'Green-Blue']].columns),fill_color = '#FD8E72',
                                                line_color ='darkslategray', align = 'center', 
                                                font = dict(color = 'blue', size = 18)), 
                                                cells = dict(values = [df['Statistic'], 
                                                df['Green-Red'], df['Blue-Red'], 
                                                df['Green-Blue']],
                                                fill_color = 'lavender', align = 'center', line_color = 'darkslategray',
                                                height = 30)))
                                                                                          
        
        fig.update_traces(cells_font = dict(size = 15))
        # In update layout both the margin and height control the spacing between tables in the vertical direction. 
        fig.update_layout(title = title, title_font = dict(size = 18, color = 'green',
                family = 'Arial'), title_x = 0, title_y = 1.0, height = 175,
                          margin = dict(l = 0, r = 0, b = 0, t = 20))
        st.write('')
        st.write('')
        st.write(fig)         
        
class Cell_Counter(Colocalization):
    
    def __init__(self):
         
        super().__init__() 
        
    def thresh(self, crop):
        
        try: 
            if self.cell_count:
                st.write('')
                st.write('')
                st.markdown('<p class = "purple-font-venti">Live vs. dead cell analysis.</p>', unsafe_allow_html = True) 
                # self.img = cv.imread('/Users/danfeldheim/Documents/cell_image_pro_app/240621_Channel1_20x_01_DC.png')
                
                self.img = crop.copy()
                self.gray_img  = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
                # st.image(self.gray_img)
                # st.write(self.gray_img)
                self.gray_img = cv.GaussianBlur(self.gray_img, (5, 5), 0)
                # st.image(self.gray_img)
                
                ret, self.thresh = cv.threshold(self.gray_img, 20, 255, cv.THRESH_BINARY)
                self.thresh = cv.erode(self.thresh, None, iterations=2)
                self.thresh = cv.dilate(self.thresh, None, iterations=2)
                # st.image(thresh)
                # st.write(thresh)
               
                self.cell_count = 0
                # CHAIN_APPROX_SIMPLE throws out some points so the contour around the edge will be a dotted line
                # contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                self.contours, self.hierarchy = cv.findContours(self.thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
               
                
                # Create dicts for cell parameters
                self.areaDict = {}
                self.aspectRatioDict = {}
                self.avgColorDict = {}
                self.perimeterDict = {}
                self.redMeanDict = {}
                self.greenMeanDict = {}
                self.blueMeanDict = {}
              
                for contour in self.contours:
                    # Set a minimum area 
                    if cv.contourArea(contour) > 150:
                        
                        self.cell_count += 1
                       
                        # Fit an elipse to each cell
                        ellipse = cv.fitEllipse(contour)
                        # A more direct route to the major and minor axis distances
                        (x,y), (MA, ma), angle = cv.fitEllipse(contour)
                        cv.ellipse(self.img, ellipse, (255,255, 255), 2, cv.LINE_AA)
                        
                        # Draw the contour that follows the cell perfectly
                        cv.drawContours(self.img, contour, -1, (0, 255, 255), 2)
                        
                        self.areaDict[self.cell_count] = cv.contourArea(contour)
                        
                        # Get aspect ratio
                        # Several methods shown here
                        
                        # Method 1: Also good for getting the average rgb intensities inside a rectangular contour around the cell
                        x,y,w,h = cv.boundingRect(contour)
                        # Draw the bounding rectangle
                        # cv.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
                        # Calculate aspect ratio
                        # if w > h:
                        #     self.aspect_ratio = float(w)/h
                        # else:
                        #     self.aspect_ratio = float(h)/w
                        # st.write('Method 1: ', self.aspect_ratio)
                        
                        # Get the average RGB intensities
                        self.mean_intensities = np.array(cv.mean(self.img[y:y+h,x:x+w])).astype(np.uint8)
                        self.redMeanDict[self.cell_count] = self.mean_intensities[0]
                        self.greenMeanDict[self.cell_count] = self.mean_intensities[1]
                        self.blueMeanDict[self.cell_count] = self.mean_intensities[2]
                        # st.write('Average color (RGB): ', self.mean_intensities)
                         
                        # Method 2-strange thing is that ma is minor axis, MA major axis, but MA/ma < 1
                        self.aspect_ratio = round(ma/MA, 3)
                        # st.write('Method 2: ', self.aspect_ratio)
                        
                        # Method 3
                        # rect = cv.minAreaRect(contour)
                        # st.write(rect)
                        # # st.write(rect)
                        # width = rect[1][0]
                        # height = rect[1][1]
                        # if width > height:
                        #     self.aspect_ratio = width/height
                        # else:
                        #     self.aspect_ratio = height/width
                        # st.write('Method 3: ', self.aspect_ratio)
                        
                        # self.aspect_ratio = float(w)/h
                        self.aspectRatioDict[self.cell_count] = self.aspect_ratio
                        
                        self.perim = cv.arcLength(contour, True)
                        self.perimeterDict[self.cell_count] = self.perim
                        
                        # Label every cell with a number
                        M = cv.moments(contour)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv.putText(self.img, text = str(self.cell_count), org=(cx,cy),
                                fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255),
                                thickness = 2, lineType=cv.LINE_AA)
             
                # Print the threshold image
                # st.image(thresh, use_column_width = True, clamp = True)
                # Print the real image with contours and labels
                st.image(self.img, use_column_width = True, clamp = True)
                # st.write(self.cell_count)
                
                # st.write('Area = ', self.areaDict)
                # st.write('AR = ', self.aspectRatioDict)
                # st.write('Perimeter = ', self.perimeterDict)
               
                number_of_cells = [x for x in range(1,self.cell_count + 1)]
                ar_list = [value for key, value in self.aspectRatioDict.items()]
                # st.write(ar_list)
                red_list = [value for key, value in self.redMeanDict.items()]
                green_list = [value for key, value in self.greenMeanDict.items()]
                blue_list = [value for key, value in self.blueMeanDict.items()]
                
                st.write('')
                st.write('')
                live_dead_threshold = st.slider('Select an intensity ratio for Live-Dead Analysis', 50, 255, 100, 10)
                
                # Create a live-dead list
                self.live_dead_list = []
                for intensity in red_list:
                    if intensity > live_dead_threshold:
                        self.live_dead_list.append('Dead')
                    else:
                        self.live_dead_list.append('Live')
               
                self.cell_df = pd.DataFrame({"Cell Number":number_of_cells, 
                                             # "Aspect Ratio":ar_list, 
                                              "Mean Red Intensity":red_list, 
                                              "Mean Green Intensity":green_list,
                                              "Mean Blue Intensity":blue_list,
                                              "Cell Status":self.live_dead_list})
                # st.write(self.cell_df)
                fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1,1], header = dict(values = list(
                                                        self.cell_df[['Cell Number', 
                                                                      # 'Aspect Ratio', 
                                                                      'Mean Red Intensity', 
                                                                      'Mean Green Intensity', 
                                                                      'Mean Blue Intensity', 
                                                                      'Cell Status']].columns),fill_color = '#FD8E72',
                                                        line_color='darkslategray', align = 'center', 
                                                        font = dict(color = 'blue', size = 18)), 
                                                        cells = dict(values = [self.cell_df['Cell Number'], 
                                                                               # self.cell_df['Aspect Ratio'], 
                                                                               self.cell_df['Mean Red Intensity'], 
                                                                               self.cell_df['Mean Green Intensity'], 
                                                                               self.cell_df['Mean Blue Intensity'],
                                                                               self.cell_df['Cell Status']],
                                                        fill_color = 'lavender', align = 'center', line_color = 'darkslategray',
                                                        height = 30)))
                                                                                                  
                
                fig.update_traces(cells_font = dict(size = 15))
                # In update layout both the margin and height control the spacing between tables in the vertical direction. 
                fig.update_layout(title = "Cell Analysis", title_font = dict(size = 18, color = 'green',
                        family = 'Arial'), title_x = 0, title_y = 1.0, height = 300, width = 1000,
                                  margin = dict(l = 0, r = 0, b = 0, t = 35))
                st.write('')
                st.write('')
                st.write(fig)         
                
                double_cell_list = []
                for ar in ar_list:
                    if ar >2.0:
                        double_cell_list.append("Possible Double Cell")
                    else:
                        double_cell_list.append("Single Cell")

                with st.expander('See Notes Regarding Cell Under Counting'):
                 
                    st.write(
                        'The following cells may be so close that they were counted as one cell:' "\n\n")
                    
                    for number, status in enumerate(double_cell_list):
                        if status == 'Possible Double Cell':
                            st.write('Cell #' + str(number + 1))
                            
                number_of_objects = str(len(self.cell_df))
                live_cells = self.cell_df['Cell Status'].str.count('Live').sum()
                dead_cells = self.cell_df['Cell Status'].str.count('Dead').sum()
                multiplets = double_cell_list.count('Possible Double Cell')
                             
                # Add up cell counts
                self.cell_totals_df = pd.DataFrame({'Number of Objects Found':number_of_objects, 
                                'Live Cells':live_cells,
                                'Dead Cells':dead_cells,
                                'Possible Multiplets':multiplets}, index = [0])
                
                
                fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1,1], header = dict(values = list(
                                                        self.cell_totals_df[['Number of Objects Found', 
                                                                      'Live Cells', 
                                                                      'Dead Cells', 
                                                                      'Possible Multiplets']].columns),fill_color = '#FD8E72',
                                                        line_color='darkslategray', align = 'center', 
                                                        font = dict(color = 'blue', size = 18)), 
                                                        cells = dict(values = [self.cell_totals_df['Number of Objects Found'], 
                                                                               self.cell_totals_df['Live Cells'], 
                                                                               self.cell_totals_df['Dead Cells'],
                                                                               self.cell_totals_df['Possible Multiplets']],
                                                        fill_color = 'lavender', align = 'center', line_color = 'darkslategray',
                                                        height = 30)))
                                                                                                  
                
                fig.update_traces(cells_font = dict(size = 15))
                # In update layout both the margin and height control the spacing between tables in the vertical direction. 
                fig.update_layout(title = "Cell Totals", title_font = dict(size = 18, color = 'green',
                        family = 'Arial'), title_x = 0, title_y = 1.0, height = 500, width = 1000,
                                  margin = dict(l = 0, r = 0, b = 0, t = 35))
                
                st.write('')
                st.write('')
                st.write(fig) 
                
        except:
            pass
        

        
       
        
        
        

# Run 
if __name__ == '__main__':

    # Call Cell_Counter class with all the classes and methods above it inherits 
    obj1 = Cell_Counter()
    header = obj1.header()
    sidebar = obj1.sidebar()
    upload = obj1.upload_image()
    crop = obj1.crop_denoise()
    display = obj1.display_images()
    # Pass crop so the histo method inherts self.image_array
    histo = obj1.histo(crop)
    gaussian = obj1.gaussian_test()
    colocAnalysis = obj1.colocalizationAnalysis()
    threshhold = obj1.thresh(crop)


    


    
    
    
    
    
    
    