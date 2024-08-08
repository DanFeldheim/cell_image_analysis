#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:19:59 2024

This app imports cell microscopy images and analyzes them for various features such as colocalization. 
Image cropping, denoising, thresholding, cell counting, and colocalization using Mander's, Spearman, and Pearson coefficients are 
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
import cv2 as cv
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
import mpld3
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
            
            # Plot color channel histogram 
            self.color_histogram = st.checkbox(label = 'Plot color histogram')
            
            # Perform a gaussian analysis
            self.color_gaussian = st.checkbox(label = 'Gaussian Analysis')
            
            # Count cells
            self.cell_count = st.checkbox(label = 'Live-Dead Count')
            if self.cell_count:
                self.min_contour_area = st.number_input("Minimum Cell Area", value = 150, min_value = 2, max_value = 500)
                self.threshold = st.number_input("Threshold", value = 20, min_value = 5, max_value = 100)
                self.intensity = st.checkbox(label = 'Total RGB Intensity')
                
            # Perform a colocalization analysis
            self.colocalization = st.checkbox(label = 'Colocalization')
            if self.colocalization:
                self.bleed_correction = st.number_input("Bleed Correction", value = 0, min_value = 0, max_value = 50)
                self.correct_channel = st.radio('Select Channel to Correct', [':red[Red]', ':green[Green]', ':blue[Blue]'])           
           
class Prep_Image(Setup):        
    """Class to upload and perform image processing"""
    
    def __init__(self):
        
        super().__init__()
      
    # Create an upload button for data   
    def upload_image(self):
        
        st.write('')
        notification_box(icon='warning', title='Warning', textDisplay = 'The file uploaded must be in jpg, png, or jpeg format', styles = None, externalLink = None, url = None, key ='foo')
        
        self.image_file = st.file_uploader('Choose an image', type = ['jpg', 'png', 'jpeg', 'tiff'])
        
        if not self.image_file:
            st.stop()
        
        # Import image
        self.image = Image.open(self.image_file)
        self.image = self.image.convert('RGB')
        # st.write('image: ', self.image)
        
        st.markdown('<p class = "purple-font-venti">Select an area to process.</p>', unsafe_allow_html = True) 
        
    # Instantiate the cropper box and convert contents into np array
    # Calls brightness and denoising functions
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
        
        return self.image_array

    # Displays cropped image
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
            
    # Applies an image enhancer
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
        # pilImage = pilImage.resize((600, 400))
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
                
    # Several tests for a gaussian distribution are included here
    # This is needed because the pearsonr test of colocalization requires a gaussian distribution
    # The Spearman test does not
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
                st.markdown(''':blue[Congratulations! All channels passsed the Shapiro-Wilk test.]''')
 
    # Run several tests for colocalization            
    def colocalizationAnalysis(self):
        
        # Wrap in try/except to catch any errors
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
                
                # Make some room
                st.write('')
                st.write('')
                
                # Flatten the arrays into 1D lists
                flat_img = list(self.flatten(split_image[0], split_image[1], split_image[2]))
                flat_img = [array.tolist() for array in flat_img]
                # st.write('Before Correction: ', flat_img)
                
                # Subtract the bleed correction value from the desired channel
                if self.correct_channel == ':red[Red]':
                    for i in range(len(flat_img[0])):
                            flat_img[0][i] = flat_img[0][i] - self.bleed_correction
                elif self.correct_channel == ':green[Green]':
                    for i in range(len(flat_img[1])):
                            flat_img[1][i] = flat_img[1][i] - self.bleed_correction
                else:
                    for i in range(len(flat_img[2])):
                            flat_img[2][i] = flat_img[2][i] - self.bleed_correction
                            
                # Set all negative numbers to 0
                for l in flat_img:
                    for i in range(len(l)):
                        if l[i] < 0:
                            l[i] = 0      
                
                # Plot each channel vs. the others and get pearson coeffs
                st.markdown('<p class = "green-font">Intensity Correlation Plots</p>', unsafe_allow_html = True) 
                coloc_analysis = list(self.pearson_coloc(flat_img[0], flat_img[1], flat_img[2]))
                coloc_analysis = [round(x,2) for x in coloc_analysis]

                # Change any p value <0.005 to "<0.005" for easier interpreation
                # Involves changing every other element in list up to the last p value (element 5 so use 6 in the slice)
                coloc_analysis[1:6:2] = ['<0.005' if x < 0.005 else x for x in coloc_analysis[1:6:2]]
                # st.write(coloc_analysis)
                
                # Create strings for confidence limits
                g_r_conf = str(coloc_analysis[6]) + ' to ' + str(coloc_analysis[7])
                r_b_conf = str(coloc_analysis[8]) + ' to ' + str(coloc_analysis[9])
                b_g_conf = str(coloc_analysis[10]) + ' to ' + str(coloc_analysis[11])
                
                # Create a dictionary and dataframe for the pearson analysis
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
                
                # Generates the table of results
                self.create_table(pearson_df, 'Pearson Colocalization Results')
                     
                # Calculate Spearman's rank correlation and return as list of lists
                spearman_analysis = list(self.spearman(flat_img[0], flat_img[1], flat_img[2]))
                
                spearman_r = [spearman_analysis[0][0], spearman_analysis[1][0], spearman_analysis[2][0]]
                spearman_r = [round(x,2) for x in spearman_r]
                spearman_p = [spearman_analysis[0][1], spearman_analysis[1][1], spearman_analysis[2][1]]
                
                spearman_p = [round(x,2) for x in spearman_p]
                # Change any p value <0.005 to "<0.005" for easier interpreation
                spearman_p = ['<0.005' if x < 0.005 else x for x in spearman_p]
        
                # Spearman dict and dataframe
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
                
                # Generates the table of results
                self.create_table(spearman_df, 'Spearman Rank Colocalization Results')
                
                # Calculate Mander's M1 and M2 coefficients
                manders = self.manders(flat_img)
                self.manders_df = pd.DataFrame({'Manders Coeffs':['M1', 'M2'], 
                                                'Red-Green':[round(manders[0],3),round(manders[1],3)], 
                                                'Red-Blue':[round(manders[2],3),round(manders[3],3)],
                                                'Green-Blue':[round(manders[4],3), round(manders[5],3)]})
        
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
        
        # color = ('b','g','r')
        color = ('r','g','b')
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
    """Counts cells and determines live vs. dead cells based upon the amount of cy5 fluorescence.
        Cy5 is used as the live-dead indicator, selected as either the red or green channels depending on how the image was acquired. 
        Dead cells have much more cy5 than live cells."""
    
    def __init__(self):
         
        super().__init__() 
        
    def contour(self, crop):
        
        try: 
            if self.cell_count:
                st.write('')
                st.write('')
                st.markdown('<p class = "purple-font-venti">Live vs. dead cell analysis.</p>', unsafe_allow_html = True) 
                
                # Grab the cropped image and convert to gray scale
                self.img = crop.copy()
                # Convert to grayscale in prep for findContours
                self.gray_img  = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
                # Smooth the image
                self.gray_img = cv.GaussianBlur(self.gray_img, (5, 5), 0)
                # Thresh the image to convert all pixels to 0 or 255
                ret, self.thresh = cv.threshold(self.gray_img, self.threshold, 255, cv.THRESH_BINARY)
                self.thresh = cv.erode(self.thresh, None, iterations=2)
                self.thresh = cv.dilate(self.thresh, None, iterations=2)
                
                # CHAIN_APPROX_SIMPLE throws out some points so the contour around the edge will be a dotted line
                # contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                self.contours, self.hierarchy = cv.findContours(self.thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                
        except:
            self.contours = np.array([])
     
        # return self.contours
    
    def cell_stats(self):
        # Calculates a number of features of each individual cell found in the image such as mean RGB intensities in each cell
        try:
            # Create dicts for cell parameters
            if self.cell_count:
                
                self.contour_dict = {}
                self.areaDict = {}
                self.aspectRatioDict = {}
                self.avgColorDict = {}
                self.perimeterDict = {}
                self.redMeanDict = {}
                self.greenMeanDict = {}
                self.blueMeanDict = {}
                
                cell_number = 0
                
                for contour in self.contours:
                    
                    # Set a minimum area for a cell to be counted
                    if cv.contourArea(contour) > self.min_contour_area:
                       
                        cell_number += 1
                        
                        # Add contour to dict
                        self.contour_dict[cell_number] = contour
                
                        # Fit an elipse to each cell
                        ellipse = cv.fitEllipse(contour)
                        # A more direct route to the major and minor axis distances
                        (x,y), (MA, ma), angle = cv.fitEllipse(contour)
                        cv.ellipse(self.img, ellipse, (255,255,255), 2, cv.LINE_AA)
                        
                        # Draw the contour that follows the cell perfectly
                        cv.drawContours(self.img, contour, -1, (0, 255, 255), 2)
                        
                        # Get the cell area
                        self.areaDict[cell_number] = cv.contourArea(contour)
                        
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
                        self.redMeanDict[cell_number] = self.mean_intensities[0]
                        self.greenMeanDict[cell_number] = self.mean_intensities[1]
                        self.blueMeanDict[cell_number] = self.mean_intensities[2]
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
                        self.aspectRatioDict[cell_number] = self.aspect_ratio
                        
                        self.perim = cv.arcLength(contour, True)
                        self.perimeterDict[cell_number] = self.perim
                        
                        # Label every cell with a number
                        M = cv.moments(contour)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv.putText(self.img, text = str(cell_number), org=(cx,cy),
                                fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255),
                                thickness = 2, lineType=cv.LINE_AA)
                
            # st.write(self.contour_dict)
            # st.stop()
            
            # Print the threshold image
            # st.image(self.thresh, use_column_width = True, clamp = True)
            
            # Print the real image with contours and labels
            st.image(self.img, use_column_width = True, clamp = True)
            
            # Download the image with the contours marked
            self.download_image(self.img, "Download Cell Contours")
            
            # Get values from dictionaries to append to tables
            number_of_cells = [x for x in range(1, cell_number + 1)]
            aspectRatio_list = [value for key, value in self.aspectRatioDict.items()]
            red_list = [value for key, value in self.redMeanDict.items()]
            green_list = [value for key, value in self.greenMeanDict.items()]
            blue_list = [value for key, value in self.blueMeanDict.items()]
            # area_list = [value for key, value in self.areaDict.items()]
            
            st.write('')
            st.write('')
            live_dead_threshold = st.slider('Select an intensity threshold for Live-Dead Analysis', 50, 255, 100, 10)
            hpa_color_radio = st.radio('HPA Color',[':red[Red]', ':green[Green]'], horizontal = True)
            
            # Create a live-dead list
            if hpa_color_radio == ':red[Red]':
                self.live_dead_list = ['Dead' if intensity > live_dead_threshold else 'Live' for intensity in red_list]
               
            else:
                self.live_dead_list = ['Dead' if intensity > live_dead_threshold else 'Live' for intensity in green_list]
           
            self.cell_df = pd.DataFrame({"Cell Number":number_of_cells, 
                                          # "Aspect Ratio":aspectRatio_list, 
                                          "Mean Red Intensity":red_list, 
                                          "Mean Green Intensity":green_list,
                                          "Mean Blue Intensity":blue_list,
                                          "Cell Status":self.live_dead_list,
                                          })
            
            # st.write(self.cell_df)
            # st.stop()
            
            # Render a plotly figure (pass in headers and then values as dicts)
            fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1,1], header = dict(values = list(
                                                    self.cell_df[['Cell Number', 
                                                                  'Mean Red Intensity', 
                                                                  'Mean Green Intensity', 
                                                                  'Mean Blue Intensity', 
                                                                  'Cell Status',
                                                                  ]].columns),fill_color = '#FD8E72',
                                                    line_color='darkslategray', align = 'center', 
                                                    font = dict(color = 'blue', size = 18)), 
                                                    cells = dict(values = [self.cell_df['Cell Number'], 
                                                                            # self.cell_df['Aspect Ratio'], 
                                                                            self.cell_df['Mean Red Intensity'], 
                                                                            self.cell_df['Mean Green Intensity'], 
                                                                            self.cell_df['Mean Blue Intensity'],
                                                                            self.cell_df['Cell Status']
                                                                            ],
                                                    fill_color = 'lavender', align = 'center', line_color = 'darkslategray',
                                                    height = 30)))
            # Other styles are set here
            fig.update_traces(cells_font = dict(size = 15))
            # In update layout both the margin and height control the spacing between tables in the vertical direction. 
            fig.update_layout(title = "Cell Analysis", title_font = dict(size = 18, color = 'green',
                    family = 'Arial'), title_x = 0, title_y = 1.0, height = 300, width = 1000,
                              margin = dict(l = 0, r = 0, b = 0, t = 35))
            st.write('')
            st.write('')
            st.write(fig)      
            
            # Button to download dataframe as csv file
            self.live_dead_button = st.download_button(label = 'Download', data = self.cell_df.to_csv(), file_name = 'live_dead.csv', mime = 'text/csv')
            
            # Identify 2 or more cells that are so close together that they are counted as one cell
            double_cell_list = []
            for ar in aspectRatio_list:
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
            fig.update_layout(title = "All Cell Totals", title_font = dict(size = 18, color = 'green',
                    family = 'Arial'), title_x = 0, title_y = 1.0, height = 100, width = 1000,
                              margin = dict(l = 0, r = 0, b = 0, t = 35))
            
            st.write('')
            st.write('')
            st.write(fig) 
            
            # Button to download dataframe as csv file
            self.summary_button = st.download_button(label = 'Download', data = self.cell_totals_df.to_csv(), file_name = 'live_dead.csv', mime = 'text/csv', key = 'summary')

        except:
            pass
        
    def total_intensity(self):
        # Calculates the total RGB intensities for each live cell and calculates the mean/cell and std/cell
        
        if self.intensity:
        
            # Remove the contours of the dead cells from self.contour_dict
            # Get cell numbers for dead cells
            # Filter self.cell_df by dead cell
            self.dead_df = self.cell_df[self.cell_df['Cell Status'] == 'Dead']
            self.dead_list = self.dead_df['Cell Number'].tolist()
            # Remove those entries from the contour dict
            self.live_cntr_dict = {key:value for key,value in self.contour_dict.items() if key not in self.dead_list}
            # st.write(self.live_cntr_dict.keys())
            # st.write(self.live_cntr_dict.values())
            # st.write(len(self.live_cntr_dict))
            # st.stop()
            
            # Create dicts for cell intensities
            self.red_intensity_dict = {}
            self.green_intensity_dict = {}
            self.blue_intensity_dict = {}
            self.red_per_live_dict = {}
            self.green_per_live_dict = {}
            self.blue_per_live_dict = {}
            
            # Start a counter for # of cells
            cell_number = 0
        
            for contour in self.live_cntr_dict.values():

                # Set a minimum area for a cell to be counted
                # if cv.contourArea(contour) > self.min_contour_area:
                cell_number += 1
                
                # Get RGB intensities for every pixel inside all the contours
                # for i in range(len(contour)):
                # Create a mask image that contains the contour filled in
                mask = np.zeros_like(self.img[:, :, 0])
                # Fill in the contours. The -1 fills in all contours.
                cv.drawContours(mask, [contour], -1, 255, thickness = cv.FILLED)

                red_sum = np.sum(self.img[:, :, 0][mask == 255])
                green_sum = np.sum(self.img[:, :, 1][mask == 255])
                blue_sum = np.sum(self.img[:, :, 2][mask == 255])
               
                self.red_intensity_dict[cell_number] = red_sum
                self.green_intensity_dict[cell_number] = green_sum
                self.blue_intensity_dict[cell_number] = blue_sum
                        
            # st.write(self.red_intensity_dict)
            # st.write(self.green_intensity_dict)
            # st.write(self.blue_intensity_dict)
            
            # Calculate the average and standard deviations
            # Put dict values in lists
            self.red_list = [value for value in self.red_intensity_dict.values()]
            self.green_list = [value for value in self.green_intensity_dict.values()]
            self.blue_list = [value for value in self.blue_intensity_dict.values()]
            self.red_mean = round(np.mean(self.red_list))
            self.green_mean = round(np.mean(self.green_list))
            self.blue_mean = round(np.mean(self.blue_list))
            self.red_std = round(np.std(self.red_list))
            self.green_std = round(np.std(self.green_list))
            self.blue_std = round(np.std(self.blue_list))
            
            # st.write(self.red_mean)
            # st.write(self.red_std)
            
            # If the sum of all cells in of interest
            # self.RGB_intensity_df = pd.DataFrame({'Live Cells':[str(len(self.live_cntr_dict))],
            #                                       'Red':[sum(self.red_intensity_dict.values())],
            #                                       'Green':[sum(self.green_intensity_dict.values())],
            #                                       'Blue':[sum(self.blue_intensity_dict.values())]})    
            
            # Use mean and std of all cells. This will enable statistical differences between two cell images to be calculated
            self.RGB_intensity_df = pd.DataFrame({'Live Cells':[str(len(self.live_cntr_dict))]})  
            
            self.RGB_intensity_df['Mean Red/Cell'] = round(self.red_mean)
            self.RGB_intensity_df['STD Red/Cell'] = round(self.red_std)
            
            self.RGB_intensity_df['Mean Green/Cell'] = round(self.green_mean)
            self.RGB_intensity_df['STD Green/Cell'] = round(self.green_std)
            
            self.RGB_intensity_df['Mean Blue/Cell'] = round(self.blue_mean)
            self.RGB_intensity_df['STD Blue/Cell'] = round(self.blue_std)
            
            # st.write(self.RGB_intensity_df)
            # st.stop()
  
            fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1,1], header = dict(values = list(
                                                                  self.RGB_intensity_df[[
                                                                  'Live Cells', 
                                                                  'Mean Red/Cell',
                                                                  'STD Red/Cell',
                                                                  'Mean Green/Cell',
                                                                  'STD Green/Cell',
                                                                  'Mean Blue/Cell',
                                                                  'STD Blue/Cell'
                                                                  ]].columns),
                                                                  fill_color = '#FD8E72', line_color = 'darkslategray', align = 'center', 
                                                                  font = dict(color = 'blue', size = 14)), 
                                                                  cells = dict(values = 
                                                                           [self.RGB_intensity_df['Live Cells'], 
                                                                            self.RGB_intensity_df['Mean Red/Cell'], 
                                                                            self.RGB_intensity_df['STD Red/Cell'],
                                                                            self.RGB_intensity_df['Mean Green/Cell'],
                                                                            self.RGB_intensity_df['STD Green/Cell'],
                                                                            self.RGB_intensity_df['Mean Blue/Cell'],
                                                                            self.RGB_intensity_df['STD Blue/Cell']],
                                                    fill_color = 'lavender', align = 'center', line_color = 'darkslategray', height = 30)))
    
            fig.update_traces(cells_font = dict(size = 15))
            # In update layout both the margin and height control the spacing between tables in the vertical direction. 
            fig.update_layout(title = "Total Live Cell RGB Intensities", title_font = dict(size = 18, color = 'green',
                    family = 'Arial'), title_x = 0, title_y = 1.0, height = 100, width = 1000,
                              margin = dict(l = 0, r = 0, b = 0, t = 35))
            
            st.write('')
            st.write('')
            st.write(fig) 
            
            # Button to download dataframe as csv file
            self.live_cell_analysis_button = st.download_button(label = 'Download', data = self.RGB_intensity_df.to_csv(), file_name = 'live_cell_RGB.csv', mime = 'text/csv', key = 'live')

    

# Run 
if __name__ == '__main__':

    # Call Cell_Counter class with all the classes and methods above it inherits 
    obj1 = Cell_Counter()
    header = obj1.header()
    sidebar = obj1.sidebar()
    upload = obj1.upload_image()
    crop = obj1.crop_denoise()
    display = obj1.display_images()
    # Pass crop so the histo method inherits self.image_array
    histo = obj1.histo(crop)
    gaussian = obj1.gaussian_test()
    colocAnalysis = obj1.colocalizationAnalysis()
    contour = obj1.contour(crop)
    stats = obj1.cell_stats()
    if obj1.cell_count:
        intensities = obj1.total_intensity()
    


    


    
    
    
    
    
    
    