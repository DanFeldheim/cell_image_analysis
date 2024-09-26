#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:11:49 2024

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


class Flow_Control():
    """This class makes all of the calls to other classes and methods."""
    
    def __init__(self, state_dict):
        
        self.state_dict = state_dict
        st.markdown('<p class = "small-font">Velocity Sciences Image Processor</p>', unsafe_allow_html = True) 
         
    def all_calls(self):
        # Set up the app sidebar
        calls = Setup(state_dict).sidebar()
        # Make calls to load an image, crop, and enhance the image
        prep = Prep_Image(state_dict)
        upload = prep.upload_image()
        crop = prep.cropping() 
        denoise = prep.denoise()
        enhance = prep.enhance_details(self.state_dict['denoised cropped image array'])
        bright = prep.brighten(self.state_dict['enhanced image array'])
        display = prep.display_images()
        thresh_img = prep.threshold_image(self.state_dict['denoised cropped image array'])
        split = prep.RGB_split(self.state_dict['denoised cropped image array'])
        
        # Count cells, determine live from dead cells, and analyze RGB intensity 
        if self.state_dict['cell_count'] == True:
            cell_counter = Cell_Counter(state_dict)
            count = cell_counter.contour()
            draw_contours = cell_counter.cell_stats()
            total_RGB = cell_counter.total_intensity()
        
        # Perform a gaussian analysis of RGB intensities. This is important for performing a PearsonR colocalization analysis
        if self.state_dict['gaussian'] == True:
            gauss = Gauss_Test(state_dict)
            gauss_test = gauss.gaussian_test()
        
        # Analyze for colocalization
        if self.state_dict['colocalization'] == True:
            
            # Apply bleed corrections
            coloc = Colocalization(state_dict)
            box_height = coloc.box_height()
            
            if self.state_dict['total red - red from green']:
                min_index_vals = coloc.get_index_vals(box_height['green_box'], 'green', 'red')
                
            if self.state_dict['total green - green from red']:
                min_index_vals = coloc.get_index_vals(box_height['red_box'], 'red', 'green')
                
            if self.state_dict['total blue - blue from green']:
                min_index_vals = coloc.get_index_vals(box_height['green_box'], 'green', 'blue')
             
            if self.state_dict['total green - green from blue']:
                min_index_vals = coloc.get_index_vals(box_height['blue_box'], 'blue', 'green')
                
            if self.state_dict['total blue - blue from red']:
                min_index_vals = coloc.get_index_vals(box_height['red_box'], 'red', 'blue')
                
            if self.state_dict['total red - red from blue']:
                min_index_vals = coloc.get_index_vals(box_height['blue_box'], 'blue', 'red')
            
            # Start a plot layout and fill with either raw intensity data or intensity data corrected for bleed
            pearson_plot_layout = coloc.pearson_plot_layout()    
            
            # To correct for bleed
            try: 
                # If a bleed correction box was checked so that min_index_vals exists, do a bleed correction
                if min_index_vals:
                    # Throw an error if a bleed correction slope can't be fit.
                    try:
                        bleed_data = coloc.bleed_dataframe(min_index_vals)
                        bleed_slope = coloc.linear_fit(bleed_data)
                    except:
                        st.error('The color channel you just selected cannot be corrected. \
                                  Please deselect that channel and choose another channel to correct.')
                    # Assuming a bleed correction slope could be calculated, do the correction
                    bleed_sub = coloc.bleed_subtraction(bleed_slope)
                    # Generate lists of data to plot
                    pearson_corrected_data_plots = coloc.pearson_plot_corrected_data()
                    # Plot the data. Note that no matter what correction was selected, the first element in
                    # the list is always RGorB, either the corrected or uncorrected versions
                    pearson_insert_data = coloc.pearson_plot_data(RG_list = pearson_corrected_data_plots[0],
                                                                  RB_list = pearson_corrected_data_plots[1],
                                                                  GB_list = pearson_corrected_data_plots[2])
                    
                    # Plot the data that was used to find the slope for the bleed correction
                    bleed_plots = coloc.bleed_correction_plots(bleed_data)
                   
            # No bleed correction
            except:
                # If no bleed correction is needed, generate intensity plots using the flat RGB 
                # lists in self.state_dict
                pearson_plot_layout = coloc.pearson_plot_layout()
                pearson_insert_data = coloc.pearson_plot_data(RG_list = [self.state_dict['flat red'], 
                                                                             self.state_dict['flat green']],
                                                              RB_list = [self.state_dict['flat red'], 
                                                                             self.state_dict['flat blue']], 
                                                              GB_list = [self.state_dict['flat blue'], 
                                                                             self.state_dict['flat green']])
         
      
            # Return a list of pearson analysis values
            pearson_stats = coloc.pearsonr(pearson_insert_data)
            # Display the values in a table
            pearsonr_table = coloc.pearsonr_spearmanr_results_table(pearson_stats, "Pearson Colocalization Results")
            # Do the same for spearmanr (pearson_insert_data is just the intensity data so don't worry about the name)
            spearman_stats = coloc.spearmanr(pearson_insert_data)
            pearsonr_table2 = coloc.pearsonr_spearmanr_results_table(spearman_stats, "Spearman Colocalization Results")
            # Calculate Manders coefficients. Pass in the RGB lists
            manders_coeff = coloc.manders(pearson_insert_data)
            manders_table = coloc.manders_table(manders_coeff)
            
   
class Setup():
    """
    Instantiates the working directory and markdown styles. 
    Creates a header and sidebar for the app and an upload button for data. 
    Also contains miscillaneous methods that are called from other classes throughout the app.
    """
    
    # Set fonts and basic layout.
    def __init__(self, state_dict):
       
        self.state_dict = state_dict
        
        # Create a Working Directory
        self.state_dict['directory'] = "/Users/danfeldheim/Documents/cell_image_pro_app/"
        
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
        
    def sidebar(self):
        
        with st.sidebar:
            self.brightness = st.slider("Brightness", min_value = -50, max_value = 50, value = 0)
            self.state_dict["brightness"] = self.brightness
            
            # Adjustable imaging smoothing parameters
            self.sigma_s = st.number_input("Sigma S", value = 10, min_value = 1, max_value = 30)
            self.state_dict["sigma_s"] = self.sigma_s
            
            self.sigma_r = st.number_input("Sigma R", value = 0.05, min_value = 0.01, max_value = 0.2)
            self.state_dict["sigma_r"] = self.sigma_r
            
            # Select box for color channels to remove from the image
            self.color = st.multiselect("Choose colors to remove.", ["Green", "Red", "Blue"])
            self.state_dict["color"] = self.color
            
            # Plot color channel histogram 
            # self.color_histogram = st.checkbox(label = 'Plot color histogram')
            # self.state_dict["histogram"] = self.color_histogram
            
            # Perform a gaussian analysis
            self.color_gaussian = st.checkbox(label = 'Gaussian Analysis')
            self.state_dict["gaussian"] = self.color_gaussian
            
            # Count cells
            self.cell_count = st.checkbox(label = 'Live-Dead Count')
            self.state_dict["cell_count"] = self.cell_count
            
            if self.state_dict['cell_count'] == True:
                
                self.min_contour_area = st.number_input("Minimum Cell Area", value = 150, 
                                                        min_value = 2, max_value = 500)
                
                self.state_dict["contour_area"] = self.min_contour_area
                
                self.threshold = st.number_input("Threshold", value = 20, min_value = 5, 
                                                 max_value = 100)
                
                self.state_dict["threshold"] = self.threshold
                # self.intensity = st.checkbox(label = 'Total RGB Intensity')
                # self.state_dict["intensity"] = self.intensity
                
            # Perform a colocalization analysis
            self.colocalization = st.checkbox(label = 'Colocalization')
            self.state_dict["colocalization"] = self.colocalization
           
            if self.state_dict["colocalization"] == True:
                
                with st.expander('Auto Bleed Correction'):
            
                    st.write('Subtract:')
                    self.total_red_red_from_green = st.checkbox(label = ':green[Green] - :red[Red]')           
                    self.state_dict['total red - red from green'] = self.total_red_red_from_green
                    
                    self.total_green_green_from_red = st.checkbox(label = ':red[Red] - :green[Green]')           
                    self.state_dict['total green - green from red'] = self.total_green_green_from_red
                    
                    self.total_blue_blue_from_green = st.checkbox(label = ':green[Green] - :blue[Blue]')           
                    self.state_dict['total blue - blue from green'] = self.total_blue_blue_from_green
                    
                    self.total_green_green_from_blue = st.checkbox(label = ' :blue[Blue] - :green[Green]')           
                    self.state_dict['total green - green from blue'] = self.total_green_green_from_blue
                    
                    self.total_blue_blue_from_red = st.checkbox(label = ':red[Red] - :blue[Blue]')           
                    self.state_dict['total blue - blue from red'] = self.total_blue_blue_from_red
                    
                    self.total_red_red_from_blue = st.checkbox(label = ':blue[Blue] - :red[Red]')           
                    self.state_dict['total red - red from blue'] = self.total_red_red_from_blue
    
    # This method is a general table generator                      
    def create_table(self, df, title):
        
        fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1], 
                                        header = dict(values = list(df[['Statistic', 'Green-Red', 
                                        'Blue-Red', 'Green-Blue']].columns),fill_color = '#FD8E72',
                                        line_color ='darkslategray', align = 'center', 
                                        font = dict(color = 'blue', size = 18)), 
                                        cells = dict(values = [df['Statistic'], 
                                        df['Green-Red'], df['Blue-Red'], 
                                        df['Green-Blue']],
                                        fill_color = 'lavender', 
                                        align = 'center', 
                                        line_color = 'darkslategray',
                                        height = 30)))
                                                                                          
        
        fig.update_traces(cells_font = dict(size = 15))
        
        # In update layout both the margin and height control the spacing between tables in the vertical direction. 
        fig.update_layout(title = title, 
                          title_font = dict(size = 18, 
                                            color = 'green',
                                            family = 'Arial'), 
                          title_x = 0, 
                          title_y = 0.95,
                          height = 185,
                          margin = dict(l = 0, r = 0, b = 0, t = 30))
     
        st.write(fig)    
        
    # def download_image(self, image, btn_text):
    """Download method. Would rather have this here, but it creates a lot of padding above 
    when this class method is called from another class for some reason."""
    #     pilImage = Image.fromarray(image)
    #     buffer = io.BytesIO()
    #     pilImage.save(buffer, format = "PNG")
    #     btn = st.download_button(
    #         label = btn_text,
    #         data = buffer,  
    #         file_name = "image.png",
    #         mime = "image/png",
    #         use_container_width = False
    #     ) 


class Prep_Image():        
    """Class to upload and perform image processing"""
    
    def __init__(self, state_dict):
        
        self.state_dict = state_dict
      
    # Create an upload button for data   
    def upload_image(self):
        
        st.write('')
        notification_box(icon='warning', 
                         title='Warning', 
                         textDisplay = 'The file uploaded must be in jpg, png, or jpeg format', 
                         styles = None, 
                         externalLink = None, 
                         url = None, 
                         key ='foo')
        
        self.image_file = st.file_uploader('Choose an image', 
                                           type = ['jpg', 'png', 'jpeg', 'tiff'])
        
        if not self.image_file:
            st.stop()
        
        # Import image
        self.image = Image.open(self.image_file)
        self.image = self.image.convert('RGB')
        self.image_array = np.array(self.image)
        self.state_dict['original image'] = self.image
        self.state_dict['original np image'] = self.image_array
        
    def cropping(self):
        
        st.markdown('<p class = "purple-font-venti">Use the cropper box to select an area to process.</p>', 
                    unsafe_allow_html = True) 
        self.cropped_img = st_cropper(self.state_dict['original image'], 
                                      realtime_update = True, 
                                      box_color = '#0000FF',
                                      aspect_ratio = (1,1))
        
        # Convert to numpy array
        self.cropped_img_array = np.array(self.cropped_img)
        self.state_dict['cropped image'] = self.cropped_img
        self.state_dict['cropped image array'] = self.cropped_img_array
        
    def denoise(self):
        """Denoises the image using opencv's color image non-local means denoiser: 
            https://www.javatpoint.com/fastnlmeansdenoising-in-python"""
        
        # Denoise the image
        self.denoised_img_array = cv.fastNlMeansDenoising(self.state_dict['cropped image array'], 
                                                          None, 15, 7, 21)  
        
        self.state_dict['denoised cropped image array'] = self.denoised_img_array
    
   # Applies an image enhancer
    def enhance_details(self, img_array):
        
        self.enhanced_img_array = cv.detailEnhance(img_array, 
                                                   sigma_s = self.state_dict["sigma_s"], 
                                                   sigma_r = self.state_dict["sigma_r"])
        
        self.state_dict['enhanced image array'] = self.enhanced_img_array
        self.state_dict['enhanced image'] = Image.fromarray(self.enhanced_img_array, 'RGB')
    
    def brighten(self, img_array):
        
        self.brightened_image_array = cv.convertScaleAbs(img_array, 
                                                         beta = self.state_dict['brightness']) 
        
        self.state_dict['brightened image array'] = self.brightened_image_array
        
        self.state_dict['brightened image'] = Image.fromarray(self.brightened_image_array, 'RGB')
         
    # Displays cropped image
    def display_images(self):
        
        col1, col2 = st.columns([1,1])
        
        with col1:
            
            # Display images
            st.write('')
            st.markdown('<p class = "green-font-sm">Cropped Image</p>', unsafe_allow_html = True) 
            
            if self.state_dict["color"]:
                # Delete one or 2 color channels from the image and display
                # Extract colors
                self.color_deletion_array = self.color_picker()
                
                # Convert to pil image to display
                self.color_deletion_img = Image.fromarray(self.color_deletion_array, 'RGB')
                
                color_deletion_bigger = cv.resize(self.color_deletion_array, (900, 900))
                
                st.image(color_deletion_bigger)
                
                # st.image(self.color_deletion_img)
                # Back to np array to download
                self.color_deletion_array = np.array(self.color_deletion_img)
                
                # down = Setup(state_dict).download_image(self.color_deletion_array, "Download False-Colored Image")
                self.download_image(self.color_deletion_array, "Download False-Colored Image")
                
            else:
                cropped_image_bigger = cv.resize(self.state_dict['cropped image array'], (900, 900))
                
                st.image(cropped_image_bigger)
                # st.image(self.state_dict['cropped image'])
                # down = Setup(state_dict).download_image(self.state_dict['cropped image array'], "Download Cropped Image")
                
                self.download_image(self.state_dict['cropped image array'], "Download Cropped Image")
                
        with col2:
            st.write('')
            st.markdown('<p class = "green-font-sm">Enhanced Image</p>', unsafe_allow_html = True) 
            
            enhanced_bigger = cv.resize(self.state_dict['brightened image array'], (900, 900))
            
            st.image(enhanced_bigger)
            # st.image(self.state_dict['brightened image'])
            # download = Setup(state_dict).download_image(self.state_dict['brightened image array'], "Download Enhanced Image")
            
            self.download_image(self.state_dict['brightened image array'], "Download Enhanced Image")
        
    def color_picker(self):
        """Function to remove 1 or 2 colors from the image. Hard coded to apply to the cropped image only."""
  
        for color in self.state_dict['color']:
            if color == 'Red':
                self.state_dict['cropped image array'][...,0] *= 0
            elif color == 'Green':
                self.state_dict['cropped image array'][...,1] *= 0
            elif color == 'Blue':
                self.state_dict['cropped image array'][...,2] *= 0

        return self.cropped_img_array

    def threshold_image(self, img_array):
        
        try:
            # Convert to grayscale in prep for findContours
            self.gray_img  = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            # Smooth the image
            self.gray_img = cv.GaussianBlur(self.gray_img, (5, 5), 0)
            # Thresh the image to convert all pixels to 0 or 255
            
            ret, self.thresh = cv.threshold(self.gray_img, 
                                            self.state_dict['threshold'], 255, cv.THRESH_BINARY)
            
            self.thresh = cv.erode(self.thresh, None, iterations=2)
            
            self.thresh = cv.dilate(self.thresh, None, iterations=2)
            
            self.state_dict['threshold image'] = self.thresh  
            
        except KeyError:
            pass
        
    def RGB_split(self, img_array):
        
        red = img_array[:, :, 0]   
        green = img_array[:, :, 1]
        blue = img_array[:, :, 2]
        
        self.state_dict['red_channel'], self.state_dict['green_channel'], self.state_dict['blue_channel'] = red, green, blue
    
        red_flat = red.flatten()
        green_flat = green.flatten()
        blue_flat = blue.flatten()
       
        self.state_dict['flat red'], self.state_dict['flat green'], self.state_dict['flat blue'] = red_flat.tolist(), green_flat.tolist(), blue_flat.tolist()
        flat_rgb_df = pd.DataFrame({'red': self.state_dict['flat red'], 
                                    'green': self.state_dict['flat green'], 
                                    'blue': self.state_dict['flat blue']})
        
        self.state_dict['flat RGB df'] = flat_rgb_df
        
    def download_image(self, image, btn_text):
        """Wanted to put this in the Setup class but it created a 
        lot of padding above the button for some reason.
        """
        
        pilImage = Image.fromarray(image)
        buffer = io.BytesIO()
        pilImage.save(buffer, format = "PNG")
        btn = st.download_button(
            label = btn_text,
            data = buffer,  
            file_name = "image.png",
            mime = "image/png",
            use_container_width = False
        ) 
               
        
class Gauss_Test():
    """Class to perform gaussian curve analysis"""
     
    def __init__(self, state_dict):
         
             self.state_dict = state_dict

    # Several tests for a gaussian distribution are included here
    # This is needed because the pearsonr test of colocalization requires a gaussian distribution
    # The Spearman test does not
    def gaussian_test(self):
       
        st.write('')
        st.write('')
        # Box plots of the rgb channels to check for normal distribution
        # If median line isn't in the middle or if the whiskers aren't the same size, the distribution isn't normal
        # and pearson's correlation test isn't valid
        st.markdown('<p class = "purple-font-venti">Normal Distribution Tests</p>', 
                    unsafe_allow_html = True) 
        
        st.markdown('<p class = "green-font">Box Plots</p>', 
                    unsafe_allow_html = True) 
        
        fig = plt.figure(figsize = (7, 2))
      
        box = self.state_dict['flat RGB df'].boxplot(fontsize = 7, grid = False)
        plt.xlabel('Channel', fontsize = 8)
        plt.ylabel('Intensity', fontsize = 8)
        st.pyplot(fig)
        
        # Q-Q plots
        st.markdown('<p class = "green-font">Q-Q Plots</p>', unsafe_allow_html = True) 
        fig, axs = plt.subplots(1,3, figsize = (7, 2))
        stats.probplot(np.array(self.state_dict['flat red']), 
                       dist = 'norm', plot = axs[0])
        
        axs[0].set_title('Red Channel')
        
        stats.probplot(np.array(self.state_dict['flat green']), 
                       dist = 'norm', plot = axs[1])
        
        axs[1].set_title('Green Channel')
        stats.probplot(np.array(self.state_dict['flat blue']), 
                       dist = 'norm', plot = axs[2])
        
        axs[2].set_title('Blue Channel')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Shapiro-Wilk test
        st.write('')
        st.markdown('<p class = "green-font">Shapiro-Wilk Test</p>', unsafe_allow_html = True) 
        shapiro_red = stats.shapiro(self.state_dict['flat red'])
        shapiro_green = stats.shapiro(self.state_dict['flat green'])
        shapiro_blue = stats.shapiro(self.state_dict['flat blue'])
       
        if shapiro_red.pvalue or shapiro_green.pvalue or shapiro_blue.pvalue <0.05:
            st.markdown(''':red[Warning: The Shapiro-Wilk test indicates that 
                        the intensities of at least one channel are not normally distributed.      
                        (p values < 0.05 mean the distribution is not normal.)  
                        In this case, the Pearson analysis of colocalization may not be 
                        as reliable as the Spearman analysis.]''')
                        
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


class Colocalization():        
    """Class to perform colocalization analysis"""
     
    def __init__(self, state_dict):
         
        self.state_dict = state_dict
        
        # Create dicts to store results of method calls
        self.min_index_val_dict = {}
        self.bleed_dict = {}
        self.slope_dict = {}
        self.bleed_subtraction_dict = {}
        self.all_possibilities_list = ['total red - red from green',
                                       'total green - green from red',
                                       'total blue - blue from green',
                                       'total green - green from blue',
                                       'total blue - blue from red',
                                       'total red - red from blue']

        st.write('')
        st.write('')
        st.markdown('<p class = "purple-font-venti">Colocalization Analysis</p>',
                    unsafe_allow_html = True) 

    def box_height(self):
       
        # Get the n pts closest to the axis effected by bleed, i.e., n points close to the y axis
        # Determine the bin height along the y axis to look for data points
        # The bins are n boxes from 0 to the largest y value. 
        # One data point-the smallest on the x axis will be extracted from each bin. 
        red_boxHeight = round(self.state_dict['flat RGB df'].max()[0]/6)
        green_boxHeight = round(self.state_dict['flat RGB df'].max()[1]/6)
        blue_boxHeight = round(self.state_dict['flat RGB df'].max()[2]/6)
        
        self.state_dict['red_box'] = red_boxHeight
        self.state_dict['green_box'] = green_boxHeight 
        self.state_dict['blue_box'] = blue_boxHeight
        
        return self.state_dict
    
    def get_index_vals(self, box_height, color1, color2):
        
        index_vals_list = []
        
        for i in range(0,5):
            
            low_val = int(i*box_height)
            high_val = int((i+1)*box_height)
            
            # Get all rows where the color intensity is between y_low and y_high
            # Create a temp df of points within the bin
            temp_df = self.state_dict['flat RGB df'][self.state_dict['flat RGB df'][color1].between(low_val, high_val)]
            # Get the index value where the color of interest (e.g., red) is the lowest
            min_index = temp_df[color2].idxmin()
      
            index_vals_list.append(min_index)
        
        key_name = 'total ' + color2 + ' - ' + color2 + ' from ' + color1 + '_min_index'

        self.min_index_val_dict[key_name] = [index_vals_list, color1, color2]
       
        return self.min_index_val_dict
       
    def bleed_dataframe(self, min_index_vals):
       
        for key, value in min_index_vals.items(): 
       
            # Create a df of the green intensity corresponding to the minimum red intensity within the bin
            self.bleed_df = self.state_dict['flat RGB df'].loc[self.state_dict['flat RGB df'].index[value[0]]]
            
            key_name = key + '_bleed_df'
            
            self.bleed_dict[key_name] = [self.bleed_df, value[1], value[2]]
  
        return self.bleed_dict
            
    def linear_fit(self, bleed_df):      
        
        for key, value in bleed_df.items():
            
            # Get slope and intercept for those intensities
            slope, intercept = np.polyfit(value[0].loc[:, value[2]], value[0].loc[:, value[1]], 1, full = False)
            
            key_name = 'total ' + value[2] + ' - ' + value[2] + ' from ' + value[1] + '_slope'
           
            self.slope_dict[key_name] = [slope, value[1], value[2]]
            
        return self.slope_dict
    
    def bleed_subtraction(self, bleed_slope):
        
        for key, value in bleed_slope.items():
            
            bleed_sub_list = []
            
            # Get the bleed slope using the colors in the bleed_slope dict (need to convert colors to lowercase
            # and add the word flat to grab the correct data in the state_dict)
            bleed_slope, color1, color2 = value[0], 'flat ' + value[1], 'flat ' + value[2]
            
            for i,j in zip(self.state_dict[color2], self.state_dict[color1]):
                
                self.bleed_sub = round(i - (j/bleed_slope))
              
                if self.bleed_sub < 0:
                    bleed_sub_list.append(0)
                    
                else:
                    bleed_sub_list.append(self.bleed_sub)
            
            key_name = 'total ' + value[2] + ' - ' + value[2] + ' from ' + value[1] + '_subtraction'
           
            self.bleed_subtraction_dict[key_name] = [bleed_sub_list, color1, color2]
        
    def pearson_plot_layout(self):
        """Lays out a 1 row x 3 col grid of intensity plots for R v G v B"""
       
        # Create 3 plots
        self.fig, self.axs = plt.subplots(1,3, figsize = (7, 2))
        # Plot R vs G
        self.axs[0].set_ylim([0, 255])
        self.axs[0].set_xlim([0, 255])
        self.axs[0].set_title('Green vs. Red')
        # axs[0].set_xlabel('Red Channel')
        # axs[0].set_ylabel('Green Channel')
        self.axs[1].set_title('Blue vs. Red')
        self.axs[1].set_ylim([0, 255])
        self.axs[1].set_xlim([0, 255])
        # ax2.xlabel('Red Channel')
        # ax2.ylabel('Blue Channel')
        self.axs[2].set_title('Green vs. Blue')
        self.axs[2].set_ylim([0, 255])
        self.axs[2].set_xlim([0, 255])
        # ax3.xlabel('Blue Channel')
        # ax3.ylabel('Green Channel')
        
    def pearson_plot_corrected_data(self): 
        
        # Create lists with the x values and y values to be plotted
        # Begin by setting list values equal to uncorrected RGB values. If no correction was selected, these are plotted.
        RG_list = [self.state_dict['flat red'], self.state_dict['flat green']]
        RB_list = [self.state_dict['flat red'], self.state_dict['flat blue']]
        GB_list = [self.state_dict['flat blue'], self.state_dict['flat green']]
        
        # If a correction was requested, change the list
        # First 3 lines are Red, Green
        if {'total red - red from green_subtraction','total green - green from red_subtraction'} \
            <= self.bleed_subtraction_dict.keys():
            
            RG_list = [self.bleed_subtraction_dict['total red - red from green_subtraction'][0],
                       self.bleed_subtraction_dict['total green - green from red_subtraction'][0]]
            
        elif {'total red - red from green_subtraction'} <= self.bleed_subtraction_dict.keys():
           
            RG_list = [self.bleed_subtraction_dict['total red - red from green_subtraction'][0],
                       self.state_dict['flat green']]
             
        elif {'total green - green from red_subtraction'} <= self.bleed_subtraction_dict.keys():
             
            RG_list = [self.state_dict['flat red'],
                       self.bleed_subtraction_dict['total green - green from red_subtraction'][0]]
        
        # Next 3 are Blue, Green
        if {'total blue - blue from green_subtraction','total green - green from blue'} \
            <= self.bleed_subtraction_dict.keys():
            
            GB_list = [self.bleed_subtraction_dict['total blue - blue from green_subtraction'][0],
                       self.bleed_subtraction_dict['total green - green from blue'][0]]   
        
        elif {'total blue - blue from green_subtraction'} <= self.bleed_subtraction_dict.keys():
                                        
            GB_list = [self.bleed_subtraction_dict['total blue - blue from green_subtraction'][0],
                       self.state_dict['flat green']]
        
        elif {'total green - green from blue_subtraction'} <= self.bleed_subtraction_dict.keys():
                                        
            GB_list = [self.state_dict['flat blue'],
                       self.bleed_subtraction_dict['total green - green from blue_subtraction'][0]]
            
        # Last 3 are Blue, Red
        if {'total blue - blue from red_subtraction','total red - red from blue_subtraction'} \
            <= self.bleed_subtraction_dict.keys():
            
            RB_list = [self.bleed_subtraction_dict['total red - red from blue_subtraction'][0],
                       self.bleed_subtraction_dict['total blue - blue from red_subtraction'][0]]
            
        elif {'total red - red from blue_subtraction'} <= self.bleed_subtraction_dict.keys():
           
            RB_list = [self.bleed_subtraction_dict['total red - red from blue_subtraction'][0],
                       self.state_dict['flat blue']]
             
        elif {'total blue - blue from red_subtraction'} <= self.bleed_subtraction_dict.keys():
             
            RB_list = [self.state_dict['flat red'],
                       self.bleed_subtraction_dict['total blue - blue from red_subtraction'][0]]
      
        return RG_list, RB_list, GB_list          
        
    def pearson_plot_data(self, RG_list = None, RB_list = None, GB_list = None):  
            
        st.markdown('<p class = "green-font">Intensity Correlation Plots</p>', unsafe_allow_html = True) 
        
        self.axs[0].scatter(RG_list[0], RG_list[1], s = 1)
        self.axs[1].scatter(RB_list[0], RB_list[1], s = 1)
        self.axs[2].scatter(GB_list[0], GB_list[1], s = 1)
   
        plt.tight_layout()
        st.pyplot(self.fig)
        
        return RG_list, RB_list, GB_list
        
    def bleed_correction_plots(self, bleed_data):
        
        with st.expander('Bleed Correction Plots'):
            
            # Create a matplotlib figure objecdt
            fig = plt.figure(figsize = (7, 4))
            
            # Start a plot position number
            n = 1
            # Draw plots until there are no more to draw
            while n <= len(bleed_data.keys()):
                for key, value in bleed_data.items(): 
                   
                    ax = plt.subplot(2, 3, n)
                    ax.set_title(str(value[1]) + ' vs ' + str(value[2]))
                    ax.scatter(value[0][value[2]], value[0][value[1]], s = 1)
                    n += 1
                 
            plt.tight_layout()
            st.pyplot(fig)
       
      
    def pearsonr(self, RGB_lists):
        
        # Plot each channel vs. the others and get pearson coeffs
        RG_pears = scipy.stats.pearsonr(RGB_lists[0][0], RGB_lists[0][1])
        RB_pears = scipy.stats.pearsonr(RGB_lists[1][0], RGB_lists[1][1])
        GB_pears = scipy.stats.pearsonr(RGB_lists[2][0], RGB_lists[2][1])
        
        RG_CL = RG_pears.confidence_interval(confidence_level = 0.95)
        GB_CL = GB_pears.confidence_interval(confidence_level = 0.95)
        RB_CL = RB_pears.confidence_interval(confidence_level = 0.95)
        
        pearson_results_dict = {'RG': {'r2':round(RG_pears.statistic, 2), 'pvalue':round(RG_pears.pvalue, 2), 
                                           'low_CL':round(RG_CL[0],2), 'high_CL':round(RG_CL[1],2)}, 
                                'GB': {'r2':round(GB_pears.statistic,2), 'pvalue':round(GB_pears.pvalue,2), 
                                           'low_CL':round(GB_CL[0],2), 'high_CL':round(GB_CL[1],2)},
                                'RB': {'r2':round(RB_pears.statistic,2), 'pvalue':round(RB_pears.pvalue,2), 
                                           'low_CL':round(RB_CL[0],2), 'high_CL':round(RB_CL[1],2)}}
        
        return pearson_results_dict
    
    def spearmanr(self, RGB_lists):
        
        RG_spear = scipy.stats.spearmanr(RGB_lists[0][0], RGB_lists[0][1])
        RB_spear = scipy.stats.spearmanr(RGB_lists[1][0], RGB_lists[1][1])
        GB_spear = scipy.stats.spearmanr(RGB_lists[2][0], RGB_lists[2][1])
        
        spearman_results_dict = {'RG': {'r2':round(RG_spear.statistic, 2), 'pvalue':round(RG_spear.pvalue, 2)},  
                                'GB': {'r2':round(GB_spear.statistic,2), 'pvalue':round(GB_spear.pvalue,2)}, 
                                'RB': {'r2':round(RB_spear.statistic,2), 'pvalue':round(RB_spear.pvalue,2)}}
        
        return spearman_results_dict
    
    def pearsonr_spearmanr_results_table(self, results_dict, title):
        """Imports results from pearsonr and spearmanr and adds them to tables using the create_table method in 
        the Setup class."""
        
        # Change any pvalue < 0.005 to '<0.005'
        for outer_key, inner_dict in results_dict.items():
            if inner_dict['pvalue'] < 0.005:
                inner_dict['pvalue'] = '<0.005'
            # Add an interpretation of the R2 value
            if 0.75 <= inner_dict['r2'] <= 1.0: 
                inner_dict['Interpretation'] ='Strongly Colocalized'
            elif 0.50 <= inner_dict['r2'] <= 0.74:
                inner_dict['Interpretation'] = 'Weakly Colocalized'
            elif -0.50 <= inner_dict['r2'] <= 0.49:
                inner_dict['Interpretation'] = 'Random Mixture'
            else:
                inner_dict['Interpretation'] = 'Separated'
         
            try:
               # Add the low to high confidence limits
               inner_dict['95% CL'] = str(inner_dict['low_CL']) + ' to ' + str(inner_dict['high_CL'])
            
            except:
               inner_dict['95% CL'] = "NA"
        
        # Create a dictionary and dataframe for the pearson analysis
        results_df = pd.DataFrame({'Statistic':['Correlation (-1 to 1)', 'p value', '95% CL', 'Interpretation'],
                              'Green-Red':[results_dict['RG']['r2'], results_dict['RG']['pvalue'],
                                           results_dict['RG']['95% CL'], results_dict['RG']['Interpretation']],
                              'Blue-Red':[results_dict['RB']['r2'], results_dict['RB']['pvalue'],
                                           results_dict['RB']['95% CL'], results_dict['RB']['Interpretation']], 
                              'Green-Blue':[results_dict['GB']['r2'], results_dict['GB']['pvalue'],
                                           results_dict['GB']['95% CL'], results_dict['GB']['Interpretation']]})
   
        # Generate the table of results by calling the create table method in Setup
        table = Setup(state_dict).create_table(results_df, title)
        
    def manders(self, RGB_lists):
        
        # Create df of intensities for each channel
        # Call the first 2 lists in RGB_lists Red1 and Green1, the next 2 lists Blue2, Green2...
        rgb_df = pd.DataFrame({'Red1':RGB_lists[0][0], 'Green1':RGB_lists[0][1], 
                               'Blue2':RGB_lists[1][0], 'Green2':RGB_lists[1][1],
                                'Blue3':RGB_lists[2][0], 'Red3':RGB_lists[2][1]})
        
        rgb_df['rxg'] = rgb_df['Red1'] * rgb_df['Green1']
        rgb_df['gxb'] = rgb_df['Blue2'] * rgb_df['Green2']
        rgb_df['rxb'] = rgb_df['Blue3'] * rgb_df['Red3']
        
        # Add up the intensities of each channel
        total_red1, total_red3 = rgb_df['Red1'].sum(), rgb_df['Red3'].sum()
        total_green1, total_green2 = rgb_df['Green1'].sum(), rgb_df['Green2'].sum()
        total_blue2, total_blue3 = rgb_df['Blue2'].sum(), rgb_df['Blue3'].sum()
       
        rg_count = 0
        gr_count = 0
        
        rb_count = 0
        br_count = 0
        
        gb_count = 0
        bg_count = 0
        
        for row in rgb_df.itertuples():
            if row.rxg > 0:
                rg_count += row.Red1
                gr_count += row.Green1
            if row.rxb > 0:
                rb_count += row.Red3
                br_count += row.Blue3
            if row.gxb > 0:
                gb_count += row.Green2
                bg_count += row.Blue2    

        rg_M1 = rg_count/total_red1
        rg_M2 = gr_count/total_green1
        rb_M1 = rb_count/total_red3
        rb_M2 = br_count/total_blue3
        gb_M1 = gb_count/total_green2
        gb_M2 = bg_count/total_blue2
        
        
        return rg_M1, rg_M2, rb_M1, rb_M2, gb_M1, gb_M2
    
    def manders_table(self, manders):
        

        
        self.manders_df = pd.DataFrame({'Manders Coeffs':['M1', 'M2'], 
                                        'Green-Red':[round(manders[0],2),round(manders[1],2)], 
                                        'Blue-Red':[round(manders[2],2),round(manders[3],2)],
                                        'Green-Blue':[round(manders[4],2), round(manders[5],2)]})

        fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1], 
                                        header = dict(values = list(self.manders_df[['Manders Coeffs', 'Green-Red', 
                                                'Blue-Red', 'Green-Blue']].columns),fill_color = '#FD8E72',
                                                line_color='darkslategray', align = 'center', 
                                                font = dict(color = 'blue', size = 18)), 
                                                cells = dict(values = [self.manders_df['Manders Coeffs'], 
                                                self.manders_df['Green-Red'], 
                                                self.manders_df['Blue-Red'], 
                                                self.manders_df['Green-Blue']],
                                                fill_color = 'lavender', 
                                                align = 'center', 
                                                line_color='darkslategray',
                                                height = 30)))
        
        fig.update_traces(cells_font = dict(size = 15))
        
        fig.update_layout(title = 'Manders Co-occurance Results', 
                          title_font = dict(size = 18, color = 'green',
                          family = 'Arial'), 
                          title_x = 0.0, 
                          title_y = 0.95, 
                          height = 175,
                          margin = dict(l = 0, r = 0, b = 0, t = 35))
        # st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write(fig) 
 

class Cell_Counter():
    """Counts cells and determines live vs. dead cells based upon the 
    amount of cy5 fluorescence. Cy5 is used as the live-dead indicator, 
    selected as either the red or green channels depending on how the 
    image was acquired. Dead cells have much more cy5 than live cells.
    """
    
    def __init__(self, state_dict):
         
        self.state_dict = state_dict
        
    def contour(self):
 
        st.write('')
        st.write('')
        st.markdown('<p class = "purple-font-venti">Live vs. dead cell analysis.</p>', 
                    unsafe_allow_html = True) 
  
        # CHAIN_APPROX_SIMPLE throws out some points so the contour around the edge will be a dotted line
        # contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        self.contours, self.hierarchy = cv.findContours(self.state_dict['threshold image'], 
                                                        cv.RETR_LIST, 
                                                        cv.CHAIN_APPROX_NONE)
        self.state_dict['contours'] = self.contours

    def cell_stats(self):
        # Calculates a number of features of each individual cell found in the image such as mean RGB intensities in each cell
        # Create dicts for cell parameters
        # if self.state_dict['cell_count']:
            
        self.contour_dict = {}
        self.areaDict = {}
        self.aspectRatioDict = {}
        self.avgColorDict = {}
        self.perimeterDict = {}
        self.redMeanDict = {}
        self.greenMeanDict = {}
        self.blueMeanDict = {}
        
        cell_number = 0
        
        for contour in self.state_dict['contours']:
            
            # Set a minimum area for a cell to be counted
            if cv.contourArea(contour) > self.state_dict['contour_area']:
               
                cell_number += 1
                
                # Add contour to dict
                self.contour_dict[cell_number] = contour
        
                # Fit an elipse to each cell
                ellipse = cv.fitEllipse(contour)
                # A more direct route to the major and minor axis distances
                (x,y), (MA, ma), angle = cv.fitEllipse(contour)
                cv.ellipse(self.state_dict['cropped image array'], 
                           ellipse, (255,255,255), 2, cv.LINE_AA)
                
                # Draw the contour that follows the cell perfectly
                cv.drawContours(self.state_dict['cropped image array'], 
                                contour, -1, (0, 255, 255), 2)
                
                # Get the cell area
                self.areaDict[cell_number] = cv.contourArea(contour)
                
                # Get aspect ratio
                # Several methods shown here
                
                # Method 1: Also good for getting the average rgb intensities 
                # inside a rectangular contour around the cell
                
                x,y,w,h = cv.boundingRect(contour)
                
                # Draw the bounding rectangle
                # cv.rectangle(self.state_dict['cropped image array'],(x,y),(x+w,y+h),(0,255,0),2)
                # Calculate aspect ratio
                # if w > h:
                #     self.aspect_ratio = float(w)/h
                # else:
                #     self.aspect_ratio = float(h)/w
                # st.write('Method 1: ', self.aspect_ratio)
                
                # Get the average RGB intensities
                self.mean_intensities = np.array(cv.mean(self.state_dict['cropped image array'][y:y+h,x:x+w])).astype(np.uint8)
                self.redMeanDict[cell_number] = self.mean_intensities[0]
                self.greenMeanDict[cell_number] = self.mean_intensities[1]
                self.blueMeanDict[cell_number] = self.mean_intensities[2]
                 
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
                cv.putText(self.state_dict['cropped image array'], 
                           text = str(cell_number), 
                           org=(cx,cy),
                           fontFace = cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale = 2, 
                           color = (255,255,255),
                           thickness = 2, lineType=cv.LINE_AA)
        
        # Print the real image with contours and labels
        st.image(self.state_dict['cropped image array'], 
                 use_column_width = True, 
                 clamp = True)
        
        # Download the image with the contours marked by calling download_image method in Prep_Image class
        Prep_Image(state_dict).download_image(self.state_dict['cropped image array'], 
                                              "Download Cell Contours")
        
        # Get values from dictionaries to append to tables
        number_of_cells = [x for x in range(1, cell_number + 1)]
        aspectRatio_list = [value for key, value in self.aspectRatioDict.items()]
        red_list = [value for key, value in self.redMeanDict.items()]
        green_list = [value for key, value in self.greenMeanDict.items()]
        blue_list = [value for key, value in self.blueMeanDict.items()]
        # area_list = [value for key, value in self.areaDict.items()]
        
        st.write('')
        st.write('')
        
        fig = plt.figure(figsize = (7, 2))
        red_hist = plt.hist(red_list, 
                            color = 'red', 
                            bins = 50, 
                            label = 'Red Channel')
        
        green_hist = plt.hist(green_list, 
                              color = 'green', 
                              bins = 50, 
                              label = 'Green Channel')
        
        plt.xlabel('Intensity', fontsize = 8)
        plt.ylabel('Pixels', fontsize = 8)
        plt.title('Pixel Intensity Histograms')
        st.pyplot(fig)
        
        st.write('')
        st.write('')
        live_dead_threshold = st.slider('Select an intensity threshold for Live-Dead Analysis', 
                                        75, 200, 100, 5)
        
        hpa_color_radio = st.radio('HPA Color',
                                   [':red[Red]', 
                                    ':green[Green]'], 
                                   horizontal = True)
        
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
        fig.update_layout(title = "Cell Analysis", 
                          title_font = dict(size = 18, 
                                            color = 'green',
                                            family = 'Arial'), 
                          title_x = 0, 
                          title_y = 0.95, 
                          height = 300, 
                          width = 1000,
                          margin = dict(l = 0, r = 0, b = 0, t = 35))
        st.write('')
        st.write('')
        st.write(fig)      
        
        # Button to download dataframe as csv file
        self.live_dead_button = st.download_button(label = 'Download', 
                                                   data = self.cell_df.to_csv(), 
                                                   file_name = 'live_dead.csv', 
                                                   mime = 'text/csv')
        
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
        
        st.write('')
        fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1,1], 
                                        header = dict(values = list(
                                                self.cell_totals_df[['Number of Objects Found', 
                                                                     'Live Cells', 
                                                                     'Dead Cells', 
                                                                     'Possible Multiplets']].columns),
                                            fill_color = '#FD8E72',
                                            line_color='darkslategray', 
                                            align = 'center', 
                                            
                                        font = dict(color = 'blue', size = 18)), 
                                                cells = dict(values = [self.cell_totals_df['Number of Objects Found'], 
                                                                        self.cell_totals_df['Live Cells'], 
                                                                        self.cell_totals_df['Dead Cells'],
                                                                        self.cell_totals_df['Possible Multiplets']],
                                                fill_color = 'lavender', align = 'center', line_color = 'darkslategray',
                                                height = 30)))
                                                                                          
        
        fig.update_traces(cells_font = dict(size = 15))
        
        # In update layout both the margin and height control the spacing between tables in the vertical direction. 
        fig.update_layout(title = "Cell Totals", 
                          title_font = dict(size = 18, 
                                            color = 'green',
                                            family = 'Arial'), 
                          title_x = 0, 
                          title_y = 0.95, 
                          height = 100, 
                          width = 1000,
                          margin = dict(l = 0, r = 0, b = 0, t = 25))
        
        st.write('')
        st.write('')
        st.write(fig) 
        
        # Button to download dataframe as csv file
        self.summary_button = st.download_button(label = 'Download', 
                                                 data = self.cell_totals_df.to_csv(), 
                                                 file_name = 'live_dead.csv', 
                                                 mime = 'text/csv', 
                                                 key = 'summary')

    def total_intensity(self):
        # Calculates the total RGB intensities for each live cell and calculates the mean/cell and std/cell
        
        # Remove the contours of the dead cells from self.contour_dict
        # Get cell numbers for dead cells
        # Filter self.cell_df by dead cell
        self.dead_df = self.cell_df[self.cell_df['Cell Status'] == 'Dead']
        self.dead_list = self.dead_df['Cell Number'].tolist()
        
        # Remove those entries from the contour dict
        self.live_cntr_dict = {key:value for key,value in self.contour_dict.items() if key not in self.dead_list}
        
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
            mask = np.zeros_like(self.state_dict['cropped image array'][:, :, 0])
            # Fill in the contours. The -1 fills in all contours.
            cv.drawContours(mask, [contour], -1, 255, thickness = cv.FILLED)

            red_sum = np.sum(self.state_dict['cropped image array'][:, :, 0][mask == 255])
            green_sum = np.sum(self.state_dict['cropped image array'][:, :, 1][mask == 255])
            blue_sum = np.sum(self.state_dict['cropped image array'][:, :, 2][mask == 255])
           
            self.red_intensity_dict[cell_number] = red_sum
            self.green_intensity_dict[cell_number] = green_sum
            self.blue_intensity_dict[cell_number] = blue_sum
        
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
        
        fig = go.Figure(data = go.Table(columnwidth = [1,1,1,1,1], 
                                        header = dict(values = list(
                                            self.RGB_intensity_df[[
                                                'Live Cells', 
                                                'Mean Red/Cell',
                                                'STD Red/Cell',
                                                'Mean Green/Cell',
                                                'STD Green/Cell',
                                                'Mean Blue/Cell',
                                                'STD Blue/Cell'
                                                ]].columns),
                                            fill_color = '#FD8E72', 
                                            line_color = 'darkslategray', 
                                            align = 'center', 
                                            font = dict(color = 'blue', size = 14)), 
                                            cells = dict(values = 
                                                         [self.RGB_intensity_df['Live Cells'], 
                                                         self.RGB_intensity_df['Mean Red/Cell'], 
                                                        self.RGB_intensity_df['STD Red/Cell'],
                                                        self.RGB_intensity_df['Mean Green/Cell'],
                                                        self.RGB_intensity_df['STD Green/Cell'],
                                                        self.RGB_intensity_df['Mean Blue/Cell'],
                                                        self.RGB_intensity_df['STD Blue/Cell']],
                                            fill_color = 'lavender', 
                                            align = 'center', 
                                            line_color = 'darkslategray', 
                                            height = 30)))

        fig.update_traces(cells_font = dict(size = 15))
        
        # In update layout both the margin and height control the spacing between tables in the vertical direction. 
        fig.update_layout(title = "Total Live Cell RGB Intensities", 
                          title_font = dict(size = 18, 
                                            color = 'green',
                                            family = 'Arial'), 
                          title_x = 0, 
                          title_y = 0.95, 
                          height = 100, 
                          width = 1000,
                          margin = dict(l = 0, r = 0, b = 0, t = 25))
        
        st.write('')
        st.write('')
        st.write(fig) 
        
        # Button to download dataframe as csv file
        self.live_cell_analysis_button = st.download_button(label = 'Download', 
                                                            data = self.RGB_intensity_df.to_csv(), 
                                                            file_name = 'live_cell_RGB.csv', 
                                                            mime = 'text/csv', 
                                                            key = 'live')


# Run 
if __name__ == '__main__':
    
    state_dict = {}
    obj1 = Flow_Control(state_dict)
    all_calls = obj1.all_calls()




