import pyrealsense2 as rs
import numpy as np
from skimage import morphology
import cv2
import os
import glob
import imutils
import pandas as pd
from skimage.measure import label, regionprops
from ipywidgets import IntProgress, HTML, VBox, Layout
from IPython.display import display
import math
from random import seed
from random import randint, random
from skimage import transform
from skimage.util import random_noise
from scipy import ndimage
from skimage import exposure
import matplotlib.pyplot as plt

seed(1)

class DecisionSystem:
    
    def __init__(self, save_results=True):
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = 848
        self.intrinsics.height = 480
        self.intrinsics.fx = 424.5785827636719
        self.intrinsics.fy = 424.5785827636719
        self.intrinsics.ppx = 422.18994140625
        self.intrinsics.ppy = 244.84666442871094
        self.intrinsics.model = rs.distortion.brown_conrady
        
        self.path_color = '../data/color/'
        self.path_depth = '../data/depth/'
        self.path_results = '../data/results/'
        self.save_results = save_results
        
        
    def load_images(self, disk=3, comp=9000):
        path = os.getcwd()        
        if (not os.path.isdir(self.path_results)):
            os.mkdir(self.path_results)
        self.images= []  
        weight_data = pd.read_csv(os.path.join("..", 'data', 'id_weight_sex_uuid.csv'), usecols=['uuid', 'id', 'sex', 'weight'])
        size= weight_data.shape[0]
        f = IntProgress(min=0, max=size) # instantiate the bar
        label = HTML()
        box = VBox([f, label])
        display(box)
        for index, row in weight_data.iterrows():
            img = Image(self, os.path.basename(row.uuid), row.id, row.sex, row.weight, comp)
            img.calculate_mask(diskSize=disk)
            img.calculate_parameters()
            img.calculate_height()
            
            if self.save_results:
                img.save_results()
            
            self.images.append(img)
            f.value += 1 # signal to increment the progress bar
            label.value = u'{index} / {size}'.format(index=f.value, size=size)
                
    def load_aug_images(self, disk=3, comp=9000, f_name='aug_data'):
        path = os.getcwd()        
        if (not os.path.isdir(self.path_results)):
            os.mkdir(self.path_results)
        self.images= []  
        weight_data = pd.read_csv(os.path.join("..", 'data', 'id_weight_sex_uuid.csv'), usecols=['uuid', 'id', 'sex', 'weight'])
        images = [f[:-4] for f in os.listdir(self.path_color)]
        size= len(images)
        f = IntProgress(min=0, max=size) # instantiate the bar
        label = HTML()
        box = VBox([f, label])
        display(box)
        
        columns = ['filename', 'id', 'sex', 'weight', 'height', 'complete','area','area_weighted','mayor_axis','minor_axis',
            'percent_area','eccentricity','perimeter','width_bbox','height_bbox', 'mask_method', 'symmetry', 'orientation', 'distance']
        self.info = pd.DataFrame(columns=columns)
        cont = 0
        for file in images:
            info = weight_data[weight_data['uuid']==file.split('_')[0]]
            row = info.iloc[0]
            img = Image(self, file, row.id, row.sex, row.weight, comp)
            img.calculate_mask(diskSize=disk)
            img.calculate_parameters()
            img.calculate_height()
            
            if self.save_results:
                img.save_results()
            
            #self.images.append(img)
            f.value += 1 # signal to increment the progress bar
            label.value = u'{index} / {size}'.format(index=f.value, size=size)
            
            info = img.get_info()
            self.info = pd.concat([self.info, pd.DataFrame(info, columns= columns, index=[cont])])
            cont +=1
        self.info.to_csv(os.path.join("..", 'data', f_name+'.csv'))
            
    
    def aug_images(self, filtered_data, decision_system):
        folder_aug = '../data/augmentation'
        if not os.path.exists(folder_aug):
            os.mkdir(folder_aug)

        if not os.path.exists(folder_aug+'/color'):
            os.mkdir(folder_aug+'/color')

        if not os.path.exists(folder_aug+'/depth'):
            os.mkdir(folder_aug+'/depth')

        if not os.path.exists(folder_aug+'/results'):
            os.mkdir(folder_aug+'/results')
        
        count_ids = filtered_data.groupby(['id'])['id'].count()
        count_ids = count_ids[count_ids<30]
        
        size= count_ids.shape[0]
        f = IntProgress(min=0, max=size)
        label = HTML()
        box = VBox([f, label])
        display(box)        
        
        for index, cont_id in count_ids.iteritems():
            cont = cont_id
            images_id = filtered_data[filtered_data['id']==index]
            max_aug = 10 if cont_id <5 else 20 if cont_id <15 else 20
            max_aug=50
            while (cont < max_aug):
                for _, row_image in images_id.iterrows():
                    if cont < max_aug:
                        img = Image(decision_system, row_image.filename, row_image.id, row_image.sex, row_image.weight)
                        color, depth, t = img.transform_image()
                        img_name = row_image.filename+'_n'+str(int(t[0]))+'_a'+str(t[1])+'_b'+str(t[2])+'_i'+str(int(t[3]))
                        np.save(folder_aug+'/color/'+img_name+'.npy',color)
                        np.save(folder_aug+'/depth/'+img_name+'.npy',depth)
                        cont +=1
                    else:
                        break
            f.value += 1 # signal to increment the progress bar
            label.value = u'{index} / {size}'.format(index=f.value, size=size)
            
            
    def save_info(self, f_name='final_data'):
        columns = ['filename', 'id', 'sex', 'weight', 'height', 'complete','area','area_weighted','mayor_axis','minor_axis',
            'percent_area','eccentricity','perimeter','width_bbox','height_bbox', 'mask_method', 'symmetry','symmetry_v', 'orientation', 'distance']
        self.info = pd.DataFrame(columns=columns)
        for image, cont in zip(self.images, range(1,len(self.images)+1)):
            info = image.get_info()
            #self.info = self.info.append(info, ignore_index=True)
            self.info = pd.concat([self.info, pd.DataFrame(info, columns= columns, index=[cont])])
        self.info.to_csv(os.path.join("..", 'data', f_name+'.csv'))
        
    def load_data(self, f_name='final_data'):
        self.info = pd.read_csv(os.path.join("..", 'data', f_name+'.csv'))
        self.info['area_bb'] = self.info['width_bbox'].values * self.info['height_bbox'].values
            

class Image:
    
    def __init__(self, system, filename, id_lamb, sex, weight, component=5000):
        self.system = system
        self.filename = filename
        self.id = id_lamb
        self.sex = sex
        self.weight = weight
        self.component = component
        
        #self.path_color = '../data/color/'
        #self.path_depth = '../data/depth/'
        #self.path_results = '../data/results/'
        
        self.color_image = np.load(self.system.path_color+self.filename+'.npy',allow_pickle=True)
        
        self.depth_image = np.load(self.system.path_depth+self.filename+'.npy',allow_pickle=True)
        
        self.mask = None
        self.height, self.width, _ = self.color_image.shape
        
        self.area = None
        self.area_weighted = None
        self.mayor_axis = None
        self.minor_axis = None
        self.symmetry = None
        self.symmetry_v = None
        self.left_corner = None
        self.centroid = None
        self.orientation = None
        self.percent_area = None
        self.eccentricity = None
        self.perimeter = None
        self.width_bbox = None
        self.height_bbox = None
        self.orientation = None
        self.distance = None
        
        self.height = None
        
        self.complete_info = False
        self.mask_method = 'NEW'
        
    def get_info(self):
        return {
            'filename': self.filename,
            'id': self.id,
            'sex': self.sex,
            'weight': self.weight,
            'complete': self.complete_info,
            'area': self.area,
            'area_weighted': self.area_weighted,
            'symmetry' : self.symmetry,
            'symmetry_v' : self.symmetry_v,
            'mayor_axis': self.mayor_axis,
            'minor_axis': self.minor_axis,
            'percent_area': self.percent_area,
            'eccentricity': self.eccentricity,
            'perimeter': self.perimeter,
            'width_bbox': self.width_bbox,
            'height_bbox': self.height_bbox,
            'mask_method': self.mask_method,
            'orientation': self.orientation,
            'distance': self.distance,
            'height': self.height
        }
    
    def transform_image(self):
        img_transformed = self.color_image.copy()
        depth_transformed = self.depth_image.copy()
        
        noise = False
        blur_size = 0
        intensity = False
        percent = 0
        
        if random()>0.5:
            #img_transformed = random_noise(img_transformed, mode='gaussian')
            noise = False
            
        if random()>0.5:
            angle = randint(-15, 15)            
        else:
            angle = randint(165, 195)
        img_transformed = transform.rotate(img_transformed, angle=angle, mode='symmetric', preserve_range=True)
        depth_transformed = transform.rotate(depth_transformed, angle=angle, mode='symmetric', preserve_range=True)
        
        if random()>0.5:
            blur_size = randint(1, 5)
            img_transformed = ndimage.uniform_filter(img_transformed, size=(blur_size, blur_size, 1))
        if random()>0.5:
            percent = random()*0.1
            v_min, v_max = np.percentile(img_transformed, (0.2, 99.8))
            img_transformed = exposure.rescale_intensity(img_transformed, in_range=(v_min, v_max))
            intensity = True
            
        if img_transformed.max()<=1.05:
            img_transformed = (img_transformed*255)
        img_transformed = img_transformed.astype(self.color_image.dtype)
        depth_transformed = depth_transformed.astype(self.depth_image.dtype)
        return img_transformed, depth_transformed, [noise, angle, blur_size, intensity, percent]
        
        
    def calculate_mask(self, diskSize= 9):
        img_n = cv2.normalize(self.color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #Apply sharpen kernel
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        image_sharp = cv2.filter2D(src=img_n, ddepth=-1, kernel=kernel)
        self.sharp_img =  image_sharp
        #Get luminance from LAB color space
        img_lab = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2LAB)
        lum_img = img_lab[:,:,0]
        lum_scale = cv2.convertScaleAbs(lum_img)
        self.lab_img = lum_scale
        #CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9,9))
        self.clahe = clahe.apply(self.lab_img)
        #otsu binarization
        _,equ_otsu = cv2.threshold(self.clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.clahe_otsu = equ_otsu
        #opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diskSize,diskSize))
        opening = cv2.morphologyEx(equ_otsu, cv2.MORPH_OPEN, kernel)
        #clean mask
        self.original_mask = opening
        self.mask = opening
        cnt = self.get_biggest_area()
        if cnt is not None:
            cleaned_mask = np.zeros(opening.shape,np.uint8)
            self.mask= cv2.drawContours(cleaned_mask,[cnt],0,255,-1)
            
        """
        img = cv2.normalize(self.color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #Apply sharpen kernel
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        self.sharp_img =  image_sharp
        #Get luminance from LAB color space
        img_lab = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2LAB)
        lum_img = img_lab[:,:,0]
        lum_scale = cv2.convertScaleAbs(lum_img)
        self.lab_img = lum_scale
        #Otsu binarization
        _,lum_bin = cv2.threshold(lum_scale,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.lab_mask = lum_bin
        #Find contours
        lum_areas = self.count_area_mask(lum_bin)
        kernel = np.ones((5,5),np.uint8)
        if len(lum_areas)==1:
            lum_closing = cv2.morphologyEx(lum_bin, cv2.MORPH_CLOSE, kernel)
            closing_areas = self.count_area_mask(lum_closing)
            if len(closing_areas)==1:
                final_mask= lum_closing
                self.mask_method = 'LAB_closing'
            else:
                lum_opening = cv2.morphologyEx(lum_bin, cv2.MORPH_OPEN, kernel)
                opening_areas = self.count_area_mask(lum_opening)
                final_mask= lum_opening
                self.mask_method = 'LAB_opening'
        else:
            #Calculate gray method
            gray_img = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)
            self.gray_img = gray_img
            gray_scale = cv2.convertScaleAbs(gray_img)
            _,gray_bin= cv2.threshold(gray_scale,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            final_mask = cv2.morphologyEx(gray_bin, cv2.MORPH_OPEN, kernel)
            self.mask_method = 'Gray_opening'

        self.original_mask = final_mask
        self.mask = self.clean_mask(final_mask, diskSize)
        """
        
    def count_area_mask(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas= []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area>self.component:
                x,y,w,h = cv2.boundingRect(cnt)
                limit = True if x==0 or y ==h or (x+w)==mask.shape[1] or (y+h)==mask.shape[0] else False
                if not limit:
                    areas.append(area)
        areas.sort()
        return areas
    
    def clean_mask(self, mask_in, diskSize= 9):
        #OPENING
        #kernel = np.ones((5,5),np.uint8)*255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diskSize,diskSize))
        closing = cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel)
        
        #BIGGEST CONTOUR DONT TOUCH BOUNDARIES
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnt = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            if (x> 0 and y>0 and (x+w)<(closing.shape[1]-1) and (y+h)<(closing.shape[0]-1)):
                filtered_cnt.append(cnt)

        big_area = 0
        big_cnt = None
        for cnt in filtered_cnt:
            area = cv2.contourArea(cnt)
            if area>big_area:
                big_area = area
                big_cnt = cnt
        cleaned_mask = np.zeros(closing.shape,np.uint8)
        if big_area>0:
            cv2.drawContours(cleaned_mask,[big_cnt],0,255,-1)
                
        return cleaned_mask

        
        """        
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(mask_in, cv2.MORPH_CLOSE, kernel)
        mask_depth = closing.astype(bool)
        
        # Removing small objects
        cleaned_mask_depth = morphology.remove_small_objects(mask_depth, min_size=500, connectivity=1)

        # Each object is labelled with a number from 0 to number of regions - 1
        ret, markers = cv2.connectedComponents(cleaned_mask_depth.astype(np.uint8) * 255)
        markers = markers + 1  # So we will label background as 0 and regions from 1 to number of regions
        markers[cleaned_mask_depth == 0] = 0  # background is region 0

        # Removing regions touching the borders of the image
        etiquetas_borde_superior = np.unique(markers[0, :])
        etiquetas_borde_inferior = np.unique(markers[-1, :])
        etiquetas_borde_izquierdo = np.unique(markers[:, 0])
        etiquetas_borde_derecho = np.unique(markers[:, -1])
        etiquetas_bordes = np.unique( np.concatenate([etiquetas_borde_superior, etiquetas_borde_inferior, etiquetas_borde_izquierdo, etiquetas_borde_derecho]))
        for label in etiquetas_bordes:
            if label > 0:
                markers[markers == label] = 0
        # Applying the mask to the image a segmented image is obtained
        mask = markers.astype(np.uint8) * 255
        cleaned = morphology.remove_small_objects(mask, min_size=500, connectivity=1)
        cleaned2 = cleaned.astype(np.uint8)*255
        # Copy the thresholded image.
        im_floodfill = cleaned2.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = cleaned2.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = cleaned2 | im_floodfill_inv

        label_img = morphology.label(im_out,connectivity=1)
        size = np.bincount(label_img.ravel())
        if len(size) == 1:
            return np.zeros_like(mask_in)
        biggest_label = size[1:].argmax() + 1

        clump_mask = label_img == biggest_label

        selem = morphology.disk(diskSize)

        clump_maskA = clump_mask.astype(np.uint8)*255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diskSize,diskSize))
        #kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(clump_maskA, cv2.MORPH_OPEN, kernel)
        #eroded = cv2.erode(clump_maskA,selem)

        label2_img = morphology.label(closing,connectivity=1)
        size2 = np.bincount(label2_img.ravel())
        if len(size2) == 1:
            return np.zeros_like(mask_in)
        biggest_label2 = size2[1:].argmax() + 1
        clump_mask2 = label2_img == biggest_label2

        mask = clump_mask2.astype(np.uint8)*255

        return mask
        """
    
    def get_biggest_area(self):
        """
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_area = 0
        big_cnt = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area>big_area and area>self.component:
                big_area = area
                big_cnt = cnt
        """
        #BIGGEST CONTOUR DONT TOUCH BOUNDARIES
        contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnt = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            if (x> 0 and y>0 and (x+w)<(self.mask.shape[1]-1) and (y+h)<(self.mask.shape[0]-1)):
                filtered_cnt.append(cnt)

        big_area = 0
        big_cnt = None
        for cnt in filtered_cnt:
            area = cv2.contourArea(cnt)
            if area>big_area and area>self.component:
                big_area = area
                big_cnt = cnt
        return big_cnt
    
    def calculate_parameters(self):
        heightMask, widthMask = self.mask.shape
        #contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        cnt =  self.get_biggest_area()
        self.image_contour = self.color_image.copy()
              
        if cnt is not None:
            cv2.drawContours(self.image_contour, [cnt], 0, (0,255,0), 3)
            #cnt = np.vstack(contours)

            # Check pixels within limits
            img_coord_x_up_left_corner, img_coord_y_up_left_corner, width, height = cv2.boundingRect(cnt)
            self.width_bbox = width
            self.height_bbox = height

            img_coord_x_up_right_corner = img_coord_x_up_left_corner + width - 1
            img_coord_y_up_right_corner = img_coord_y_up_left_corner

            img_coord_x_bottom_left_corner = img_coord_x_up_left_corner
            img_coord_y_bottom_left_corner = img_coord_y_up_left_corner + height - 1

            are_corners_within = ((0 <= img_coord_x_up_left_corner < widthMask - 1) and (0 <= img_coord_y_up_left_corner < heightMask - 1) 
                                  and (0 <= img_coord_x_up_right_corner < widthMask) and (0 <= img_coord_y_up_right_corner < heightMask) and
                                  (0 <= img_coord_x_bottom_left_corner < widthMask) and (0 <= img_coord_y_bottom_left_corner < heightMask))

            if are_corners_within:            
                depth_value = cv2.mean(self.depth_image, self.mask)[0]

                up_right_corner = rs.rs2_deproject_pixel_to_point(self.system.intrinsics,
                                                                  [img_coord_x_up_right_corner,
                                                                   img_coord_y_up_right_corner],
                                                                  depth_value)
                up_right_corner = np.array(up_right_corner)
                up_left_corner = rs.rs2_deproject_pixel_to_point(self.system.intrinsics,
                                                                 [img_coord_x_up_left_corner,
                                                                  img_coord_y_up_left_corner],
                                                                 depth_value)
                up_left_corner = np.array(up_left_corner)
                bottom_left_corner = rs.rs2_deproject_pixel_to_point(self.system.intrinsics,
                                                                     [img_coord_x_bottom_left_corner,
                                                                      img_coord_y_bottom_left_corner],
                                                                     depth_value)
                bottom_left_corner = np.array(bottom_left_corner)
                distancia_horizontal = np.linalg.norm(up_left_corner[:2] - up_right_corner[:2])
                distancia_vertical = np.linalg.norm(up_left_corner[:2] - bottom_left_corner[:2])
                
                self.left_corner = (img_coord_x_up_left_corner, img_coord_y_up_left_corner)
                
                self.mask_bbox = self.color_image.copy()
                cv2.rectangle(self.mask_bbox, self.left_corner, (self.left_corner[0] + self.width_bbox - 1, self.left_corner[1] + self.height_bbox - 1), (0, 255, 0), 5)

                area_rectangulo_real = distancia_horizontal * distancia_vertical
                area_rectangulo_pixeles = width * height
                escala = area_rectangulo_real / area_rectangulo_pixeles
                area_mascara_pixeles = cv2.contourArea(cnt)#cv2.countNonZero(self.mask)
                area_mascara_real = area_mascara_pixeles * escala
                
                self.area = area_mascara_pixeles
                self.area_weighted = area_mascara_real


                # Elllipse
                """
                cnt = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cnt = imutils.grab_contours(cnt)
                contours = None
                countour_area = 0
                for c in cnt:
                    area = cv2.contourArea(c)
                    if area > 10000 and area > countour_area:
                        contours= c
                        countour_area = area

                if countour_area > 0: 
                """
                (x, y), (axis_x, axis_y), angle = cv2.fitEllipse(cnt)
                if axis_x > axis_y:
                    self.mayor_axis = axis_x
                    self.minor_axis = axis_y
                else:
                    self.mayor_axis = axis_y
                    self.minor_axis = axis_x

                # symmetry
                matrix = cv2.getRotationMatrix2D( center=(x,y), angle=angle-90, scale=1 )
                image = cv2.warpAffine( src=self.mask, M=matrix, dsize=(self.mask.shape[1], self.mask.shape[0]) )
                MA_2, ma_2 = int(self.mayor_axis/2), int(self.minor_axis/2)
                x_int, y_int = int(x), int(y)
                x_sim, x_sim2 = (x_int- MA_2), (x_int+ MA_2)
                y_sim, y_sim2 = (y_int- ma_2), (y_int+ ma_2)
                y_sim = 0 if y_sim<0 else y_sim
                x_sim = 0 if x_sim<0 else x_sim
                y_sim2 = image.shape[0]-1 if y_sim2>(image.shape[0]-1) else y_sim2
                x_sim2 = image.shape[1]-1 if x_sim2>(image.shape[1]-1) else x_sim2
                #if y_sim>0 and x_sim>0 and x_sim2<image.shape[1] and y_sim2<image.shape[0]:
                image_bbox = image[y_sim:y_sim2, x_sim:x_sim2]
                # symmetry_horizontal
                image_part1 = image[y_sim:y_int, :]
                image_part2 = image[y_int:y_sim2, :]
                mirror=cv2.flip(image_part2,0)
                if(image_part1.shape[0]!=mirror.shape[0]):
                    if mirror.shape[0] < image_part1.shape[0]:
                        diff = image_part1.shape[0] - mirror.shape[0]                        
                        image_part1 = image_part1[diff:,:]
                    else:
                        diff = mirror.shape[0] - image_part1.shape[0]                        
                        mirror = mirror[diff:,:]
                union = cv2.bitwise_and(image_part1,image_part1,mask=mirror)
                union_pixels = cv2.countNonZero(union)
                mask_pixels = cv2.countNonZero(self.mask)
                self.symmetry = 2*union_pixels/mask_pixels
                
                # symmetry_vertical
                image_part1 = image[:, x_sim:x_int]
                image_part2 = image[:, x_int:x_sim2]
                mirror=cv2.flip(image_part2,1)
                if(image_part1.shape[1]!=mirror.shape[1]):
                    if mirror.shape[1] < image_part1.shape[1]:
                        diff = image_part1.shape[1] - mirror.shape[1]                        
                        image_part1 = image_part1[:,diff:]
                    else:
                        diff = mirror.shape[1] - image_part1.shape[1]                        
                        mirror = mirror[:,diff:]
                union = cv2.bitwise_and(image_part1,image_part1,mask=mirror)
                union_pixels = cv2.countNonZero(union)
                mask_pixels = cv2.countNonZero(self.mask)
                self.symmetry_v = 2*union_pixels/mask_pixels

                self.mask_ellipse = self.color_image.copy()
                elipse = cv2.ellipse(self.mask_ellipse, (int(x), int(y)), (int(axis_x / 2), int(axis_y / 2)), angle, 0, 360, (255, 0, 0), 5)


                # Distance ellipse center VS image center
                c_img = [self.mask.shape[1] / 2, self.mask.shape[0] / 2]
                c_eli = [int(y), int(x)]
                self.distance = math.sqrt(((c_img[0] - c_eli[0]) ** 2) + ((c_img[1] - c_eli[1]) ** 2))

                # Eccentricity: Moment of the center of the area with the center of the ellipse (distance)
                self.eccentricity = math.sqrt(((self.mayor_axis/2)**2)-((self.minor_axis/2)**2))/(self.mayor_axis/2)

                # Perimeter of the lamb area
                self.perimeter = cv2.arcLength(cnt, True)

                # Percentage Bounding box VS Lamb
                area_lamb = cv2.contourArea(cnt)
                bbox_area = self.width_bbox * self.height_bbox
                self.percent_area = (area_lamb * 100) / bbox_area

                self.centroid = (int(x), int(y))
                self.orientation = angle

                self.complete_info = True
                    
             
            
    def save_results(self):
        cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_color.png'), cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        depth_colorized = cv2.applyColorMap(np.uint8(cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)),cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_depth.png'), cv2.cvtColor(depth_colorized, cv2.COLOR_BGR2RGB))        
        if self.complete_info:
            cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_ellipse.png'), cv2.cvtColor(self.mask_ellipse, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_bbox.png'), cv2.cvtColor(self.mask_bbox, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_contour.png'), cv2.cvtColor(self.image_contour, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_mask.png'), cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB))
        else:
            cv2.imwrite(os.path.join(self.system.path_results,self.filename+'_mask.png'), cv2.cvtColor(self.original_mask, cv2.COLOR_GRAY2RGB))
        

        
    def calculate_height(self):
        img = self.depth_image
        _, mask = cv2.threshold(self.depth_image,900,255,cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        size_h, size_w = mask.shape
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = []
        contour_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>20000 and area<200000:
                if area>contour_area:
                    contour_area = area
                    contour = cnt
        if contour_area>500:
            x,y,w,h = cv2.boundingRect(contour)

            border = 50
            v = [x-border, x+w+border,y-border, y+h+border]
            v[0] = v[0] if v[0]>=0 else 0
            v[1] = v[1] if v[1]<size_w else size_w-1
            v[2] = v[2] if v[2]>=0 else 0
            v[3] = v[3] if v[3]<size_h else size_h-1

            img = self.depth_image[v[2]:v[3], v[0]:v[1]]
            mask = mask[v[2]:v[3], v[0]:v[1]]
        depth_value = cv2.mean(img, mask)[0]
        height = rs.rs2_deproject_pixel_to_point(self.system.intrinsics, [100,100], depth_value)[2] #xyz
        self.height = height / 1000