#%%
import glob
import os
from unittest.mock import Mock
import cv2
import pyrealsense2 as rs
import numpy as np
import math

import pandas as pd

import os
import time
from collections import deque
from datetime import timedelta
from enum import Enum, auto
from statistics import mean
from typing import NamedTuple

import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import UnivariateSpline
from skimage import morphology
from skimage.measure import label, regionprops
import imutils
import datetime


class ImageExpert:
    def __init__(self, parent_window, width, height):
        self.parent_window = parent_window
        self.centre_x, self.centre_y = width // 2, height // 2

    def create_mask(self, strategy='fast'):
        self.clip_depth_image()
        mask = self.get_foreground_mask_fast_strategy()
        return mask

    def clip_depth_image(self):
        camera = self.parent_window.camera
        if camera.depth_image is None:
            return
        camera.depth_image[camera.depth_image < camera.min_depth] = 0
        camera.depth_image[camera.depth_image > camera.max_depth] = 0


    def calculate_bounding_rectangle(self, image, parameters):
        x = parameters['x']
        y = parameters['y']
        width = parameters['width']
        height = parameters['height']
        cv.rectangle(image, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 2)
        return image


    def calculate_mask_parameters(self, mask):
        if not np.any(mask):
            return None

        heightMask, widthMask = mask.shape
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        if not contours:
            return None

        cnt = np.vstack(contours)

        # Check pixels within limits
        img_coord_x_up_left_corner, img_coord_y_up_left_corner, width, height = cv.boundingRect(cnt)

        img_coord_x_up_right_corner = img_coord_x_up_left_corner + width - 1
        img_coord_y_up_right_corner = img_coord_y_up_left_corner

        img_coord_x_bottom_left_corner = img_coord_x_up_left_corner
        img_coord_y_bottom_left_corner = img_coord_y_up_left_corner + height - 1

        are_corners_within = ((0 <= img_coord_x_up_left_corner < widthMask - 1) and
                              (0 <= img_coord_y_up_left_corner < heightMask - 1) and
                              (0 <= img_coord_x_up_right_corner < widthMask) and
                              (0 <= img_coord_y_up_right_corner < heightMask) and
                              (0 <= img_coord_x_bottom_left_corner < widthMask) and
                              (0 <= img_coord_y_bottom_left_corner < heightMask)
                              )

        if not are_corners_within:
            return None

        camera = self.parent_window.camera
        depth_value = cv.mean(camera.depth_image, mask)[0]

        up_right_corner = rs.rs2_deproject_pixel_to_point(camera.intrinsics,
                                                          [img_coord_x_up_right_corner,
                                                           img_coord_y_up_right_corner],
                                                          depth_value)
        up_right_corner = np.array(up_right_corner)

        up_left_corner = rs.rs2_deproject_pixel_to_point(camera.intrinsics,
                                                         [img_coord_x_up_left_corner,
                                                          img_coord_y_up_left_corner],
                                                         depth_value)
        up_left_corner = np.array(up_left_corner)

        bottom_left_corner = rs.rs2_deproject_pixel_to_point(camera.intrinsics,
                                                             [img_coord_x_bottom_left_corner,
                                                              img_coord_y_bottom_left_corner],
                                                             depth_value)
        bottom_left_corner = np.array(bottom_left_corner)

        distancia_horizontal = np.linalg.norm(up_left_corner[:2] - up_right_corner[:2])
        distancia_vertical = np.linalg.norm(up_left_corner[:2] - bottom_left_corner[:2])

        area_rectangulo_real = distancia_horizontal * distancia_vertical
        area_rectangulo_pixeles = width * height
        escala = area_rectangulo_real / area_rectangulo_pixeles
        area_mascara_pixeles = cv.countNonZero(mask)
        area_mascara_real = area_mascara_pixeles * escala


        # Elllipse
        cnt = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = imutils.grab_contours(cnt)
        contours = None
        countour_area = 0
        for c in cnt:
            area = cv2.contourArea(c)
            if area > 10000 and area > countour_area:
                contours= c
                countour_area = area

        if countour_area == 0:
            return None
        (x, y), (axis_x, axis_y), angle = cv2.fitEllipse(contours)
        if axis_x > axis_y:
            MA = axis_x
            ma = axis_y
        else:
            MA = axis_y
            ma = axis_x

        # symmetry
        matrix = cv2.getRotationMatrix2D( center=(x,y), angle=angle, scale=1 )
        image = cv2.warpAffine( src=mask, M=matrix, dsize=(mask.shape[1]+200, mask.shape[0]+200) )
        MA_2 = int(MA/2)
        ma_2 = int(ma/2)
        x_int = int(x)
        y_int = int(y)
        x_sim = x_int - MA_2
        y_sim = y_int - ma_2
        x_sim2 = x_sim+int(MA)
        y_sim2 = y_sim+int(ma)
        if(x_sim<0):
            x_sim=int(0)
        if(y_sim<0):
            y_sim=int(0)
        image_bbox = image[y_sim:y_sim2, x_sim:x_sim2]

        image_part1 = image[y_sim:y_sim2, x_sim:int(x)]
        image_part2 = image[y_sim:y_sim2, int(x):x_sim2]
        mirror=cv2.flip(image_part2,1)
        if(image_part1.shape[0]!=mirror.shape[0]):
            mirror = mirror[:image_part1.shape[0], :]
        if(image_part1.shape[1]!=mirror.shape[1]):
            mirror = mirror[:,:image_part1.shape[1]]
        union = cv2.bitwise_and(image_part1,image_part1,mask=mirror)
        union_pixels = cv.countNonZero(union)
        mask_pixels = cv.countNonZero(mask)
        symmetry = 2*union_pixels/mask_pixels

        elipse = cv2.ellipse(mask, (int(x), int(y)), (int(axis_x / 2), int(axis_y / 2)), angle, 0, 360, (255, 0, 0), 2)

        label_img = label(elipse)
        regions = regionprops(label_img)
        if (len(regions) == 1):
            props = regions[0]
        else:
            return None

        # Bounding box
        minr, minc, maxr, maxc = props.bbox

        # Percentage Bounding box VS Lamb
        area_lamb = cv2.contourArea(contours)
        percent_area = (area_lamb * 100) / props.bbox_area

        # Distance ellipse center VS image center
        c_img = [mask.shape[1] / 2, mask.shape[0] / 2]
        c_eli = [props.centroid[1], props.centroid[0]]
        distance = math.sqrt(((c_img[0] - c_eli[0]) ** 2) + ((c_img[1] - c_eli[1]) ** 2))

        # Eccentricity: Moment of the center of the area with the center of the ellipse (distance)
        eccentricity = math.sqrt(((MA/2)**2)-((ma/2)**2))/(MA/2)

        # Perimeter of the lamb area
        perimeter = cv2.arcLength(contours, True)

        return {'area': area_mascara_real,
                'x': img_coord_x_up_left_corner,
                'y': img_coord_y_up_left_corner,
                'width': width,
                'height': height,
                #'contours': contours,
                'major axis': MA,
                'minor axis': ma,
                'centroid': props.centroid,
                'orientation': props.orientation,
                '% area': percent_area,
                'center distance': distance,
                'eccentricity': eccentricity,
                'perimeter': perimeter,
                'symmetry': symmetry}

    def clean_mask(self, mask_in):
        diskSize= 9

        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(mask_in, cv2.MORPH_CLOSE, kernel)

        mask_depth = closing.astype(bool)
        # Removing small objects
        cleaned_mask_depth = morphology.remove_small_objects(mask_depth, min_size=500, connectivity=1)

        # Each object is labelled with a number from 0 to number of regions - 1
        ret, markers = cv.connectedComponents(cleaned_mask_depth.astype(np.uint8) * 255)
        markers = markers + 1  # So we will label background as 0 and regions from 1 to number of regions
        markers[cleaned_mask_depth == 0] = 0  # background is region 0

        # Removing regions touching the borders of the image
        etiquetas_borde_superior = np.unique(markers[0, :])
        etiquetas_borde_inferior = np.unique(markers[-1, :])
        etiquetas_borde_izquierdo = np.unique(markers[:, 0])
        etiquetas_borde_derecho = np.unique(markers[:, -1])
        etiquetas_bordes = np.unique(np.concatenate([etiquetas_borde_superior,
                                                     etiquetas_borde_inferior,
                                                     etiquetas_borde_izquierdo,
                                                     etiquetas_borde_derecho]))
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

    def areas_big_contours(self, mask):
        cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas= []
        for cnt in cnts:
            area = cv.contourArea(cnt)
            if area>5000:
                x,y,w,h = cv.boundingRect(cnt)
                limit = True if x==0 or y ==h or (x+w)==mask.shape[1] or (y+h)==mask.shape[0] else False
                if not limit:
                    areas.append(area)
        areas.sort()
        return areas

    def get_foreground_mask_fast_strategy(self):
        img = cv2.normalize(self.parent_window.camera.color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #Apply sharpen kernel
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
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
        lum_areas = self.areas_big_contours(lum_bin)
        kernel = np.ones((5,5),np.uint8)
        if len(lum_areas)==1:
            lum_closing = cv2.morphologyEx(lum_bin, cv2.MORPH_CLOSE, kernel)
            closing_areas = self.areas_big_contours(lum_closing)
            if len(closing_areas)==1:
                final_mask= lum_closing
                self.mask_method = 'LAB_closing'
            else:
                lum_opening = cv2.morphologyEx(lum_bin, cv2.MORPH_OPEN, kernel)
                opening_areas = self.areas_big_contours(lum_opening)
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

        mask = self.clean_mask(final_mask)
        self.mask = final_mask
        self.cleaned_mask = mask.copy()

        return mask


class Processor:
    def __init__(self):
        self.image_expert=ImageExpert(parent_window=Mock(), width=848, height=480)
        self.image_expert.parent_window.camera = Mock()
        self.image_expert.parent_window.camera.intrinsics = rs.intrinsics()
        self.image_expert.parent_window.camera.intrinsics.width = 848
        self.image_expert.parent_window.camera.intrinsics.height = 480
        self.image_expert.parent_window.camera.intrinsics.fx = 424.5785827636719
        self.image_expert.parent_window.camera.intrinsics.fy = 424.5785827636719
        self.image_expert.parent_window.camera.intrinsics.ppx = 422.18994140625
        self.image_expert.parent_window.camera.intrinsics.ppy = 244.84666442871094
        self.image_expert.parent_window.camera.intrinsics.model = rs.distortion.brown_conrady
        self.image_expert.parent_window.camera.min_depth = 150
        self.image_expert.parent_window.camera.max_depth = 1150

    def process_image(self, depth_filename, color_filename, mosaic_path):
        color_image = np.load(color_filename,allow_pickle=True)
        depth_image = np.load(depth_filename,allow_pickle=True)
        depth_colorized = cv2.applyColorMap(np.uint8(cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)),
                                            cv2.COLORMAP_JET)

        self.image_expert.parent_window.camera.color_image = color_image
        self.image_expert.parent_window.camera.depth_image = depth_image

        mask = self.image_expert.create_mask(strategy='fast')
        if mask is None:
            return None
        mask_parameters = self.image_expert.calculate_mask_parameters(mask)
        if mask_parameters is None:
            return None

        foreground_image =  cv2.cvtColor(self.image_expert.cleaned_mask, cv2.COLOR_GRAY2RGB )
        rectangle_image = self.image_expert.calculate_bounding_rectangle(image=foreground_image, parameters=mask_parameters)

        fig_1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        fig_2 = cv2.cvtColor(self.image_expert.cleaned_mask, cv2.COLOR_RGB2BGR)
        fig_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        fig_4 = cv2.cvtColor(rectangle_image, cv2.COLOR_RGB2BGR)
        mosaico_imagen = cv2.vconcat([cv2.hconcat([fig_1, fig_2]), cv2.hconcat([fig_3, fig_4 ])])
        output_filename = os.path.join(mosaic_path, os.path.basename(depth_filename).replace('.npy', '_mosaico.png'))
        cv2.imwrite(output_filename, mosaico_imagen)


        file_name=os.path.basename(depth_filename)[:-4]
        test_path = mosaic_path[:mosaic_path[:-1].rindex('/')]+'/mask/'
        plt.imshow(self.image_expert.sharp_img),plt.axis('off')
        plt.savefig(os.path.join(test_path,file_name+'_sharp.png'), dpi=200, bbox_inches='tight',pad_inches = 0)
        cv2.imwrite(os.path.join(test_path,file_name+'_lab.png'), cv2.cvtColor(self.image_expert.lab_img, cv2.COLOR_GRAY2RGB))
        cv2.imwrite(os.path.join(test_path,file_name+'_lab_mask.png'), cv2.cvtColor(self.image_expert.lab_mask, cv2.COLOR_GRAY2RGB))
        cv2.imwrite(os.path.join(test_path,file_name+'_mask.png'), cv2.cvtColor(self.image_expert.mask, cv2.COLOR_GRAY2RGB))
        cv2.imwrite(os.path.join(test_path,file_name+'_cleanedmask.png'), cv2.cvtColor(self.image_expert.cleaned_mask, cv2.COLOR_GRAY2RGB))
        cv2.imwrite(os.path.join(test_path,file_name+'_original.png'), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))


        return [mask_parameters['area'],
                mask_parameters['width'],
                mask_parameters['height'],
                mask_parameters['major axis'],
                mask_parameters['minor axis'],
                mask_parameters['% area'],
                mask_parameters['eccentricity'],
                mask_parameters['perimeter'],
                mask_parameters['symmetry'],
                self.image_expert.mask_method]
