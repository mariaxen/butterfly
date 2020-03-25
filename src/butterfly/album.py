#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:23:05 2020

@author: maria
"""

from __future__ import division
import os
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import scipy
import pylab
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from math import atan2

# just for fun making further development easier and with joy
pi = scipy.pi
dot = scipy.dot
sin = scipy.sin
cos = scipy.cos
ar = scipy.array
rand = scipy.rand
arange = scipy.arange
plot = pylab.plot
show = pylab.show
axis = pylab.axis
grid = pylab.grid
title = pylab.title
rad = lambda ang: ang * pi / 180  # lovely lambda: degree to radian


def Rotate2D(pts, cnt, ang=pi / 4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts - cnt, ar([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])) + cnt


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    #    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

# name_of_omic = 'plasma_s'
# pix_size = 40
def create_album(DF, name_of_omic, pix_size):
    """
   Create your album that contains all the pictures you are training on
   Each picture is one omics dataset for one patient and one trimester
   For example name_of_omic = 'plasma_s'
   and pix_size = 40

    #Your dataframe is a pd df that combines all omics

    Get it from
    os.chdir('/Users/Maria/Desktop/Stanford/omics_data/Data')
    DF = pyreadr.read_r('/Users/Maria/Desktop/Stanford/omics_data/Data/omics.RData')
    DF = DF["DF"]
    """

    # Select and prepare your chosen omics
    omic = [col for col in DF if col.startswith(name_of_omic)]
    omic.append("patientID")
    omics_df = DF[omic]
    omics_df = omics_df.transpose()
    patient_IDs = omics_df.iloc[omics_df.shape[0]-1]
    omics_df = omics_df.drop(omics_df.index[omics_df.shape[0]-1])
    omics_df = pd.DataFrame(StandardScaler().fit_transform(omics_df))
    
    # omics_df = np.log(omics_df)
    pca = TSNE(perplexity=25)
    principalComponents = pca.fit_transform(omics_df)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal.component.1',
                                        'principal.component.2'])
    points = principalDf.values

    """
    Find your minimum bounding rectangle and rotate it
    """

    # If your really want to see your box, uncomment
    #    plt.scatter(principalDf['principal.component.1'],
    #                    principalDf['principal.component.2'], alpha = 0.3)
    bbox = minimum_bounding_rectangle(points)
    #    plt.fill(bbox[:,0], bbox[:,1], alpha=0.3)
    #    plt.axis('equal')
    #    plt.show()

    xDiff = bbox[2, 0] - bbox[1, 0]
    yDiff = bbox[2, 1] - bbox[1, 1]
    angle = atan2(xDiff, yDiff)

    principalDf_r = Rotate2D(principalDf.values,
                             ar([bbox[2, 0], bbox[2, 1]]), angle)
    #    bbox_r = Rotate2D(bbox,
    #                   ar([bbox[2,0],bbox[2,1]]), angle)

    #    plt.scatter(principalDf['principal.component.1'],
    #                    principalDf['principal.component.2'])
    #    plt.fill(bbox[:,0], bbox[:,1], alpha=0.4)
    #    axis('image')
    #    grid(True)
    #    show()

    #    plt.scatter(principalDf_r[:,0],
    #                principalDf_r[:,1])
    #    plt.fill(bbox_r[:,0], bbox_r[:,1], alpha=0.4)
    #    axis('image')
    #    grid(True)
    #    title('Rotate2D around a point')
    #    show()

    # Time to create a grid to base your photo on
    nx, ny = (pix_size + 1, pix_size + 1)
    x = np.linspace(min(principalDf_r[:, 0]), max(principalDf_r[:, 0]), nx)
    y = np.linspace(min(principalDf_r[:, 1]), max(principalDf_r[:, 1]), ny)

    # And now create your album
    album = []

    for p_tr in range(len(omics_df.columns)):

        principalDf_rp = pd.DataFrame(principalDf_r)
        principalDf_rp.loc[:, 2] = omics_df.loc[:, p_tr].values
        principalDf_rp.columns = ['pr1', 'pr2', 'patient_trimester']

        pic = np.zeros((pix_size, pix_size))

        for i in range(pix_size):

            for j in range(pix_size):

                pixel = principalDf_rp[((principalDf_rp['pr1'] < x[i + 1]) & (principalDf_rp['pr1'] > x[i]) &
                                        (principalDf_rp['pr2'] < y[j + 1]) & (principalDf_rp['pr2'] > y[j]))]

                if pixel.empty:
                    pic[i, j] = 0
                else:
                    pic[i, j] = np.mean(pixel['patient_trimester'])

        #    pic = pic.flatten()
        album.append(pic)

    return album, patient_IDs
    # plot_im = plt.imshow(pic, cmap = 'RdGy')
