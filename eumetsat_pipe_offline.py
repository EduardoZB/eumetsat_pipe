import datetime as dt
import numpy as np
from osgeo import gdal
from osgeo import osr
import os
import pyresample as pr
from satpy import Scene
from satpy.resample import get_area_def
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

### Select a specific area to display with pyresample

# create some information on the reference system
area_id = 'Spain'
description = 'Geographical Coordinate System clipped on Spain'
proj_id = 'Spain'
# specifing some parameters of the projection
proj_dict = {"proj": "longlat", "ellps": "WGS84", "datum": "WGS84"}
# calculate the width and height of the aoi in pixels
llx = -10 # lower left x coordinate in degrees
lly = 35 # lower left y coordinate in degrees
urx = 4 # upper right x coordinate in degrees
ury = 45 # upper right y coordinate in degrees
resolution = 0.003 # target resolution in degrees
# calculating the number of pixels
width = int((urx - llx) / resolution)
height = int((ury - lly) / resolution)
area_extent = (llx,lly,urx,ury)
# defining the area
area_def = pr.geometry.AreaDefinition(area_id, proj_id, description,
                                      proj_dict, width, height, area_extent)

### Define the function that converts the .nat file to .tif, .png or .jpg files containing the selected dataset image

def nat2tif(file, color, calibration, area_def, dataset, reader, label, dtype,
            radius, epsilon, nodata, out_type, bright, contrast, strtime,
            endtime):
    # open the file
    scn = Scene(filenames = {reader: [file]})
    filetime = scn.end_time
    if filetime<strtime or filetime>endtime:
        return
    scn.load([dataset])
    # let us extract the longitude and latitude data
    lons, lats = scn[dataset].area.get_lonlats()
    # now we can apply a swath definition for our output raster
    swath_def = pr.geometry.SwathDefinition(lons=lons, lats=lats)
    # and finally we also extract the data
    values = scn[dataset].values
    # we will now change the datatype of the arrays
    # depending on the present data this can be changed
    lons = lons.astype(dtype)
    lats = lats.astype(dtype)
    values = values.astype(dtype)

    # Handling array shape for composite images
    if color.lower() == 'mono':
        nbands = 1
    elif color.lower() == 'rgb':
        nbands = 3
        values = values.transpose(1,2,0)
    else:
        print('Not a valid color scheme')

    # now we can already resample our data to the area of interest
    values = pr.kd_tree.resample_nearest(swath_def, values,
                                         area_def,
                                         radius_of_influence=radius, # meters
                                         epsilon=epsilon,
                                         fill_value=False)
    # let us join our filename based on the input file's basename and its end time
    nowutc = filetime.strftime('%H%M')
    nowday = filetime.strftime('%d%m')
    outnamev = os.path.basename(file)[:4] +'_'+ nowutc +'_'+ area_def.area_id + str(label)  
    outdir = './output'+nowday
    # we are going to check if the outdir exists and create it if it doesnt
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Handling a bug where pyresample with natural_enh sums the values of all channels to the red channel twice
    if dataset.lower() == 'natural_enh':
        redch = np.zeros(values.shape)
        redch = (values[:,:,0] - values[:,:,1]*2 - values[:,:,2]*2)/2
        values[:,:,0] = redch
    # now we define some metadata for our raster file
    cols = values.shape[1]
    rows = values.shape[0]
    pixelWidth = (area_def.area_extent[2] - area_def.area_extent[0])/ cols
    pixelHeight = (area_def.area_extent[1] - area_def.area_extent[3])/ rows
    originX = area_def.area_extent[0]
    originY = area_def.area_extent[3] 
    
    # the output of this if is in .tif
    if out_type.lower() == 'tif':
        # add file extension to name
        outname = os.path.join(outdir, outnamev + '.tif')
        
        # here we actually create the file
        driver = gdal.GetDriverByName('GTiff')

        if color.lower() == 'mono':
            
            outRaster = driver.Create(outname, cols, rows, 1)
            # writing the metadata
            outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0,
                                       pixelHeight))
            # creating a new band and writting the data
            outband = outRaster.GetRasterBand(1)
            outband.SetNoDataValue(nodata) #specified no data value by user
            outband.WriteArray(values) # writting the values
            outRasterSRS = osr.SpatialReference() # create CRS instance
            outRasterSRS.ImportFromEPSG(4326) # get info for EPSG 4326
            outRaster.SetProjection(outRasterSRS.ExportToWkt()) # set CRS as WKT
            # clean up
            outband.FlushCache()
            outband = None
            outRaster = None
        elif color.lower() == 'rgb':
            outRaster = driver.Create(outname, cols, rows, 3)
            # writing the metadata
            outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0,
                                           pixelHeight))
            # creating a new band and writting the data
            outRaster.GetRasterBand(1).WriteArray(values[:,:,0])
            outRaster.GetRasterBand(2).WriteArray(values[:,:,1])
            outRaster.GetRasterBand(3).WriteArray(values[:,:,2])
            #outband.SetNoDataValue(nodata) #specified no data value by user
            outRasterSRS = osr.SpatialReference() # create CRS instance
            outRasterSRS.ImportFromEPSG(4326) # get info for EPSG 4326
            outRaster.SetProjection(outRasterSRS.ExportToWkt()) # set CRS as WKT
            # clean up
            outRaster.FlushCache()
            outRaster = None

    # this output saves the image through cv2 library
    elif out_type.lower() == 'cv2':
        # add file extension to name
        outname = os.path.join(outdir, outnamev + '.png')
        # scaling for mpl
        values = np.interp(values, (np.percentile(values,1), np.percentile(
                           values,99)), (0, 255))
        if color.lower() == 'rgb':
            # OpenCV inverts the channel order for some reason
            cv2out = np.zeros(values.shape)
            cv2out[:,:,0] = values[:,:,2]
            cv2out[:,:,1] = values[:,:,1]
            cv2out[:,:,2] = values[:,:,0]
            # normalize and manually correct constrast
            cv2out = cv2.convertScaleAbs(cv2out, beta=bright, alpha=contrast)
            # save the output
            cv2.imwrite(outname, cv2out)
            cv2out = None
        if color.lower() == 'mono':
            # normalize and manually correct constrast
            cv2out = cv2.convertScaleAbs(values, beta=bright, alpha=contrast)
            # save the output
            cv2.imwrite(outname, cv2out)
            cv2out = None        
    # this output plots an image with matplotlib
    elif out_type.lower() == 'plt':
        # add file extension to name
        outname = os.path.join(outdir, outnamev + 'mpl.png') 
        # scaling for mpl
        values = np.interp(values, (np.percentile(values,1), np.percentile(
                           values,99)), (0, 1))
        # defining coordinate reference system
        crs = area_def.to_cartopy_crs()
        # Initiatie a subplot and axes with the CRS information defined above
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs),
                               figsize=(10, 8))
        # Add coastline features to the plot
        ax.coastlines()
        # Define a grid to be added to the plot
        gl = ax.gridlines(draw_labels=True, linestyle='--',
                          xlocs=range(int(originX),
                          int(area_def.area_extent[2]),5),
                          ylocs=range(int(area_def.area_extent[1]),
                          int(originY),5))
        gl.top_labels=False
        gl.right_labels=False
        gl.xformatter=LONGITUDE_FORMATTER
        gl.yformatter=LATITUDE_FORMATTER
        gl.xlabel_style={'size':14}
        gl.ylabel_style={'size':14}

        ax.set_global()
        # In the end, we can plot our image data...
        img = ax.imshow(values, transform=crs, extent=crs.bounds, origin="upper")
        # Define a title for the plot
        plt.title(dataset + " image of " + area_def.area_id + 
                  ", recorded by MSG at " + nowutc + ' hours of ' +
                  filetime.strftime('%d/%m/%Y'), fontsize=12, pad=20.0)
        # save output
        fig.savefig(outname)
        
    else:
        print('Not a valid output type. Select TIF, CV2 or MPL')
    values = None

# Access the recipe and transform the data as requested

df=pd.read_table('eumetsat_recipe.txt',header=None,index_col=0)

if df[1]['starttime'].lower() == 'recent':
    strt = dt.datetime.utcnow()-dt.timedelta(seconds=1200)
    endt = dt.datetime.utcnow()
elif isinstance(dt.datetime.strptime(df[1]['starttime'],'%d/%m/%Y %H:%M'), dt.datetime):
    strt = dt.datetime.strptime(df[1]['starttime'],'%d/%m/%Y %H:%M')
    endt = dt.datetime.strptime(df[1]['endtime'],'%d/%m/%Y %H:%M')
else:
    print('Not a valid time interval. Please, write RECENT or a start and finish dates in D/M/Y HH:MM format')

    
# Look for .nat files in working directory and process them according to recipe

entries = os.listdir()
for entry in entries:
    if entry.endswith('.nat'):
        ntr = str(entry)
        if df[1]['area_def'].lower() == 'area_def':
            warea = area_def
        else:
            warea = pr.load_area('/home/z/anaconda3/envs/py38/lib/python3.8/site-packages/satpy/etc/areas.yaml', df[1]['area_def'])
        nat2tif(file = ntr, 
                color = df[1]['color'],
                calibration = df[1]['calibration'],
                area_def = warea,  
                dataset = df[1]['dataset'], 
                reader = df[1]['reader'], 
                label = df[1]['label'], 
                dtype = df[1]['dtype'], 
                radius = int(df[1]['radius']), 
                epsilon = float(df[1]['epsilon']), 
                nodata = float(df[1]['nodata']),
                out_type = df[1]['out_type'],
                bright = int(df[1]['brightness']),
                contrast = int(df[1]['contrast']),
                strtime = strt,
                endtime = endt)
