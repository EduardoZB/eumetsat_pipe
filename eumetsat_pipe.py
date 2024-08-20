import eumdac
import datetime as dt
import shutil
import requests
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

### Select a specific area to display with pyresample

# Create some information on the reference system
area_id = 'Spain'
description = 'Geographical Coordinate System clipped on Spain'
proj_id = 'Spain'
# Specifing some parameters of the projection
proj_dict = {"proj": "longlat", "ellps": "WGS84", "datum": "WGS84"}
# Calculate the width and height of the aoi in pixels
llx = -10 # lower left x coordinate in degrees
lly = 35 # lower left y coordinate in degrees
urx = 4 # upper right x coordinate in degrees
ury = 45 # upper right y coordinate in degrees
resolution = 0.005 # target resolution in degrees
# Calculating the number of pixels
width = int((urx - llx) / resolution)
height = int((ury - lly) / resolution)
area_extent = (llx,lly,urx,ury)
# Defining the area
area_def = pr.geometry.AreaDefinition(area_id, proj_id, description, proj_dict,
                                      width, height, area_extent)
print(area_def)



### Define the function that converts the .nat file to .tif, .png or .jpg files containing the selected dataset image

def nat2tif(file, calibration, area_def, dataset, reader, label, dtype, radius,
            epsilon, nodata, out_type, bright, contrast):
    # open the file
    scn = Scene(filenames = {reader: [file]})

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
    # now we can already resample our data to the area of interest
    values = pr.kd_tree.resample_nearest(swath_def, values,
                                         area_def,
                                         radius_of_influence=radius, # in meters
                                         epsilon=epsilon,
                                         fill_value=False)
    
    # let us join our filename based on the input file's basename and its end time    
    filetime = scn.end_time
    nowutc = filetime.strftime('%H%M')
    nowday = filetime.strftime('%d%m')
    outnamev = os.path.basename(file)[:4]+'_'+nowutc+'_'+str(label)    
    outdir = './output'+nowday
    # we are going to check if the outdir exists and create it if it doesnt
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
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
        outRaster = driver.Create(outname, cols, rows, 1)
        
        # writing the metadata
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        
        # creating a new band and writting the data
        outband = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(nodata) #specified no data value by user
        outband.WriteArray(np.array(values)) # writting the values
        outRasterSRS = osr.SpatialReference() # create CRS instance
        outRasterSRS.ImportFromEPSG(4326) # get info for EPSG 4326
        outRaster.SetProjection(outRasterSRS.ExportToWkt()) # set CRS as WKT
    
        # clean up
        outband.FlushCache()
        outband = None
        outRaster = None
    
    # this output saves the image through cv2 library
    elif out_type.lower() == 'cv2':
        # add file extension to name
        outname = os.path.join(outdir, outnamev + '.png')        
        
        # preparing the data
        data = np.array(values)

        cv2out = cv2.convertScaleAbs(data, beta=bright, alpha=contrast)
        
        # save the output
        cv2.imwrite(outname, cv2out)
        
    # this output plots an image with matplotlib
    elif out_type.lower() == 'plt':
        # add file extension to name
        outname = os.path.join(outdir, outnamev + 'mpl.png')
        
        # preparing the data
        data = np.array(values)
                
        # save output
        plt.imshow(data)
        plt.gca().set(title=label, xlabel='Longitude', ylabel='Latitude')
        plt.xticks([0, cols], [area_def.area_extent[0], area_def.area_extent[2]])
        plt.yticks([0, rows], [area_def.area_extent[3], area_def.area_extent[1]])
        plt.savefig(outname)
        
    else:
        print('Not a valid output type')

### Define the funciton that converts .nat file in full color png. Resampling has to be managed by satpy

def nat2rgb(file, area, dataset, reader, label):

    # read the file
    scn = Scene(filenames = {reader:[file]})
    # get file time
    filetime = scn.end_time
    nowutc = filetime.strftime('%H%M')
    nowday = filetime.strftime('%d%m')
    outdir = './output'+nowday
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outnamev = os.path.basename(file)[:4]+'_'+nowutc+'_'+str(label)+'_'+area
    outname = os.path.join(outdir, outnamev +'.png')
    scn.load([dataset])
    # resample to europe
    local_scn = scn.resample(area)
    # save the resampled dataset/composite to disk
    local_scn.save_dataset(dataset, filename=outname)

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


### Module for extracting the files from eumetsat

# Insert your personal key and secret into the single quotes
consumer_key = 'yourAPIcredentialKEY'
consumer_secret = 'yourAPIcredentialSECRET'

credentials = (consumer_key, consumer_secret)

token = eumdac.AccessToken(credentials)

try:
    print(f"This token '{token}' expires {token.expiration}")
except requests.exceptions.HTTPError as error:
    print(f"Unexpected error: {error}")

# Load product list
datastore = eumdac.DataStore(token)

# Select Meteosat Secong Generation Seviri cam
selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')

# Retrieve datasets that match our filter
products = selected_collection.search(
    dtstart=strt,
    dtend=endt)

# Warn if dataset is empty    
if len(products) < 1:
    print('No files found for this time range')

# Get and process .nat files
for product in products:
    try:
        print(product)
    except eumdac.collection.CollectionError as error:
        print(f"Error related to the collection: '{error.msg}'")
    except requests.exceptions.ConnectionError as error:
        print(f"Error related to the connection: '{error.msg}'")
    except requests.exceptions.RequestException as error:
        print(f"Unexpected error: {error}")
    for entry in product.entries:
        print(entry)
        if entry.endswith('.nat'):
            ntr = str(entry)
            if os.path.exists(entry)==False:
                with product.open(entry=ntr) as fsrc, \
                        open(fsrc.name, mode='wb') as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                print(f'Download of file {fsrc.name} finished.')
            else:
                print(f'File {ntr} already in directory.')
            if df[1]['area_def'].lower() == 'area_def':
                warea = area_def
            else:
                warea = pr.load_area('~/anaconda3/envs/py38/lib/python3.8/site-packages/satpy/etc/areas.yaml', df[1]['area_def'])

            # Here we call the nat2tif funciton for a High-Res monochromatic picture
            if df[1]['color'].lower() == 'mono':  
                nat2tif(file = ntr, 
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
                        contrast = int(df[1]['contrast']))

            # Here we call the nat2rgb funciton for a composite image
            elif df[1]['color'].lower() == 'rgb':
                        nat2rgb(file = ntr,
                        area = df[1]['area_def'],
                        dataset = df[1]['dataset'],
                        reader = df[1]['reader'], 
                        label = df[1]['label'])
