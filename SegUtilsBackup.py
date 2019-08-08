import os
    
#if 'GDAL_DATA' not in os.environ:
#   os.environ['GDAL_DATA'] = r"C:\Anaconda\pkgs\libgdal-2.3.3-h10f50ba_0\Library\share\gdal"



import rasterio as rio
#from rasterio.plot import show
import numpy as np

#from PIL import Image
#import math
#from osgeo import gdal
#from osgeo import ogr
#from osgeo import gdalconst
#from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
#from osgeo import gdal
#from osgeo import ogr
#from osgeo import gdalconst
from osgeo import osr, gdalconst, ogr, gdal
#import os, sys
#import ogr
from math import floor
import geopandas
import json
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
import shapely
import rasterio
import scipy.misc
import subprocess
import matplotlib.pyplot as plt
from PIL import Image as pilimage
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from tqdm import tqdm

import pandas
import rasterio 
#from rasterio.mask import mask
#from rasterio.features import shapes
#from shapely.geometry import shape
#from matplotlib import pyplot as plt
import SegUtils
#import numpy as np
#import shapely
#from shapely.geometry import Polygon
import aerialImageRetrieval


def download_quadrant_images_from_bing(bboxes,TempPath,DownloadedPath):
    from tilesystem import TileSystem
    for i in range(0,len(bboxes)):
        quadrant=bboxes.iloc[i]
        dst_filename = DownloadedPath+quadrant.QUADRANT+'_bing_18.tif'
        new_image_file = Path(dst_filename)
        if not new_image_file.exists():
            print("Downloading Image for Quadrant: ",quadrant.QUADRANT)
            bbox=quadrant.geometry.bounds
            imgretrieval=aerialImageRetrieval.AerialImageRetrieval(bbox[1], bbox[0], bbox[3], bbox[2],TempPath)
            if imgretrieval.max_resolution_imagery_retrieval():
                print("Successfully retrieved the image with maximum resolution!")
            else:
                print("Cannot retrieve the desired image! (Possible reason: expected tile image does not exist.)")
            #Get the file fromthe output folder (normally there should be only one)
            src_filename =TempPath+os.listdir(TempPath)[0]
            
            src_ds = gdal.Open(src_filename)
            format = "GTiff"
            driver = gdal.GetDriverByName(format)  
            dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)
            
            ###Calculate the raster details
            TileSystem.ground_resolution(bbox[0],18)#(resoltuion in meters)
            src_ds.RasterXSize
            src_ds.RasterYSize
            xres_deg=(bbox[2]-bbox[0])/src_ds.RasterXSize
            yres_deg=(bbox[3]-bbox[1])/src_ds.RasterYSize
            upperleftx_deg=bbox[0]#upperleftx
            upperlefty_deg=bbox[3]#upperlefty
        
            # Specify raster location through geotransform array
            # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
            gt = [upperleftx_deg, xres_deg, 0, upperlefty_deg, 0, -yres_deg]
            dst_ds.SetGeoTransform(gt)    
            # Get raster projection
            srs = osr.SpatialReference()
            srs.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
            dest_wkt = srs.ExportToWkt()
            dst_ds.SetProjection(dest_wkt)    
            # Close files
            dst_ds = None
            src_ds = None
            #delete the file from the output folder
            os.remove(src_filename)
        else:
            print("Image already exists, skipping.")

def split_coco(coco,train_partition):
    #Get image ids which are referenced in annotations
    image_ids = list(set([x['image_id'] for x in coco['annotations']]))
    n_orig=len(image_ids)
    n_train=round(n_orig*train_partition)
    #randomly split in 2
    np.random.shuffle(image_ids)
    id_train=image_ids[0:n_train]
    id_val=image_ids[n_train:]
    #get images split
    im_train=[x for x in coco['images'] if x['id'] in id_train]
    im_val=[x for x in coco['images'] if x['id'] in id_val]
    #get annotations split
    anno_train=[x for x in coco['annotations'] if x['image_id'] in id_train]
    anno_val=[x for x in coco['annotations'] if x['image_id'] in id_val]
    len(coco["annotations"])
    print("splitting a COCO Dataset of "+str(n_orig)+" into two partitions of "+str(len(id_train))+" and "+str(len(id_val))+" images")
    print("thereby also splitting "+str(len(coco["annotations"]))+" corresponding annotations into two partitions of "+str(len(anno_train))+" and "+str(len(anno_val))+ " annotations")
    
    Inf=coco["info"].copy()
    Lic=coco["licenses"].copy()
    Cat=coco["categories"].copy()
    TrainCOCO={"info":Inf,"licenses":Lic,"images":im_train.copy(),"annotations":anno_train.copy(),"categories":Cat}
    ValCOCO={"info":Inf,"licenses":Lic,"images":im_val.copy(),"annotations":anno_val.copy(),"categories":Cat}
    return(TrainCOCO,ValCOCO)


def merge_cocos(COCOA,COCOB):
    CatA=COCOA["categories"].copy()
    CatB=COCOB["categories"].copy()
    CatC=CatA.copy()
    #ids_a=[x['id'] for x in CatA]
    ids_b=[x['id'] for x in CatB]
    ids_c=[x['id'] for x in CatC]
    names_a=[x['name'] for x in CatA]
    names_b=[x['name'] for x in CatB]
    #names_c=[x['name'] for x in CatC]
    Reclass_Matrix=np.zeros((len(ids_b), 2))

    for i in range(0,len(names_b)):
        name_b=names_b[i]
        id_b=ids_b[i]
        #IF THE NAME FROM B IS NOT ALREADY IN A, WE CREATE A NEW CATEGORY WITH A NEW ID
        if(name_b not in names_a):         
             newid=len(ids_c)
             newcat=CatB[i].copy()
             CatC.append(newcat)
             ids_c.append(newid)
             CatC[-1].update(id=newid)
             Reclass_Matrix[i,:]=(id_b,newid)
        # IF THE NAME FROM B IS ALREADY IN A, NOTE THAT ID IN THE RECLASS MATRIX
        if name_b in names_a:
            newid=names_a.index(name_b)
            Reclass_Matrix[i,:]=(id_b,newid)
    #LOOP OVER ALL THE ANNOTATIONS IN B AND RECLASS THEM ACCORDING TO THE MATRIX
    for i in range(0, len(COCOB["annotations"])):
        id_old=COCOB["annotations"][i]["category_id"]
        #Grab the new id from the reclass matrix
        newid=int(Reclass_Matrix[np.where(Reclass_Matrix[:,0]==id_old),1])
        #Assign
        COCOB["annotations"][i]["category_id"]
    InfC=COCOA["info"].copy()
    LicC=COCOA["licenses"].copy()
    ImgC=COCOA["images"].copy()+COCOB["images"].copy()
    AnnC=COCOA["annotations"].copy()+COCOB["annotations"].copy()
    MergedCOCO={"info":InfC,"licenses":LicC,"images":ImgC,"annotations":AnnC,"categories":CatC}
    return(MergedCOCO)


def plot_geometries_over_images(Geometries,imagepath,lower,upper):
    for key in sorted(Geometries.keys())[lower:upper]:
        print(key)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        #Load the image as an array
        img = np.asarray(pilimage.open(imagepath+key))
        plt.imshow(img, interpolation='none')
    
        mp = Geometries[key]
        #mp.bounds
        #mp[1].exterior.bounds
        patches = [PolygonPatch(p, fill=False, ec='r', alpha=1, lw=0.7, zorder=1) for p in mp]
        plt.gca().add_collection(PatchCollection(patches, match_original=True))
        plt.show()
        
        

def coco_to_geometry(cocofile):
    
    #Preallocate an empty dictionary which will hold the extreacte dgeometries
    extracted_geometries = {}
    image_ids = list(set([x['image_id'] for x in cocofile['annotations']])) #unordered
    #Get file names and image ids for the images that contain annotations
    file_names = [x['file_name'] for x in cocofile['images'] if x['id'] in image_ids]   #ordered
    image_ids = [x['id'] for x in cocofile['images'] if x['id'] in image_ids]  #now also ordered
    
    
    #For each image id:
    for image_id, file_name in zip(image_ids, file_names):
        
        #Grab the annotations
        annotations = [x for x in cocofile['annotations'] if x['image_id'] == image_id]
        #Grab the coordinates
        coords = [coords['segmentation'][0] for coords in annotations] #[0] is necessary because source is a list of lists
        # Create a polygon from the coords
        mp = MultiPolygon([Polygon(np.array(coord).reshape((int(len(coord) / 2), 2))) for coord in coords])
        extracted_geometries[str(file_name)] = mp #die polygone werden den filenames zugeordnet
        
        
    return extracted_geometries
    


def grid_raster_and_annotations_to_coco(src_grid,src_im,src_ann,dst_im,prefix,dst_ann,invert_y=False,category_field="Class"):
    grid = geopandas.read_file(src_grid)#Open The Grid shp
    data = rio.open(src_im)# Open the raster tif
    ntiles=len(grid)
    
    
    print("==============================================")
    print("Preprocessing Annotation Shapefile")
    anno=geopandas.read_file(src_ann)
    anno=anno[anno["geometry"].notnull()]
    anno=anno.explode()
    anno['geometry'] = anno.geometry.buffer(0)   
    anno.to_file(driver = 'ESRI Shapefile', filename= dst_ann+"preprocessed_annotations.shp")
    print("Done Preprocessing Annotation Shapefile")
    print("==============================================")
    ##ERSTELLE KATEGORIEN FÜR COCO FORMAT
    
    
    coco_info ={}
    coco_licenses={}
    coco_images = []  #List of dict of {file_name,id, height,width}
    coco_categories=[] # List of dict of{supercategory, id, name}
    coco_annotations=[] # List of dict of {id, category_id,iscrowd,segmentation[[LISTofLISTS]],image_id,area,bbox[]}
    category_log=[] #list of classnames
    category_id_log=[]
    anno_id_log=[] #list of anno nids
    im_id_log=[]#list of image ids
    
    
    print("==============================================")
    print("Starting to process image and annotation files")
    for i in range(0,ntiles): #for every object in grid
    #for i in range(0,100):
        im_id=("{:05d}".format(prefix))+"0123"+("{:05d}".format(i))#Unique id: erste 5 ziffern=prefix, 0123 bedeutet dass es image ist, letzte 5 ziffern id
        outfile_im=dst_im+im_id+".tif"
        outfile_ann=dst_ann+"Tile_%d_Annotation.shp" % i
        #anno_tilename = f'COCO_train2016_000000{100000+i}'
        
        print("Clipping tile" + str(i) + " of "+str(ntiles))
        #print("Target: "+outfile_im)
        coords=getFeatures(grid,i) #get the polygon
        Tile, out_transform = mask(dataset=data, shapes=coords, crop=True) #crop the raster to the polygon
        #Suche bei der gelegenheit die auflösung raus, die wir sehr viel später in dieser funktion brauchen um die annotations zu skalieren
        xres=abs(out_transform[0])
        yres=abs(out_transform[4])
        out_meta = data.meta.copy() #get a copy of the metadata of the raster
        #out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform,"crs": '+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'}) #update the meta for the cropped raster
        out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform}) #update the meta for the cropped raster
        #make the dir if necessary
        if not os.path.exists(dst_im):
            os.makedirs(dst_im)
        #write the image
        with rio.open(outfile_im, "w", **out_meta) as dest:
                dest.write(Tile) 
        
        #UPDATE COCO FOR THIS IMAGE
        Image_descr={"file_name":im_id+".tif","id":im_id,"height":out_meta["height"],"width":out_meta["width"]}
        coco_images.append(Image_descr)
        im_id_log.append(im_id) #also register the image id in the log



                                 
        print("Clipping annotation" + str(i)+ "of "+str(ntiles) )
        
        
        
        
        #print("Target: "+outfile_ann)
        (xmin,ymin,xmax,ymax)=grid.bounds[i:i+1].iloc[0,:]
        subprocess.call("ogr2ogr -clipsrc %d %d %d %d "%(xmin,ymin,xmax,ymax) +'"'+ outfile_ann+'"'+" "+'"'+ dst_ann+"preprocessed_annotations.shp"+'"' ,shell=True)
    
    
    
        #print("Processing Annotations")
        anno=geopandas.read_file(outfile_ann)
        
        
        #drop missing geometries
        anno=anno[anno["geometry"].notnull()]
        if anno.empty:
            print("No Valid Annotations found, skipping Tile ", str(i))
            continue
        #EXPLODE to remove multipart features
        anno=anno.explode()
        #anno.buffer(0)        
        #simplify remaining geometries CAN CAUSE CRASHES
        #if not all(anno.geometry.is_empty): 
        #    anno.anno = anno.simplify(1, preserve_topology=True)  
        #drop small geometries   
        anno = anno[anno.geometry.area * (10 * 10) > 500] #enterne ganz kleine polys
        if anno.empty:
            print("No Valid Annotations found, skipping Tile ", str(i))
            continue        
        #anno.to_file(driver = 'ESRI Shapefile', filename= "result.shp")
        #print("Converting to Local Coordinates")
        #width=xmax-xmin
        height=ymax-ymin
        npoly=len(anno)
        
        
        
        # ADD CLASSES FOR THIS ANNO TO COCO CLASS, IF THEY ARE NOT REGISTERED THERE YET
        anno_classes=anno[category_field]          #get classes in current anno   (in relevant category column)
        anno_classes_uniq= list(set(anno_classes))            #get uniques
        for k in anno_classes_uniq:     
            if k not in category_log:            #wenn es die category noch ned gibt
                new_category_id=len(coco_categories)
                print("creating new category: ", k, " under id: ", new_category_id)   
                
                Class_descr={"supercategory":"LandCover","id":new_category_id,"name":k}  #mach neu (ids start at 0)
                coco_categories.append(Class_descr) #hänge an
                category_log.append(k) # registriere die neue klasse namentlich im log
                category_id_log.append(new_category_id)   #registriere die neue id im log
        
        
        
        
        
        
        
        
        
        #For each poly in the clipped shape:
        for j in range(0,npoly):

            #UPDATE THE GEOMETRY
            oldpoly=anno.exterior.iloc[j]
            poly_x, poly_y=oldpoly.coords.xy
            newpoly=shapely.geometry.Polygon([[(x - xmin)/xres, (y - ymin)/yres] for x, y in zip(poly_x, poly_y)])
            if invert_y:
                #print("Also inverting Y Axis")
                #Die coordinaten werden auf tile coordinaten transformiert(subtraktion), mittels der auflösung werden die meter in pixelwerte transformiert, die y axe gespiegelt
                newpoly=shapely.geometry.Polygon([[(x - xmin)/xres, (height-(y - ymin))/yres] for x, y in zip(poly_x, poly_y)])         
            anno.geometry.iloc[j]=newpoly
            
            
            
            #ADD THE ANNOTATIONS TO COCO ANNOTATIONS
            anno_id=("{:05d}".format(prefix))+"0987"+("{:05d}".format(len(anno_id_log)))#Unique id: erste 5 ziffern=prefix, 0987 bedeutet dass es anno ist, letzte 5 ziffern id
            image_id=im_id
            category_name=anno[category_field].values[j]  #find the category name
            category_id=category_id_log[category_log.index(category_name)] #find the corresponding id in this category name (this is where the logs come in)
            poly_x_n, poly_y_n=newpoly.exterior.coords.xy
            segmentation=[[round(val,3) for pair in zip(poly_x_n, poly_y_n) for val in pair]] #interleave x and y coordinates (von stackoverflow geklaut)
            bbox=newpoly.bounds
            area=newpoly.area
            iscrowd=0
            
            Anno_descr={"id":anno_id,"category_id":category_id,"iscrowd":iscrowd,"segmentation":segmentation,"image_id":image_id,"area":area,"bbox":bbox}
            coco_annotations.append(Anno_descr)
            anno_id_log.append(anno_id)  #also register the anno id in the log
            
            
            
            
            

                    
    
    
    
    print("Finished processing image and annotation files")
    print("==============================================")
    
    print("Creating Coco-styled Database")
    coco_style_db={"info":coco_info,"licenses":coco_licenses,"images":coco_images,"annotations":coco_annotations,"categories":coco_categories}
    registers=[category_log,category_id_log,anno_id_log,im_id_log]
    
    return(coco_style_db,registers)




def grid_raster_and_annotations_to_coco_v2(src_grid,src_im,src_ann=None,dst_im=None,prefix=None,dst_ann=None,invert_y=False,category_field="Class"):
    grid = geopandas.read_file(src_grid)#Open The Grid shp
    data = rio.open(src_im)# Open the raster tif
    ntiles=len(grid)
    
    if src_ann:      
        print("==============================================")
        print("Preprocessing Annotation Shapefile")
        anno=geopandas.read_file(src_ann)
        anno=anno[anno["geometry"].notnull()]
        anno=anno.explode()
        anno['geometry'] = anno.geometry.buffer(0)   
        anno.to_file(driver = 'ESRI Shapefile', filename= dst_ann+"preprocessed_annotations.shp")
        print("Done Preprocessing Annotation Shapefile")
        print("==============================================")
    ##ERSTELLE KATEGORIEN FÜR COCO FORMAT
    
    
    coco_info ={}
    coco_licenses={}
    coco_images = []  #List of dict of {file_name,id, height,width}
    coco_categories=[] # List of dict of{supercategory, id, name}
    coco_annotations=[] # List of dict of {id, category_id,iscrowd,segmentation[[LISTofLISTS]],image_id,area,bbox[]}
    category_log=[] #list of classnames
    category_id_log=[]
    anno_id_log=[] #list of anno nids
    im_id_log=[]#list of image ids
    
    
    print("==============================================")
    print("Starting to process image and annotation files")
    for i in range(0,ntiles): #for every object in grid
    #for i in range(0,100):
        im_id=("{:05d}".format(prefix))+"0123"+("{:05d}".format(i))#Unique id: erste 5 ziffern=prefix, 0123 bedeutet dass es image ist, letzte 5 ziffern id
        outfile_im=dst_im+im_id+".tif"
        if src_ann:    
            outfile_ann=dst_ann+"Tile_%d_Annotation.shp" % i
        #anno_tilename = f'COCO_train2016_000000{100000+i}'
        
        print("Clipping tile" + str(i) + " of "+str(ntiles))
        #print("Target: "+outfile_im)
        coords=getFeatures(grid,i) #get the polygon
        Tile, out_transform = mask(dataset=data, shapes=coords, crop=True) #crop the raster to the polygon
        #Suche bei der gelegenheit die auflösung raus, die wir sehr viel später in dieser funktion brauchen um die annotations zu skalieren
        xres=abs(out_transform[0])
        yres=abs(out_transform[4])
        out_meta = data.meta.copy() #get a copy of the metadata of the raster
        #out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform,"crs": '+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'}) #update the meta for the cropped raster
        out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform}) #update the meta for the cropped raster
        #make the dir if necessary
        if not os.path.exists(dst_im):
            os.makedirs(dst_im)
        #write the image
        with rio.open(outfile_im, "w", **out_meta) as dest:
                dest.write(Tile) 
        
        #UPDATE COCO FOR THIS IMAGE
        Image_descr={"file_name":im_id+".tif","id":im_id,"height":out_meta["height"],"width":out_meta["width"]}
        coco_images.append(Image_descr)
        im_id_log.append(im_id) #also register the image id in the log



        if src_ann:
                                 
            print("Clipping annotation" + str(i)+ "of "+str(ntiles) )
            #print("Target: "+outfile_ann)
            (xmin,ymin,xmax,ymax)=grid.bounds[i:i+1].iloc[0,:]
            subprocess.call("ogr2ogr -clipsrc %d %d %d %d "%(xmin,ymin,xmax,ymax) + outfile_ann+" "+ dst_ann+"preprocessed_annotations.shp" ,shell=True)
            #print("Processing Annotations")
            anno=geopandas.read_file(outfile_ann)
            
            
            #drop missing geometries
            anno=anno[anno["geometry"].notnull()]
            if anno.empty:
                print("No Valid Annotations found, skipping Tile ", str(i))
                continue
            #EXPLODE to remove multipart features
            anno=anno.explode()
            #anno.buffer(0)        
            #simplify remaining geometries CAN CAUSE CRASHES
            #if not all(anno.geometry.is_empty): 
            #    anno.anno = anno.simplify(1, preserve_topology=True)  
            #drop small geometries   
            anno = anno[anno.geometry.area * (10 * 10) > 500] #enterne ganz kleine polys
            if anno.empty:
                print("No Valid Annotations found, skipping Tile ", str(i))
                continue        
            #anno.to_file(driver = 'ESRI Shapefile', filename= "result.shp")
            #print("Converting to Local Coordinates")
            #width=xmax-xmin
            height=ymax-ymin
            npoly=len(anno)
            
            
            
            # ADD CLASSES FOR THIS ANNO TO COCO CLASS, IF THEY ARE NOT REGISTERED THERE YET
            anno_classes=anno[category_field]          #get classes in current anno   (in relevant category column)
            anno_classes_uniq= list(set(anno_classes))            #get uniques
            for k in anno_classes_uniq:     
                if k not in category_log:            #wenn es die category noch ned gibt
                    new_category_id=len(coco_categories)
                    print("creating new category: ", k, " under id: ", new_category_id)   
                    
                    Class_descr={"supercategory":"LandCover","id":new_category_id,"name":k}  #mach neu (ids start at 0)
                    coco_categories.append(Class_descr) #hänge an
                    category_log.append(k) # registriere die neue klasse namentlich im log
                    category_id_log.append(new_category_id)   #registriere die neue id im log
            
            
            
            
            
            
            
            
            
            #For each poly in the clipped shape:
            for j in range(0,npoly):
    
                #UPDATE THE GEOMETRY
                oldpoly=anno.exterior.iloc[j]
                poly_x, poly_y=oldpoly.coords.xy
                newpoly=shapely.geometry.Polygon([[(x - xmin)/xres, (y - ymin)/yres] for x, y in zip(poly_x, poly_y)])
                if invert_y:
                    #print("Also inverting Y Axis")
                    #Die coordinaten werden auf tile coordinaten transformiert(subtraktion), mittels der auflösung werden die meter in pixelwerte transformiert, die y axe gespiegelt
                    newpoly=shapely.geometry.Polygon([[(x - xmin)/xres, (height-(y - ymin))/yres] for x, y in zip(poly_x, poly_y)])         
                anno.geometry.iloc[j]=newpoly
                
                
                
                #ADD THE ANNOTATIONS TO COCO ANNOTATIONS
                anno_id=("{:05d}".format(prefix))+"0987"+("{:05d}".format(len(anno_id_log)))#Unique id: erste 5 ziffern=prefix, 0987 bedeutet dass es anno ist, letzte 5 ziffern id
                image_id=im_id
                category_name=anno[category_field].values[j]  #find the category name
                category_id=category_id_log[category_log.index(category_name)] #find the corresponding id in this category name (this is where the logs come in)
                poly_x_n, poly_y_n=newpoly.exterior.coords.xy
                segmentation=[[round(val,3) for pair in zip(poly_x_n, poly_y_n) for val in pair]] #interleave x and y coordinates (von stackoverflow geklaut)
                bbox=newpoly.bounds
                area=newpoly.area
                iscrowd=0
                
                Anno_descr={"id":anno_id,"category_id":category_id,"iscrowd":iscrowd,"segmentation":segmentation,"image_id":image_id,"area":area,"bbox":bbox}
                coco_annotations.append(Anno_descr)
                anno_id_log.append(anno_id)  #also register the anno id in the log
    
        
        print("Finished processing image and annotation files")
    print("==============================================")
    
    print("Creating Coco-styled Database")
    coco_style_db={"info":coco_info,"licenses":coco_licenses,"images":coco_images,"annotations":coco_annotations,"categories":coco_categories}
    registers=[category_log,category_id_log,anno_id_log,im_id_log]
    
    return(coco_style_db,registers)




def grid_raster(src_grid,src_im,dst_im,prefix):
    grid = geopandas.read_file(src_grid)#Open The Grid shp
    data = rio.open(src_im)# Open the raster tif
    ntiles=len(grid)
    print("Starting to process image and annotation files")
    print("==============================================")
    for i in range(0,ntiles): #for every object in grid
        outfile_im=dst_im+prefix+"Tile"+("{:04d}".format(i))+".tif"
        outfile_ann=dst_ann+"Tile_%d_Annotation.shp " % i
        
        
        print("Clipping tile" + str(i) + " of "+str(ntiles))
        print("Target: "+outfile_im)
        coords=getFeatures(grid,i) #get the polygon
        Tile, out_transform = mask(dataset=data, shapes=coords, crop=True) #crop the raster to the polygon
        out_meta = data.meta.copy() #get a copy of the metadata of the raster
        #out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform,"crs": '+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'}) #update the meta for the cropped raster
        out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform}) #update the meta for the cropped raster
        #make the dir if necessary
        if not os.path.exists(dst_im):
            os.makedirs(dst_im)
        #write the image
        with rio.open(outfile_im, "w", **out_meta) as dest:
                dest.write(Tile) 
                                         
        print("Clipping annotation" + str(i)+ "of "+str(ntiles) )
        print("Target: "+outfile_ann)
        (xmin,ymin,xmax,ymax)=grid.bounds[i:i+1].iloc[0,:]
        subprocess.call("ogr2ogr -clipsrc %d %d %d %d "%(xmin,ymin,xmax,ymax) + outfile_ann+" "+src_ann )
    
    
    print("Finished processing image and annotation files")
    print("==============================================")
    return(0)




def build_mosaicvrt_from_dir(CurrentTileDir):
    #builds a vrt in a directory if it doesnt already exist
    import subprocess
    if not os.path.isfile(CurrentTileDir+"mosaictmp.vrt"):
        call="gdalbuildvrt "+CurrentTileDir+"mosaictmp.vrt"+" "+ CurrentTileDir+"*.jpg"
        print(call)
        subprocess.call(call,shell=True)
    else:
        print(CurrentTileDir+"mosaictmp.vrt already exists.")
    


def grid_raster_and_annotations(src_grid,src_im,src_ann,dst_im,prefix,dst_ann):
    grid = geopandas.read_file(src_grid)#Open The Grid shp
    data = rio.open(src_im)# Open the raster tif
    ntiles=len(grid)
    print("Starting to process image and annotation files")
    print("==============================================")
    for i in range(0,ntiles): #for every object in grid
        outfile_im=dst_im+prefix+"Tile"+("{:04d}".format(i))+".tif"
        outfile_ann=dst_ann+"Tile_%d_Annotation.shp " % i
        
        
        print("Clipping tile" + str(i) + " of "+str(ntiles))
        print("Target: "+outfile_im)
        coords=getFeatures(grid,i) #get the polygon
        Tile, out_transform = mask(dataset=data, shapes=coords, crop=True) #crop the raster to the polygon
        out_meta = data.meta.copy() #get a copy of the metadata of the raster
        #out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform,"crs": '+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'}) #update the meta for the cropped raster
        out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform}) #update the meta for the cropped raster
        #make the dir if necessary
        if not os.path.exists(dst_im):
            os.makedirs(dst_im)
        #write the image
        with rio.open(outfile_im, "w", **out_meta) as dest:
                dest.write(Tile) 
                                         
        print("Clipping annotation" + str(i)+ "of "+str(ntiles) )
        print("Target: "+outfile_ann)
        (xmin,ymin,xmax,ymax)=grid.bounds[i:i+1].iloc[0,:]
        subprocess.call("ogr2ogr -clipsrc %d %d %d %d "%(xmin,ymin,xmax,ymax) + outfile_ann+" "+src_ann )
    
    
    print("Finished processing image and annotation files")
    print("==============================================")
    return(0)
    
    

def prepare_vrt(src_filename,dst_filename,resample_factor=1,OverwriteEPSG=False):  
    #resamples a vrt and possibly defines projection at the same time
    raster = gdal.Open(src_filename)
    gt =raster.GetGeoTransform()
    outres_x = str(round(abs(gt[1]),5)*resample_factor)
    outres_y = str(round(abs(gt[5]),5)*resample_factor)
    cmd=("gdalwarp -tr "+outres_x+" "+outres_y+" -r bilinear "+src_filename+"  "+dst_filename)
    print(cmd)
    subprocess.call(cmd,shell=True)
    if OverwriteEPSG:
        ds_raster=gdal.Open(dst_filename,  gdal.GA_Update)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(OverwriteEPSG)
        crs=srs.ExportToWkt()
        newcrs=srs.ExportToProj4()
        print("to new CRS: "+newcrs)
        ds_raster.SetProjection(crs)
        ds_raster=None
        
def prepare_inputs(src_filename,dst_filename,target_resolution=None,resample_factor=None,reproject_EPSG=False, OverwriteInputEPSG=False):  
    #resamples an input image and possibly defines projection at the same time
    new_image_file = Path(dst_filename)
    if not new_image_file.exists():
        if target_resolution:
            outres_x=str(target_resolution)
            outres_y=str(target_resolution)
        elif resample_factor:
           raster = gdal.Open(src_filename)
           gt =raster.GetGeoTransform()
           outres_x = str((abs(gt[1]))*resample_factor)
           outres_y = str((abs(gt[5]))*resample_factor)
        else:
            print("Target Resolution or resample factor not given. Breaking off.")
            return(1)
        if OverwriteInputEPSG:
            s_raster=gdal.Open(src_filename,  gdal.GA_Update)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(OverwriteInputEPSG)
            crs=srs.ExportToWkt()
            newcrs=srs.ExportToProj4()
            print("Setting Projection of Input to CRS: "+newcrs)
            s_raster.SetProjection(crs)
            s_raster=None 
        if reproject_EPSG: #reproject if a target EPSG is given
            #get the input epsg (easiest with rasterio) not actually necessary for gdalwarp
            #with rasterio.open(src_filename) as src:
           #     print (src.crs)
            print("Reprojecting to: ", reproject_EPSG, " and resampling to resolution: ", outres_x ," ",outres_y)
            cmd=("gdalwarp -t_srs "+reproject_EPSG+" -tr "+outres_x+" "+outres_y+" -r bilinear "+'"'+src_filename+'"'+"  "+'"'+dst_filename+'"')
            print("\n",cmd,"\n")
            return(subprocess.check_output(cmd,shell=True))
            
        else:#else just resample the resolution
            print("Resampling to resolution:",outres_x ," ",outres_y)  
            cmd=("gdalwarp -tr "+outres_x+" "+outres_y+" -r bilinear "+'"'+src_filename+'"'+"  "+'"'+dst_filename+'"')
            print("\n",cmd,"\n")
            cmd=cmd
            #subprocess.Popen("gdalwarp", "-tr ", outres_x,outres_y, "-r bilinear",src_filename, dst_filename)
            return(subprocess.check_output(cmd,shell=True))
    else: 
        print("Destination image already exists, skipping.")
##Test
#prepare_inputs(src_filename,dst_filename,resample_factor=2)
#prepare_inputs(src_filename,dst_filename,target_resolution=1,reproject_EPSG="EPSG:32632")
               
               
        #subprocess.call("gdal_edit.bat -a_srs EPSG:"+ str(OverwriteEPSG) +" "+ dst_filename)
      
      
      
      
      
# #     crs="+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +datum=potsdam +units=m +no_defs " 
#      srs = osr.SpatialReference()
#      srs.ImportFromWkt(crs)
#      crs=ds_raster.GetProjection()
#      ds_raster.
#      ds_raster=None
# #     
# =============================================================================
#     
#     srs.ImportFromEPSG(27700)
#     
#     srs.ExportToProj4()
#     srs.ExportToWkt()
#     ds.SetProjection(srs.ExportToWkt())

    #ds_raster.bounds()
    #dst_srs = osr.SpatialReference()
    #dst_srs.ImportFromEPSG(4326)
    #test=gdal.Warp(dst_filename,src_filename, dstSRS= )
    #subprocess.check_output("gdalwarp -tr "+outres_x+" "+outres_y+" -r bilinear "+src_filename+"  "+dst_filename)
    #test=None


    # dst_srs.ExportToProj4()   
    #subprocess.call("gdalwarp "+src_filename+" "+dst_filename )
    #subprocess.call("gdalwarp "+src_filename+" "+dst_filename + " -t_srs EPSG:4979")


    #subprocess.call("gdal_edit.py -a_srs EPSG:"+str(31468) +" "+ dst_filename )




def Create_Grid(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth,OL_x,OL_y,SRefWKT,Overhang_tolerance_x=0,Overhang_tolerance_y=0,round_row_nr_digits=10,round_col_nr_digits=10):

    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)
    
    Projname=outputGridfn+".prj"
    Shapename=outputGridfn+".shp"
    
    # get rows
    rows = floor(round((((ymax-ymin)-OL_y)/(gridHeight-OL_y))+Overhang_tolerance_y,round_row_nr_digits))
    row_overhang=(((ymax-ymin)-OL_y)/(gridHeight-OL_y))
    print("Optimal number of rows = ", row_overhang)
    print("Actually creating ",rows," rows")
    # get columns
    col_overhang=(((xmax-xmin)-OL_x)/(gridWidth-OL_x))
    cols = floor(round((((xmax-xmin)-OL_x)/(gridWidth-OL_x))+Overhang_tolerance_x,round_col_nr_digits))
    print("Optimal number of cols = ", col_overhang)
    print("Actually creating ",cols," cols")
    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight
    #print("ringXleftOrigin =",ringXleftOrigin)
    #print("ringXrightOrigin =",ringXrightOrigin)
    #print("ringYtopOrigin =",ringYtopOrigin)
    #print("ringYbottomOrigin =",ringYbottomOrigin)
    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(Shapename):
        os.remove(Shapename)
    outDataSource = outDriver.CreateDataSource(Shapename)
    outLayer = outDataSource.CreateLayer(Shapename,geom_type=ogr.wkbPolygon )
    featureDefn = outLayer.GetLayerDefn()
    
   # spatialRef = osr.SpatialReference()
   # spatialRef.ImportFromEPSG(EPSG)         # from EPSG
   # spatialRef.MorphToESRI()
    file = open(Projname, 'w')
    file.write(SRefWKT)
    file.close()
    
    
    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature = None

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight+OL_y
            ringYbottom = ringYbottom - gridHeight+OL_y

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth-OL_x
        ringXrightOrigin = ringXrightOrigin + gridWidth-OL_x

    # Save and close DataSources
    outDataSource = None
    

    


def create_data_dirs(pathlist):
    for i in range(0, len(pathlist)):
        current_path=pathlist[i]
        if not os.path.exists(current_path):
            os.makedirs(current_path)



def getFeatures(gdf,index):
    #Function to parse features from GeoDataFrame via json, in such a format that rasterio wants them
    return [json.loads(gdf.to_json())['features'][index]['geometry']]
    
def Create_Grid_Over(InputRasterfn,outputGridfn,Imagesize_x,Imagesize_y,OL_x,OL_y,Overhang_tolerance_x,Overhang_tolerance_y):  
    data = gdal.Open(InputRasterfn, gdalconst.GA_ReadOnly) 
    geo_transform = data.GetGeoTransform()    
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    gridWidth=abs(Imagesize_x*geo_transform[1])
    gridHeight=abs(Imagesize_y*geo_transform[5])
    Overlap_x_geo=abs(OL_x*geo_transform[1])
    Overlap_y_geo=abs(OL_y*geo_transform[5])
    SRefWKT=data.GetProjection()
    print("Creating Grid")
    print("x_min = ",x_min)
    print("x_max = ",x_max)
    print("y_min = ",y_min)
    print("y_max = ",y_max)
    print("Tilesize = ",gridWidth," ", gridHeight,"meters")
    Create_Grid(outputGridfn,x_min,x_max,y_min,y_max,gridWidth,gridHeight,Overlap_x_geo,Overlap_y_geo,SRefWKT,Overhang_tolerance_x,Overhang_tolerance_y)





def create_gridded_tiles(gridpath,rasterpath,tilepath):
    grid = geopandas.read_file(gridpath)
    data = rio.open(rasterpath)
    for i in range(0,len(grid)):
        coords=getFeatures(grid,i)
        out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
        out_meta = data.meta.copy()
        out_meta.update({"driver": "GTiff","height": out_img.shape[1],"width": out_img.shape[2],"transform": out_transform,"crs": '+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'})
        if not os.path.exists(tilepath):
            os.makedirs(tilepath)
        with rio.open(tilepath+"Tile"+("{:04d}".format(i))+".tif", "w", **out_meta) as dest:
                dest.write(out_img)  
                


def read_tiles_into_array(TilePath,Channels,GetMaxArrays):
    Tiles=os.listdir(TilePath)
    NTiles=len(Tiles)
    NChannels=len(Channels)
    ##Get a sample to read the imagesize from
    Sample=rio.open(TilePath+Tiles[1])
    Imagesize_x=Sample.width
    Imagesize_y=Sample.height
    ##Initialize Arrays
    TilesArray=np.zeros((NTiles,NChannels,Imagesize_x,Imagesize_y))
    MaxArray=np.zeros(NTiles)
    MaxArray=MaxArray-1 ##Set to -1 as nodata value

    for i in range(0, NTiles):
        data=rio.open(TilePath+Tiles[i])
        Tile=data.read()
        TilesArray[i,:,:,:]=Tile[Channels,:,:]
    
    ###Also return an array with the most common value (if desired)
    if GetMaxArrays:
        unique, counts = np.unique(Tile, return_counts=True)
        dict(zip(unique, counts))
        MaxArray[i]=unique[np.argmax(counts)]
    return(TilesArray,MaxArray)
    
 
def grid_raster_into_tiled_array(gridpath,rasterpath,tilepath,arraypath,maxarraypath,imagesize_x,imagesize_y,Channels,GetMaxArray,prefix):
    grid = geopandas.read_file(gridpath)#Open The Grid shp
    data = rio.open(rasterpath)# Open the raster tif
    NTiles= len(grid)
    NChannels=len(Channels)
    ##Initialize Arrays
    TilesArray=np.zeros((NTiles,NChannels,imagesize_x,imagesize_y))
    MaxArray=np.zeros(NTiles)
    MaxArray=MaxArray-1 ##Set to -1 as nodata value
    
    for i in range(0,len(grid)): #for every object in grid
        coords=getFeatures(grid,i) #get the polygon
        Tile, out_transform = mask(dataset=data, shapes=coords, crop=True) #crop the raster to the polygon
        out_meta = data.meta.copy() #get a copy of the metadata of the raster
        out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform,"crs": '+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'}) #update the meta for the cropped raster
        #make the dir if necessary
        if not os.path.exists(tilepath):
            os.makedirs(tilepath)
        #write the image
        with rio.open(tilepath+prefix+"Tile"+("{:04d}".format(i))+".tif", "w", **out_meta) as dest:
                dest.write(Tile) 
        #write the tile array into its position in the array
        TilesArray[i,:,:,:]=Tile[Channels,:,:]
        if GetMaxArray:
                unique, counts = np.unique(Tile, return_counts=True)
                dict(zip(unique, counts))
                MaxArray[i]=unique[np.argmax(counts)]
    
    #save the files
    np.save(arraypath, TilesArray)
    if GetMaxArray:
        np.save(maxarraypath, MaxArray)
    return(TilesArray,MaxArray)
    
def dir_tif_png_copy(current_path):
   for root, dirs, files in os.walk(current_path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        #if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".png"):
                print ("A jpeg file already exists for %s" % name)
            # If a jpeg with the name does *NOT* exist, convert one from the tif.
            else:
                outputfile = os.path.splitext(os.path.join(root, name))[0] + ".png"
                try:
                    #im = Image.open(os.path.join(root, name))
                    im=rasterio.open(os.path.join(root, name)) #rasterio funktioniert immerhin
                    im=im.read() #lade das array raus
                    if im.shape[0]==4:
                        im=im[0:3,:,:] #wir nehmen nur die ersten 3 weil RGB
                        im=im.astype(np.uint8) #ist integer, also machen wir int
                        im=im.transpose(1, 2, 0) #für numpy muss die letzte dim die RGB dim sein
                        scipy.misc.imsave(outputfile, im) #rausschreiben
                        print ("Converting png for %s" % name)
                    if im.shape[0]==1: 
                        im=im.astype(np.uint8)
                        im=im.transpose(1, 2, 0) 
                        im=im.squeeze()
                        scipy.misc.imsave(outputfile, im)
                    
                    #im.thumbnail(im.size)
                    #im.save(outputfile, "JPEG", quality=100)
                    

                except Exception as e: 
                  print(e)
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax                  
                  
def inference_on_tif(src_grid,src_im,dst_im,dst_pred,inf_model,output_image_tile=False):
    grid = geopandas.read_file(src_grid)#Open The Grid shp
    img = rasterio.open(src_im)# Open the raster tif
    ntiles=len(grid)
    pred_polys=[] #collects all detected and polygonized features
    pred_classes=[]#collects their class ids
    pred_scores=[]#collects their scores
    pred_tiles=[]#collects the index of the tile they are predicted on
    #Loop over all the tiles
    for i in tqdm(range(0,ntiles), desc="Processing and Predicting Image Tiles" ): #for every object in grid
        outfile_im=dst_im+"Tile"+("{:04d}".format(i))+".tif"
        coords=SegUtils.getFeatures(grid,i)#get geometries of the tile    
        tile_bounds=grid.bounds.iloc[i][0:2].values#get the min x min y coords of the tile for adjusting the coordinates fo the predicted polygons later 
        tile_height=grid.bounds.iloc[i][3]-grid.bounds.iloc[i][1] #get the height in map units to later invert the y coordinates
        try:
            Tile, out_transform = mask(dataset=img, shapes=coords, crop=True)
        except ValueError as error:
            print(error)
            print("Skipping Tile "+str(i))
            continue
            
        if output_image_tile:
            out_meta = img.meta.copy()
            out_meta.update({"driver": "GTiff","height": Tile.shape[1],"width": Tile.shape[2],"transform": out_transform}) #update the meta for the cropped raster
            if not os.path.exists(dst_im):
                os.makedirs(dst_im)
            if os.path.isfile(outfile_im) :
                print("Image File exists already and will not be overwritten.")
            else:
                with rasterio.open(outfile_im, "w", **out_meta) as dest:
                    dest.write(Tile)         
        Tile=Tile.transpose(1, 2, 0)#We have to switch the dimensions around so that  the rgb dimension is the last one, as required by the network
        #inference
        pred_results = inf_model.detect([Tile], verbose=0)
        n_pred_obj=len(pred_results[0]["class_ids"]) #get the number of predicted objects to iterate over
        for j in range(0,n_pred_obj):#For every predicted object:
            new_poly=[]#list for all parts of that multipolygon that gets created for the predicted object
            results_mask=pred_results[0]["masks"][:,:,j] #get the mask
            results_mask=results_mask.astype('int16') #must not be boolean
        #visualize (not neessary)
            #plt.imshow(results_mask, interpolation='nearest')
            #plt.show()
        #polygonize and grab the raster value and the geometries
            results = ({'raster_val': v, 'geometry': s} 
                for i, (s, v)
                in enumerate(
                    shapes(results_mask)))      
            #the result can be converted to a list of npolys dicts of 
            #keys raster_val and geometry, geometry being a dict of
            #keys type and coordinates, coordinates being a list
            #this stuff now gets converted into shapely objects
            results = list(results)
            for result in results:
                if result['raster_val']:  #if the geometry is not the background
                    coordlist=result['geometry']['coordinates'][0]#get a list of tuples of 2
                    coordlist_adj = [(x,(tile_height-y))+tile_bounds for x,y in coordlist] #add the coordinates of the origin
                    new_part=shapely.geometry.Polygon(coordlist_adj) #make a new polygon out of it
                    new_poly.append(new_part) # add that as part to a list
            new_poly=shapely.geometry.MultiPolygon(new_poly) #after all parts for an object are collected, merge toa multipolygon (often there will just be one part)
            if(new_poly):   #if there was something detected
                pred_polys.append(new_poly) #collect all detected and polygonized features
                pred_classes.append(pred_results[0]["class_ids"][j]) #collect their class ids
                pred_scores.append(pred_results[0]["scores"][j])#collect their scores
                pred_tiles.append(i)
    
    
    df=pandas.DataFrame({"pred_class":pred_classes,"pred_score":pred_scores,"pred_tiles":pred_tiles})
    pred_gdf=geopandas.GeoDataFrame(df,geometry=pred_polys)
    pred_gdf.crs=grid.crs
    pred_gdf.to_file(dst_pred)
    return(pred_gdf)
    
                  
                  
                  
                  
                  

def inference_procedure(src_filename,dst_filename,dst_pred,inf_model,resample_factor,
                        OverwriteEPSG,ProcessedImagesPath,GridPath,
                        TilePath,Imagesize_x=512,Imagesize_y=512,Overlap_x=0,Overlap_y=0,
                        Overhang_x=0,Overhang_y=0):
    SegUtils.prepare_vrt(src_filename,dst_filename,resample_factor=resample_factor,OverwriteEPSG=OverwriteEPSG)
    SegUtils.Create_Grid_Over(ProcessedImagesPath+"IMX_resampled.tif", GridPath+"GridX",Imagesize_x,Imagesize_y,Overlap_x,Overlap_y,Overhang_x,Overhang_y)
    SegUtils.inference_on_tif(src_grid= GridPath+"GridX.shp",
            src_im= ProcessedImagesPath+"IMX_resampled.tif",
            dst_im= TilePath,
            inf_model=model,
            output_image_tile=False,
            dst_pred=dst_pred
            )

    os.remove(GridPath+"GridX.shp")
    os.remove(GridPath+"GridX.dbf")
    os.remove(GridPath+"GridX.prj")
    os.remove(GridPath+"GridX.shx")
    os.remove(ProcessedImagesPath+"IMX_resampled.tif")
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  