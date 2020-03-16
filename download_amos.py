# download_amos.py
# Austin Abrams, 2/16/10
# a helper utility to download and unzip a lot of images from the AMOS dataset.

import os
import sys
import requests
import io
import zipfile
import threading
import time
import monitor_parser as mp
# Change this to where you want data to be dumped off.  If not supplied, defaults to
# the current working directory.
# example:

# change these parameters as necessary to download whichever camera or year or month you
# want to download.

#MONTHS_TO_DOWNLOAD = range(1,13)
# if the script crashed or the power went out or something, this flag will
# skip downloading and unzipping a month's worth of images if there's already
# a folder where it should be.  If you set this to false, then downloads
# will overwrite any existing files in case of filename conflict.
SKIP_ALREADY_DOWNLOADED = True

# maximum number of threads allowed. This can be changed.
MAX_THREADS = 100

class DownloadThread(threading.Thread):
    camera_id = None
    year = None
    month = None

    def __init__(self, camera_id, year, month,provisional_path,final_path,patches_info_path,aqi_data,cam_ids):
        threading.Thread.__init__(self)

        self.camera_id = camera_id
        self.year = year
        self.month = month
        self.path_prov=provisional_path
        self.path_end=final_path
        self.path_patches=patches_info_path
        self.cam_ids=cam_ids
        self.aqi_data=aqi_data

    def run(self):
        location = self.path_prov

        if SKIP_ALREADY_DOWNLOADED and os.path.exists(location):
            print(location + " already downloaded.")
            #print("Getting patches from " + location)
            #put_images_to_aqi_folder(self.camera_id,self.aqi_data,self.cam_ids,self.path_prov,self.path_patches,self.path_end)
        
            return

        print("downloading to " + location)
        zf = download(self.camera_id, self.month, self.year)
        print("completed downloading to " + location)

        if not zf:
            print("skipping " + location)
            return
        ensure_directory_exists(location)

        print("Extracting from " + location)
        extract(zf, location)
        #put each image in its AQI folder

        print("Getting patches from " + location)
        put_images_to_aqi_folder(self.camera_id,self.aqi_data,self.cam_ids,self.path_prov,self.path_patches,self.path_end)
        print("Done")


def download(camera_id, month, year):
    """
    Downloads a zip file from AMOS, returns a file.
    """
    last_two_digits = camera_id % 100;
    last_four_digits = camera_id % 10000;
    
    if year < 2013 or year == 2013 and month < 9:
        ZIPFILE_URL = 'http://amosweb.cse.wustl.edu/2012zipfiles/'
    else :
        ZIPFILE_URL = 'http://amosweb.cse.wustl.edu/zipfiles/'
    url = ZIPFILE_URL + '%04d/%02d/%04d/%08d/%04d.%02d.zip' % (year, last_two_digits, last_four_digits, camera_id, year, month)
    print(url)
    #print '    downloading...',
    sys.stdout.flush()
    print(url)
    try:
        result = requests.get(url)
    except requests.exceptions.RequestException as e:
        print ('error.'+e.code)
        return None
        
    handle = io.BytesIO(result.content)
    
    #print 'done.'
    sys.stdout.flush()
    
    return handle
    
def extract(file_obj, location):
    """
    Extracts a bunch of images from a zip file.
    """
    #print '    extracting zip...',
    sys.stdout.flush()
    
    zf = zipfile.ZipFile(file_obj, 'r')
    zf.extractall(location)
    zf.close()
    file_obj.close()
    
    #print 'done.'
    sys.stdout.flush()
    
def ensure_directory_exists(path):
    """
    Makes a directory, if it doesn't already exist.
    """
    dir_path = path.rstrip('/')       
 
    if not os.path.exists(dir_path):
        parent_dir_path = os.path.dirname(dir_path)
        ensure_directory_exists(parent_dir_path)

        try:
            os.mkdir(dir_path)
        except OSError:
            pass

def put_images_to_aqi_folder(camera_id,aqi_data,cam_ids,path_prov,path_patches,path_end):
    patches_to_do=mp.list_from_patches_txt(path_patches+'/'+'{:0>8}'.format(str(camera_id))+'.txt')
    #print(patches_to_do)
    patches_per_image=len(patches_to_do)
    images_made=0
    for f in os.listdir(path_prov):#per cada foto a la carpeta
        if f.endswith('.jpg'):
            area_t=40
            pix_t=70
            if mp.check_night_image(path_prov+'/'+f,area_t,pix_t): #if  a night image
                continue
            else:#si no es de nit
                date_to_get=f.split('_')[0][:4]+'-'+f.split('_')[0][4:6]+'-'+f.split('_')[0][6:8]
                aqi_index=[r for r in aqi_data if [True for p in cam_ids if p==r[0]] and r[1] == date_to_get]
                if len(aqi_index)==0:#SI EL SITE QUE HA CALCULAT EL AQI NO ES CAP DELS QUE TE LA CAMARA PER ALGUN PARAMETRE AGAFEM UN DEL MATEIX CONDAT I JASTA      
                    aqi_index=[r for r in aqi_data if r[1] == date_to_get]
                if len(aqi_index)==0:
                    aqi_folder_index=420#per saber que ha fallat
                else:
                    aqi_folder_index=aqi_index[0][2]


                try:
                    print("Getting patch for image: " + str(f), end=',    ')
                    mp.make_image_patches(150,path_prov,f,path_end+'/'+str(aqi_folder_index),patches_to_do)
                    images_made+=1
                    #raise Exception
                except Exception as e:
                    print("Error while patching")
    print("Patched "  +str(images_made)+ " images in total")


def download_cam_year_month_to(camera_id,year,month,destiny_path):
    location = destiny_path
    if SKIP_ALREADY_DOWNLOADED and os.path.exists(location):
            print(location + " already downloaded.")
            return
    print("downloading to " + location)
    zf = download(camera_id, month, year)
    print("completed downloading to " + location)

    if not zf:
        print("skipping " + location)
        return
    ensure_directory_exists(location)

    print("Extracting from " + location)
    print(zf)
    extract(zf, location)
    print("Done")

def add_thread(camera_id, year, month, provisional_path,final_path,patches_info_path,aqi_data,param_ids):
    # for all cameras...

    thread_count = threading.activeCount()
    while thread_count > MAX_THREADS:
        print("Waiting for threads to finish...")
        time.sleep(1)
        thread_count = threading.activeCount()              
    download_thread = DownloadThread(camera_id=camera_id, year=year, month=month,provisional_path=provisional_path,final_path=final_path,patches_info_path=patches_info_path,aqi_data=aqi_data,cam_ids=param_ids)
    download_thread.start()

