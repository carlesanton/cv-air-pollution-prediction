import os
#import matlab.engine
import time
import scipy.io
import sys
import time
from datetime import date
import datetime
from random import sample
import fisher as fsh
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile


import sklearn.mixture
from pollution_parser import save_pollution_data_for_cam
from math import sqrt
from random import shuffle
from itertools import compress
from sklearn import svm, decomposition
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 

from yellowbrick.regressor import ResidualsPlot


import download_amos as da
import pollution_parser as pp

##google drive
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload


parameters_for_saving= ['Ozone','SO2','CO','NO2', 'PM2.5', 'PM10']
switcher_xlabel_plot = {
    0: "Ozone concentration (ppm mg/km)",
    1: "SO2 concentration (ppb)",
    2: "CO concentration (ppm mg/km)",
    3: "NO2 concentration (ppb)",
    4: "PM2.5 concentration (mg/m3)",
    5: "PM10 concentration (mg/m3)",
}

title_switcher = {
    0: "Ozone",
    1: "SO2",
    2: "CO",
    3: "NO2",
    4: "PM2.5",
    5: "PM10",
}

dont_get_data_till_parametter = 5

def download_months_year_from_cameras(month_vector,year,cameras_to_get,src_path,db_cursor,SAVE_POLL_DATA:bool):

    for c in cameras_to_get:
        for month in month_vector:
            par_ids=[c[i] for i in range(len(c)) if i!=0 ]
            #print(par_ids)
            t=time.time()
            print('Getting AQI data')
            aqi_query_condition="where site LIKE '"+ c[1].replace(c[1].split('-')[-1],'') + "%'" #que almenys estigui en el mateix condat
            all_aqi_site_data=pp.get_fields_from_table(db_cursor,'poll_aqi','site,data,sample',aqi_query_condition)
            delete_last_lines()
            print('Got all site AQI data in: ' + '{:4.2f}'.format((time.time()-t)))
            #print(aqi_index)

            final_data_path=src_path+'/final_data/image_data'+'{:0>8}'.format(c[0])
            provisional_data_path=src_path+'final_data/downloaded_images/'+'{:0>8}'.format(c[0])+'_'+str(year)+'{:0>2}'.format(str(month))
            patches_info_path=src_path+'final_data/camera_patches_info'
            t=time.time()
            print('Starting thread to download cam: ' + '{:0>8}'.format(c[0]) + ', month: ' +'{:0>2}'.format(month) + ', year: ' + str(year))
            da.add_thread(c[0], year, month, provisional_data_path,final_data_path,patches_info_path,all_aqi_site_data,par_ids)
            if SAVE_POLL_DATA:
                save_pollution_data_for_cam(db_cursor, c,src_path)
            delete_last_lines()
            print('Thread to download cam: ' + '{:0>8}'.format(c[0]) + ', month: ' +'{:0>2}'.format(month) + ', year: ' + str(year) + '     Started in: ' + '{:4.2f}'.format((time.time()-t)))
'''
def create_classification_data(path_with_class_folders):
    samples=[]
    classes=[]
    count=0
    for f in os.listdir(path_with_class_folders): 
        if (os.path.isdir(os.path.join(path_with_class_folders, f))) and (f!='matlab' or f!='420') and count<2:
            eng=matlab.engine.start_matlab()
            (s,c)=get_features_from_folder(eng,os.path.join(path_with_class_folders, f))
            print(str(len(s))+'	'+str(len(c)))
            samples.append(s)
            classes.append(c)
            count+=1
    return (samples,classes)
def get_features_from_folder(eng:matlab.engine,folder_path):
    folder_name=folder_path.split('/')[-1]
    samples=[]
    classes=[]
    #eng.cd('mcloud')
    for f in os.listdir(folder_path):
        if f.endswith('.jpg'):
            t=time.time()
            print("Getting features from: " + f, end='\r')
            data_path=folder_path+'/features'
            check_folder_and_create(data_path,0)
            raw_desc_feat=eng.first_test(folder_path,f,data_path,[12,16,20], 8, 300)
            print('Saved faetures into: '+ data_path + '      in ' + '{:4.2f}'.format((time.time()-t)) + ' seconds')
            final_fisher_vector=[]
            for raw in raw_desc_feat:
                final_fisher_vector.append(raw)
            samples.append(final_fisher_vector)
            classes.append(int(folder_name))


    return (samples,classes)

'''
def check_folder_and_create(path_to_folder, print_b=0):
    if not os.path.exists(path_to_folder):
        try:
            os.makedirs(path_to_folder)
            if print_b:
                print('Folder ' + path_to_folder + ' created')
        except OSError as e:
            #path=check_folder_and_create(path_to_folder+'_', print_b=0)
            print(e)
            #return path
    return path_to_folder
def delete_last_lines(n=1):
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def get_sample_names(clases_to_get,max_samples_per_class,folder,class_type,descriptors,patch_sizes,check_weather_data=False):
    final_samples=[]
    final_classes=[]
    final_paths=[]
    final_features=[]
    for f in os.listdir(folder):
        if [True for c in clases_to_get if f==str(c)]:#si la carpeta es una classe que s'hagi d'agafar

            if os.path.isfile(folder+'/'+f+'/image_names.txt'): #check if there is a file with the folder names exists
                myfile=open(folder+'/'+f+'/image_names.txt','r')
                file_lines = myfile.readlines()
                image_list = [ff.replace('\n','') for ff in file_lines if ff.replace('\n','').endswith('.jpg') and check_sample_has_all_pollution_data(ff.replace('\n',''),f,folder.replace('image_data','pollution_data'),class_type)]
                if check_weather_data:
                    image_list = [ff.replace('\n','') for ff in file_lines if ff.replace('\n','').endswith('.jpg') and check_sample_has_all_weather_data(ff.replace('\n',''),f,folder.replace('image_data','pollution_data'),class_type)]

            else:
                file_list=os.listdir(folder+'/'+f)
                image_list=[ff for ff in file_list if ff.endswith('.jpg') and check_sample_has_all_pollution_data(ff,f,folder.replace('image_data','pollution_data'),class_type)]
                if check_weather_data:
                    image_list=[ff for ff in file_list if ff.endswith('.jpg') and check_sample_has_all_weather_data(ff,f,folder.replace('image_data','pollution_data'),class_type)]
            
            if len(image_list)<max_samples_per_class:#si hi ha menys samples dels que volem agafar els agafem igual pero 
                samples_per_class=len(image_list)
            else:
                samples_per_class=max_samples_per_class
            final_name_list=(sample(image_list,samples_per_class))
            final_samples.extend(final_name_list)

            #GET PATHS
            for i in range(samples_per_class):
                final_path=folder.replace('image_data','pollution_data')+'/'+f
                final_paths.append(final_path)
                #print(final_path)

            #get features for each sample
            for image in final_name_list:
                feature_path=folder+'/'+f+'/features'
                features_of_sample=[]
                for descriptor in descriptors:
                    for patch in patch_sizes:
                        features_of_sample.append(retrieve_feature_from_sample(folder,image,final_path,descriptor,patch))
                final_features.append(features_of_sample)




            #GET CLASSES
            if class_type == "AQI":#just pt the aqi value
                final_classes.extend([int(f) for i in range(samples_per_class)])
            elif class_type == "pollution":#use as classes the 6 pollution samples for that image
                final_classes.extend([retrieve_pollution_data_from_sample(folder.replace('image_data','pollution_data')+'/'+f,final_name_list[i]) for i in range(samples_per_class)])
            print("Got " + '{: >4}'.format(samples_per_class) + " samples for AQI index " + '{: >3}'.format(f) + " of cam " + folder.split('/')[-1])
            [print(np.asarray(final_classes)[i,:]) for i in range(np.asarray(final_classes).shape[0]) if sum(np.asarray(final_classes)[i,:]>150)]
    return (final_samples, final_classes, final_paths, final_features)
def get_sample_names_meateorological(clases_to_get,samples_per_class,folder,camera_shuffle,cameras_to_get):
    final_samples=[]
    final_classes=[]
    for aqi_index in clases_to_get:
        text_list=[]
        sample_path=[]
        for cam_folder in cameras_to_get:#per cada camara
            if os.path.isdir(os.path.join(folder, '{:0>8}'.format(cam_folder))):
                for aqi_folder in os.listdir(folder+'/'+'{:0>8}'.format(cam_folder)):#per cada caprpeta de index
                    if  os.path.isdir(os.path.join(folder, '{:0>8}'.format(cam_folder),aqi_folder)) and aqi_folder==str(aqi_index):
                        #print(aqi_folder + ': ' + str(aqi_index))
                        file_list=os.listdir(folder+'/'+'{:0>8}'.format(cam_folder)+'/'+aqi_folder)
                        #print(folder+'/'+'{:0>8}'.format(cam_folder)+'/'+aqi_folder)

                        [text_list.append((ff,os.path.join(folder, str('{:0>8}'.format(cam_folder)),str(aqi_index)))) for ff in file_list if ff.endswith('.txt') and check_sample_has_all_pollution_data(ff,aqi_folder,os.path.join(folder, '{:0>8}'.format(cam_folder)))]
                        
                        #print(len(text_list))
                        #print(text_list)
                        '''
                        
                        try:
                            print(text_list[0])
                        except:
                            print('len=======0')
                        '''
                print('Got data for index: '+ str(aqi_index)+' at cam: ' + '{:0>8}'.format(cam_folder))

        shuffled_name_list=text_list
        print('final aqi list name lenght: '+str(len(shuffled_name_list)))

        if camera_shuffle:
            shuffled_name_list=(sample(text_list,len(text_list)))#to get samples from every camera
            print(len(shuffled_name_list))
        
        final_name_list=(sample(shuffled_name_list,samples_per_class))
        final_samples.extend(final_name_list)
        final_classes.extend([int(aqi_index) for i in range(samples_per_class)])
        final_sample_names=[f[0] for f in final_samples]
        final_sample_paths=[f[1] for f in final_samples]
    return (final_sample_names, final_classes, final_sample_paths)
def check_sample_has_all_pollution_data(patch_name,class_folder,folder_path,class_type):
    expected_name=list(patch_name.replace(str('_'+patch_name.split('_')[-1]),''))
    expected_name[-4:]=list("0000")

    final_path=folder_path+'/'+str(class_folder)+'/'+"".join(expected_name)+'_pollution.txt'
    myfile=open(final_path,"r")
    lines = myfile.read().split('[')
    count=0
    ccc=0

    if not os.path.isfile(os.path.join(folder_path.replace('pollution_data','image_data'),class_folder,'features', str(patch_name.replace('.jpg','')+ '_12_centrist.mat' ) )):
        print(os.path.join(folder_path,class_folder,'features',patch_name))
        return False
    elif check_sample_has_all_weather_data(patch_name,class_folder,folder_path,class_type):
        for f in lines:
            if not f:
                continue
            else:
                if class_type == "AQI":
                    if count>dont_get_data_till_parametter:
                        ff=f.replace(']','')
                        fff=ff.split(',')
                        if len(fff)>1:
                            if count==6:
                                for ffff in fff:
                                    try:
                                        suport=float(ffff)
                                        ccc+=1
                                    except:
                                        return False
                            else:
                                ccc+=1

                        else:
                            try:
                                suport=float(fff[0])
                                ccc+=1
                            except:
                                return False
                        #print(len(f))
                        #print(ff)
                        #[print(char) for char in lines]
                    count+=1
                elif class_type=="pollution":
                    if count<=dont_get_data_till_parametter:
                        ff=f.replace(']','')
                        fff=ff.split(',')
                        if len(fff)>1:
                            if count==6:
                                for ffff in fff:
                                    try:
                                        suport=float(ffff)
                                        ccc+=1
                                    except:
                                        return False
                            else:
                                ccc+=1

                        else:
                            try:
                                suport=float(fff[0])                                
                                ccc+=1
                            except:
                                return False
                        #print(len(f))
                        #print(ff)
                        #[print(char) for char in lines]
                    count+=1
        if ccc<5:
            return False
        else:
            return True
def check_sample_has_all_weather_data(patch_name,class_folder,folder_path,class_type):
    expected_name=list(patch_name.replace(str('_'+patch_name.split('_')[-1]),''))
    expected_name[-4:]=list("0000")

    final_path=folder_path+'/'+str(class_folder)+'/'+"".join(expected_name)+'_pollution.txt'
    myfile=open(final_path,"r")
    lines = myfile.read().split('[')
    count=0
    ccc=0
    if not os.path.isfile(os.path.join(folder_path.replace('pollution_data','image_data'),class_folder,'features', str(patch_name.replace('.jpg','')+ '_12_centrist.mat' ) )):
        print(os.path.join(folder_path,class_folder,'features',patch_name))
        return False
    else:
        for f in lines:
            if not f:
                continue
            else:
                if class_type == "AQI":
                    if count>dont_get_data_till_parametter:
                        ff=f.replace(']','')
                        fff=ff.split(',')
                        if len(fff)>1:
                            if count==6:
                                for ffff in fff:
                                    try:
                                        suport=float(ffff)
                                        ccc+=1
                                    except:
                                        return False
                            else:
                                ccc+=1

                        else:
                            try:
                                suport=float(fff[0])
                                ccc+=1
                            except:
                                return False
                        #print(len(f))
                        #print(ff)
                        #[print(char) for char in lines]
                    count+=1
                elif class_type=="pollution":
                    if count>dont_get_data_till_parametter:
                        ff=f.replace(']','')
                        fff=ff.split(',')
                        if len(fff)>1:
                            if count==6:
                                for ffff in fff:
                                    try:
                                        suport=float(ffff)
                                        ccc+=1
                                    except:
                                        return False
                            else:
                                ccc+=1

                        else:
                            try:
                                suport=float(fff[0])
                                ccc+=1
                            except:
                                return False
                        #print(len(f))
                        #print(ff)
                        #[print(char) for char in lines]
                    count+=1
        if ccc<5:
            return False
        else:
            return True

def generate_gaussian_mixture_model(samples,number_of_gaussians):
    
    fv_gmm = sklearn.mixture.GaussianMixture(n_components=number_of_gaussians, covariance_type='diag')
    fv_gmm.fit(samples)

    #fv_gmm = fsh.generate_gmm_v2(samples,number_of_gaussians,responses)
    #[print(f.shape) for f in fv_gmm]
    #print(fv_gmm)
    return fv_gmm
def fisher_vector_encoding(pollution_data_paths,sample_name,sample_features,descriptors,patches,gaussian_models,multi_modal_weights,norm_type):
    count=0
    weigth_selector=0
    sample_fisher_vector=[]
    retrieve_count=0
    scale=1

    for descriptor_name in descriptors:
        for patch_size in patches:
            data_array=sample_features[count]
            #retrieve_feature_from_sample(sample_name,pollution_data_paths,descriptor_name,patch_size)
            #DATA NORMALIZATION
            fv=fisher_vector(np.asarray(data_array), gaussian_models[count])

            #square-rooting normalization
                #en teoria la funcio fisher_vector ja la fa
            #l2 normalization

            #fisher vector append
            '''
            print("Fisher vector " + str(count) + ' has lenght: ' + str(len(fv)))
            print("Fisher vector " + str(count) + ' has shape: ' + str((fv.shape)))
            '''
            sample_fisher_vector.append(fv)

            count+=1
    
    visual_fv_length=np.sum(np.concatenate([f.shape for f in sample_fisher_vector]))
    visual_fisher_vector=np.concatenate([(multi_modal_weights[weigth_selector]/visual_fv_length)*f for f in sample_fisher_vector])
    weigth_selector+=1


    poll_data_from_sample=retrieve_weather_data_from_sample(pollution_data_paths,sample_name)
    poll_data_length=len(poll_data_from_sample)
    met_fisher_vector=(multi_modal_weights[weigth_selector]/poll_data_length)*fisher_vector(np.asarray(poll_data_from_sample), gaussian_models[count])
    final_fisher_vector=np.concatenate([visual_fisher_vector,met_fisher_vector])
    #get data from txt
    #fv = fsh.fisher_vector(met_data, gaussian_models[count][0], gaussian_models[count][1], gaussian_models[count][2])
    #square-rooting normalization
        #en teoria la funcio fisher_vector ja la fa
    #l2 normalization

    return np.transpose(final_fisher_vector)
def fisher_vector(xx, gmm: sklearn.mixture.GaussianMixture):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    #print(d_sigma.shape)
    '''
    print('xx shape ' + str(xx.shape))
    print('d_sigma shape ' + str(d_sigma.shape))
    print('d_sigmaflat shape ' + str(d_sigma.flatten().shape))
    '''
    # Merge derivatives into a vector.
    return np.hstack(( d_mu.flatten(), d_sigma.flatten()))#np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def retrieve_feature_from_sample(folder_path,sample_name,pollution_data_folder,descriptor_name,patch_size):
    sample_aqi_value=pollution_data_folder.split('/')[-1]
    #returns array with dimensions (NxD) where N is the number of feature points extracted with that patch size and D the dimensionality of the descriptor
    #print("retrieve_feature_from_sample MUST BE TESTED")
    feature=[]
    #print(sample_name)

    expected_name=sample_name.replace('.jpg','')+'_'+str(patch_size)+'_'+descriptor_name+'.mat'
    expected_path=folder_path+'/'+str(sample_aqi_value)+'/features/'+expected_name
    try:
        array=scipy.io.loadmat(folder_path+'/'+str(sample_aqi_value)+'/features/'+expected_name)
        fea_arr=array['feaSet']['feaArr'][0][0]
        #print(fea_arr.shape)
        feature.append(fea_arr)
    except:
        print("Couldn't load feature " + expected_name +" from sample " + sample_name + " (AQI folder " + str(sample_aqi_value) + ")")

    return feature[0]

def retrieve_feature_from_sample_(folder_path,sample_name,pollution_data_folder,descriptor_name,patch_size):
    sample_aqi_value=pollution_data_folder.split('/')[-1]
    #returns array with dimensions (NxD) where N is the number of feature points extracted with that patch size and D the dimensionality of the descriptor
    #print("retrieve_feature_from_sample MUST BE TESTED")
    feature=[]
    #print(sample_name)

    expected_name=sample_name.replace('.jpg','')+'_'+str(patch_size)+'_'+descriptor_name+'.mat'
    expected_path=folder_path+'/'+str(sample_aqi_value)+'/features/'+expected_name
    try:
        array=scipy.io.loadmat(folder_path+'/'+str(sample_aqi_value)+'/features/'+expected_name)
        fea_arr=array['feaSet']['feaArr'][0][0]
        #print(fea_arr.shape)
        feature.append(fea_arr)
    except:
        print("Couldn't load feature " + expected_name +" from sample " + sample_name + " (AQI folder " + str(sample_aqi_value) + ")")

    return feature[0]

def parse_pollution_txt(final_path):
    myfile=open(final_path,"r")
    lines = myfile.read().split('[')
    measure_values=[]
    count=0
    for f in lines:
        if not f:
            continue
        else:
            if count<=dont_get_data_till_parametter:
                ff=f.replace(']','')
                fff=ff.split(',')
                if len(fff)>1:
                    if count==6:
                        for ffff in fff:
                            try:
                                measure_values.append(float(ffff))
                            except:
                                raise 
                                # 
                                print("NO HI HA VALOR")
                                measure_values.append((float(420)))                                
                                #return np.zeros(6)
                    else:
                        measure_values.append(float(fff[0])) 

                else:
                    try:
                        measure_values.append((float(fff[0])))
                    except:
                        raise
                        print("NO HI HA VALOR")
                        measure_values.append((float(420)))
                        
                        # return np.zeros(6)
                        #measure_values.append((float(fff[0])))
                #print(len(f))
                #print(ff)
                #[print(char) for char in lines]
            count+=1
    '''
    print(str(len(measure_values)) + ':     ',end='')
    [print(str(s) + ', ',end='') for s in measure_values]
    print('')
    '''
    #print(np.asarray(measure_values).shape)
    if sum(np.asarray(measure_values)>150):
        print(np.asarray(measure_values))

    return(np.asarray(measure_values))
def retrieve_pollution_data_from_sample(folder_path,sample_name):
    support_list=list(sample_name.replace(str('_'+sample_name.split('_')[-1]),''))
    support_list[-4:]=list("0000")
    expected_name="".join(support_list)+'_pollution.txt'
    final_path=folder_path+'/'+expected_name
    return parse_pollution_txt(final_path)

def retrieve_weather_data_from_sample(folder_path,sample_name):
    support_list=list(sample_name.replace(str('_'+sample_name.split('_')[-1]),''))
    support_list[-4:]=list("0000")
    expected_name="".join(support_list)+'_pollution.txt'
    final_path=folder_path+'/'+expected_name
    myfile=open(final_path,"r")
    lines = myfile.read().split('[')
    measure_values=[]
    count=0
    for f in lines:
        if not f:
            continue
        else:
            if count>dont_get_data_till_parametter:
                ff=f.replace(']','')
                fff=ff.split(',')
                if len(fff)>1:
                    if count==6:
                        for ffff in fff:
                            try:
                                measure_values.append(float(ffff))
                            except:
                                print("NO HI HA VALOR")
                                print(ffff)
                    else:
                        measure_values.append(float(fff[0])) 

                else:
                    try:
                        measure_values.append((float(fff[0])))
                    except:
                        print("NO HI HA VALOR")
                        print(fff)
                        #measure_values.append((float(fff[0])))
                #print(len(f))
                #print(ff)
                #[print(char) for char in lines]
            count+=1
    '''
    print(str(len(measure_values)) + ':     ',end='')
    [print(str(s) + ', ',end='') for s in measure_values]
    print('')
    '''
    #print(np.asarray(measure_values).shape)
    return(np.asarray(measure_values))

def count_poll_data(folder):
    count=0
    f_count=0
    count_for_index=[]
    for f in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, f)):
            f_count=0
            for ff in os.listdir((os.path.join(folder, f))):
                    if ff.endswith('.txt'):
                            count+=1
                            f_count+=1
            #print(f + ': ' + str(f_count))
            count_for_index.append((f,f_count))

    return(count_for_index)
def count_samples_for_aqi_index(root_folder):
    count_list=[]
    for f in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, f)):
            num_for_folder = count_poll_data(root_folder+'/'+f)
            count_list.append((f,num_for_folder))

    max_=250
    aqi_index_vector=np.zeros((max_,2),dtype=int)
    aqi_index_vector[:,0]=range(max_)
    [print(pair) for pair in aqi_index_vector]
    for ccc in count_list:
        for index_pair in ccc[1]:
            print(index_pair[0])
            aqi_index_vector[int(index_pair[0]),1]+=index_pair[1]

    [print('{:>3}'.format(pair[0]) + ': ' +'{:>5}'.format(pair[1])) for pair in aqi_index_vector if pair[1]!=0]
     
def save_pollution_data_for_cams(db_cursor, cameras_to_get,src_path):
    for c in cameras_to_get:
        t=time.time()
        print("Saving pollution data for cam: " + str(c[0]))
        pollution_parser.save_pollution_data_for_cam(db_cursor,c,src_path+'final_data/pollution_data'+'/'+'{:0>8}'.format(c[0]))
        print('Pollution data saved in ' + '{:4.2f}'.format((time.time()-t)))

###COMPUTE METRICS
def compute_mean(predicted):
    number_samples=float(len(predicted))
    summ=float(np.sum(np.asarray(predicted)))
    return summ/number_samples 
def compute_error_stdv(expected_vector, predicted_vector, mean_squared_error):
    errors = abs((expected_vector)-(predicted_vector))
    error_mean_error_squared_difference = (errors - mean_squared_error)**2
    #print(error_mean_error_squared_difference)
    stdv=sqrt(sum(error_mean_error_squared_difference)/len(predicted_vector))
    return stdv
def compute_stdv(expected_vector, sample_mean):
    errors = abs((expected_vector)-(sample_mean))**2
    #print(error_mean_error_squared_difference)
    stdv=sqrt(sum(errors)/len(expected_vector))
    return stdv
def compute_s_ress(predicted_vector,expected_vector):
    diff_squares = (expected_vector - predicted_vector)**2
    residual = np.sum((diff_squares))
    return np.asarray(residual)
def compute_s_tot(expected_vector):
    diff_squares = (expected_vector - compute_mean(expected_vector))**2
    [print(expected_vector[i]) for i in range(len(diff_squares)) if diff_squares[i]<1]
    residual = np.sum((diff_squares))
    return np.asarray(residual)

def normalize_data(data,norm_type = 'norm'):
    #NORMALIZATION
    if norm_type == 'norm':
                scaler = Normalizer().fit(data)
                rescaled_samples = scaler.transform(data)
    elif norm_type=='scale':
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_samples = scaler.fit_transform(data)
    else:
        rescaled_samples=data

    return rescaled_samples


############PERFORMING FUNCTIONS

def get_samples(clases_to_get,samples_per_class,folder,class_type,descriptors,patch_sizes):
    (sample_names,sample_classes,pollution_sample_path,sample_features)=get_sample_names(clases_to_get,samples_per_class,folder,class_type,descriptors,patch_sizes)
    sample_index=list(range(len(sample_names)))
    shuffle(sample_index)
    shuffled_sample_names=[]
    shuffled_sample_classes=[]
    shuffled_pollution_sample_paths=[]
    shuffled_sample_features=[]
    for i in sample_index:
        shuffled_sample_names.append(sample_names[i])
        shuffled_sample_classes.append(sample_classes[i])
        shuffled_pollution_sample_paths.append(pollution_sample_path[i])
        shuffled_sample_features.append(sample_features[i])
    print(len(shuffled_sample_features))
    print(len(shuffled_sample_features[0]))
    print(len(shuffled_sample_features[0][0][0]))
    return (shuffled_sample_names,shuffled_sample_classes,shuffled_pollution_sample_paths,shuffled_sample_features)

def get_samples_by_pollution(clases_to_get,samples_per_class,root_folder,cameras,class_type,just_data_with_image):
    pollution_folder=root_folder+'/'+'pollution_data'
    image_folder=root_folder+'/'+'image_data'
    #retrieve_pollution_data_from_sample(folder_path,sample_name)
    pollution_data=[]
    path_list=[]
    if just_data_with_image:
        for c in cameras:#for each camera we want to take the data
            for aqi_f in os.listdir(image_folder+'/'+'{:0>8}'.format(c)):#for each aqi folder
                if os.path.isdir(image_folder+'/'+'{:0>8}'.format(c)+'/'+aqi_f):
                    #print(image_folder+'/'+'{:0>8}'.format(c)+'/'+aqi_f)
                    prov_pollution_data=[parse_pollution_txt(image_folder+'/'+'{:0>8}'.format(c)+'/'+aqi_f+'/'+f) for f in os.listdir(image_folder+'/'+'{:0>8}'.format(c)+'/'+aqi_f) if f.endswith('_pollution.txt') and f!='.DS_pollution.txt']
                    prov_path_list=[image_folder+'/'+'{:0>8}'.format(c)+'/'+aqi_f+'/'+f.replace('_pollution.txt','') for f in os.listdir(image_folder+'/'+'{:0>8}'.format(c)+'/'+aqi_f) if f.endswith('_pollution.txt') and f!='.DS_pollution.txt']
                    index_to_delete=(np.arange(len(prov_path_list))[np.sum(np.asarray(prov_pollution_data),1) == 0.0])

                    for i in sorted(index_to_delete, reverse=True):
                        del prov_path_list[i]
                        del prov_pollution_data[i]
                    path_list = path_list + prov_path_list
                    pollution_data = pollution_data + prov_pollution_data
                    #pollution_data=np.concatenate((np.delete(prov_pollution_data,pollution_data)), 1)
                    
                    '''
                    print(np.asarray(pollution_data))
                    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print(len(path_list))
                    print(len(prov_name_list))
                    print(len(prov_pollution_data))
                    print(pollution_data.shape)
                    '''
                    print(np.asarray(pollution_data).shape)

    min_predict_range=np.amin(np.asarray(pollution_data),0)
    max_predict_range=np.amax(np.asarray(pollution_data),0)

    print("Minimums = " + str(min_predict_range))
    print("Maximums = " + str(max_predict_range))
    print("Maximums = " + str(max_predict_range))
    num_steps=50
    steps=[(max_predict_range[i] -min_predict_range[i])/num_steps for i in range(min_predict_range.shape[0])]
    count=0
    all_hists=np.zeros((num_steps,len(steps)))
    all_indexes=[]
    print(all_hists.shape)
    for step in steps:
        par_hist=np.zeros(num_steps)
        hist_indexs=[[] for f in range(num_steps)]
        poll_sample_index=0
        for poll_sample in pollution_data:
            sample_bin=int(num_steps*((poll_sample[count]-min_predict_range[count])/(max_predict_range[count]-min_predict_range[count])-0.0001))
            par_hist[sample_bin]+=1
            hist_indexs[sample_bin].append(poll_sample_index)
            poll_sample_index+=1

        all_hists[:,count]=par_hist
        all_indexes.append(hist_indexs)
        print(len(all_indexes))
        count+=1

    
    [print(str(np.amin(all_hists[:,a])) + ' ' + str(np.amax(all_hists[:,a]))) for a in range(len(steps))]
    print(all_hists.shape)
    sample_limit=np.asarray(pollution_data).shape[0]/num_steps
    for bin_indexes in all_indexes:
        for binn in bin_indexes:
            indexes=range(len(binn))
            if len(binn)>sample_limit:
                shuffle(indexes)

        '''
        if len(bin_indexes) > sample_limit:
            indexes=range(len(bin_indexes))
            shuffle(indexes)
            final_index_list.append(indexes[0:int(sample_limit)])
        '''
        '''
        plt.figure()
        plt.plot(all_hists[:,a],label='All samples',color='darkgreen')
        if a==len(steps)-1:
            plt.show()
        else:
            plt.draw()
        '''

    print()
'''
    (sample_names,sample_classes,pollution_sample_path)=get_sample_names(clases_to_get,samples_per_class,folder,class_type)
    sample_index=list(range(len(sample_names)))
    shuffle(sample_index)
    shuffled_sample_names=[]
    shuffled_sample_classes=[]
    shuffled_pollution_sample_paths=[]
    for i in sample_index:
        shuffled_sample_names.append(sample_names[i])
        shuffled_sample_classes.append(sample_classes[i])
        shuffled_pollution_sample_paths.append(pollution_sample_path[i])

    return (shuffled_sample_names,shuffled_sample_classes,shuffled_pollution_sample_paths)
'''


def cross_validation(sample_names,sample_classes,sample_pollution_paths,sample_features,image_folder,folds,multi_modal_weights,gaussian_mixture_keep_portion,number_of_gaussians,descriptors,patch_sizes,class_type,samples_per_aqi_index,regresors,save_results_boolean,histograms_boolean,save_to_google_drive_boolean,gmm_booleans,norm_type='norm',PCA=0):
    fold_indexes=[]
    accuracies=[]
    models_for_each_round=[]
    result_list=[]

    gaussian_mixtures_folder = image_folder.replace('image_data/00000398','gmm')



    samples_per_fold=int((len(sample_classes))/folds)
    [[fold_indexes.append(n) for i in range(samples_per_fold)] for n in range(folds)]
    fold_index_array=np.asarray(fold_indexes)
    results_list_for_each_regresor=[[] for f in regresors]
    fited_regressors=[[] for f in regresors]
    
    [print(np.asarray(sample_classes)[i,:]) for i in range(np.asarray(sample_classes).shape[0]) if sum(np.asarray(sample_classes)[i,:])>150]
    for k in range(folds):
        train_index=np.arange(len(fold_indexes))[fold_index_array != k]
        test_index=np.arange(len(fold_indexes))[fold_index_array == k]

        train_labels=[sample_classes[i] for i in train_index]
        test_labels=[sample_classes[i] for i in test_index]
        print("Train set size: " + str(len(train_index)))
        print("Test  set size: " + str(len(test_index)))
        s_tot=[compute_s_tot(np.asarray(test_labels)[:,i]) for i in range(np.asarray(test_labels).shape[1])]
        print(s_tot)
        train_histograms=data_historgrams(np.asarray(train_labels))
        test_histograms=data_historgrams(np.asarray(test_labels))
        plot_histograms(train_histograms,test_histograms, k+1,folds, histograms_boolean)
        '''
        gmm_number_selector=0#to select the number of GMM for the data (met data uses less)
        gaussian_mixture_models=[]
        gaussian_models_time=time.time()
        print("Getting Gaussian Mixture Models to encode samples with " + str(number_of_gaussians[gmm_number_selector]) + " components")
        print(descriptors)
        retrieve_count=0
        for descriptor in descriptors:
            print(descriptor)
            for patch_size in patch_sizes:
                
                #Get samples
                t=time.time()
                print("     Retrieving features from all images for descriptor " + descriptor + " patch size: " + str(patch_size))
                samples_from_descriptor=[]
                [samples_from_descriptor.append(sample_features[i][retrieve_count]) for i in train_index]
                count=0
                final_list=[]
                for sample_array in samples_from_descriptor:
                    for row in sample_array:
                        if np.random.uniform(low=0.0, high=1.0)<gaussian_mixture_keep_portion:
                            final_list.append(row)
                        #print(len(final_list))
                    count+=1
                all_features_array=np.asarray(final_list,dtype=np.float32)
                print("     Retrieved features for descriptor " + descriptor + " patch size: " + str(patch_size) + ' in: ' + '{:4.2f}'.format((time.time()-t)))    
                
                #get GMM adn save if wanted
                gaussian_mixture_models.append(retrieve_or_compute_gmm(gaussian_mixtures_folder,descriptor+str(patch_size),number_of_gaussians[gmm_number_selector],norm_type,gmm_booleans,all_features_array,k,save_to_google_drive_boolean))
               

                retrieve_count+=1

        #GMM METEOROLOGICAL
        gmm_number_selector+=1
        t=time.time()
        print("     Retrieving meteorological data for all samples")
        pollution_train_samples=np.vstack([retrieve_weather_data_from_sample(sample_pollution_paths[i],sample_names[i]) for i in train_index])
          
        
        print("     Getting gaussian mixture for meteorological data with " + str(number_of_gaussians[gmm_number_selector]) + " components")
        gaussian_mixture_models.append(retrieve_or_compute_gmm(gaussian_mixtures_folder,'weather',number_of_gaussians[gmm_number_selector],norm_type,gmm_booleans,pollution_train_samples,k,save_to_google_drive_boolean))
        delete_last_lines()
        print('All gaussian mixture models with: ' + str(number_of_gaussians[gmm_number_selector])+ ' components got in ' + '{:4.2f}'.format((time.time()-gaussian_models_time)))
        print("     Got all meteorological data in: " + '{:4.2f}'.format((time.time()-t)))  

        #generar samples
        t=time.time()
        print('Encoding samples')

        encoded_train_samples=np.vstack([fisher_vector_encoding(sample_pollution_paths[i],sample_names[i],sample_features[i],descriptors,patch_sizes,gaussian_mixture_models,multi_modal_weights,norm_type) for i in train_index]) 
        #encoded_train_samples hauria de tenir tamany Nsamples x Nfeatures
        encoded_test_samples=np.vstack([fisher_vector_encoding(sample_pollution_paths[i],sample_names[i],sample_features[i],descriptors,patch_sizes,gaussian_mixture_models,multi_modal_weights,norm_type) for i in test_index]) 
        delete_last_lines()
        print('Samples encoded in ' + '{:4.2f}'.format((time.time()-t)))
        
        #PERFORM PCA AND FIT DATA TO REGRESSOR, RETURNING AS OUTPUT THE METRICS COMPUTED
        if bool(PCA):
            print("ENCODED SAMPLES DIMENSION = " + str(len(encoded_train_samples[0])))
            
            if PCA<1:
                PCA=np.round(len(encoded_train_samples[1])/PCA)

            pca = decomposition.PCA(n_components=int(PCA))
            pca.fit(encoded_train_samples)
            # final_train_samples = pca.transform(encoded_train_samples)
            # final_test_samples = pca.transform(encoded_test_samples)
            print("SAMPLES DIMENSION AFTER PCA = " + str(len(pca.transform(encoded_train_samples)[0])))

            #save PCA model
            model_folder=histograms_boolean[1]+'/PCA'
            check_folder_and_create(model_folder)
            print("Saving PCA model")
            filename = 'PCA_basis_round'+str(k+1)+'.sav'
            pickle.dump(pca, open(model_folder+'/'+filename, 'wb')) 

            count=0
            for reg in regresors:
                try:
                    (round_regresor_results,fited_regressor)=fit_regressor_and_compute_metrics(reg,pca.transform(encoded_train_samples),train_labels,pca.transform(encoded_test_samples),test_labels,PCA,class_type,histograms_boolean,k,folds)
                    fited_regressors[count].append(fited_regressor)#al final tindra tamany RxK amb cada element [count] tindrà un regressor per cada ronda
                    results_list_for_each_regresor[count].append(round_regresor_results)
                except:
                    print("Error fiting data")
                    results_list_for_each_regresor[count].append(results_list_for_each_regresor[count][-1])
                count+=1

            models_for_each_round.append(gaussian_mixture_models)


        else:
            # final_test_samples=encoded_test_samples
            # final_train_samples=encoded_train_samples

            count=0
            for reg in regresors:
                (round_regresor_results,fited_regressor)=fit_regressor_and_compute_metrics(reg,encoded_train_samples,train_labels,encoded_test_samples,test_labels,PCA,class_type,histograms_boolean,k,folds)
                fited_regressors[count].append(fited_regressor)#al final tindra tamany RxK amb cada element [count] tindrà un regressor per cada ronda
                results_list_for_each_regresor[count].append(round_regresor_results)

                count+=1

            models_for_each_round.append(gaussian_mixture_models)
        
    #PRINT MEAN RESULTS
    count=0
    for result_list in results_list_for_each_regresor:
        print("Mean results for regressor " + str(regresors[count]).split('(')[0])

        mean_results=np.mean(np.asarray(result_list),axis=0)
        result_names=["Predicted Mean", "Predicted stdv", "RootMeanSquareError" ,"Error stdv", "Residual Sum Squares (SSres)", "Data variance (SStot)", "R^2", "Computed R^2"]
        print_results(mean_results, result_names,-1)
        
        # print(bool(save_results_boolean))

        if save_results_boolean:
            # print(bool(save_results_boolean))

            save_results(save_to_google_drive_boolean,histograms_boolean[1],class_type,regresors[count],folds,number_of_gaussians,len(fold_indexes),samples_per_aqi_index,image_folder.split('/')[-1],descriptors,patch_sizes,multi_modal_weights,gaussian_mixture_keep_portion,result_list,result_names,norm_type,PCA)
            
        count+=1
    #saves the comparaision txt file and returns the index of the regressors with max R2
    max_regressors=compare_results(histograms_boolean[1],'R^2',regresors,fited_regressors)
    '''

def fit_regressor_and_compute_metrics(regresor,final_train_samples,train_labels,final_test_samples,test_labels,PCA,class_type,histograms_boolean,fold,folds):
    #PCA


    #SVM PREDICTION AND CLASSIFICATION
    t=time.time()
    print('Training ' + str(regresor).split('(')[0])
    if class_type=="AQI":
        clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
        #creating svm
        #encoded_train_samples hauria de tenir tamany Nsamples x Nfeatures

        print(np.asarray(train_labels).shape)
        print(np.asarray(train_labels).shape)
        clf.fit(final_train_samples, (train_labels))
        delete_last_lines()
        print('SVM trained in ' + '{:4.2f}'.format((time.time()-t)))
        #get prediction accuracy 
        accuracy=clf.score(final_test_samples, np.asarray(test_labels))
        print("--------------------------------------------------------------------------",end = '')
        print("Accuracy of " + str(k+1) + " round= " + str(accuracy))
        result_names=["Accuracy"]
        round_results = np.stack((accuracy))   
    elif class_type=="pollution":

        multi_regressor = MultiOutputRegressor(regresor)
        multi_regressor.fit(final_train_samples, (train_labels))
        predicted_labels=multi_regressor.predict(final_test_samples)

        visualize_residual_plots(histograms_boolean, multi_regressor.estimators_, final_test_samples, test_labels,fold+1)

        #creating svm
        delete_last_lines()
        print( str(regresor).split('(')[0] + ' trained in ' + '{:4.2f}'.format((time.time()-t)))
        #get prediction accuracy 

        
        #METRICS COMPUTATION
        root_mean_squared_errors=np.sqrt(metrics.mean_squared_error((test_labels), (predicted_labels),multioutput='raw_values'))
        predicted_mean=[compute_mean(predicted_labels[:,i]) for i in range(len(root_mean_squared_errors))]
        error_stdv =[compute_error_stdv(predicted_labels[:,i],np.asarray(test_labels)[:,i],root_mean_squared_errors[i]) for i in range(len(root_mean_squared_errors))]
        r_squares=(metrics.r2_score(np.asarray(test_labels), (predicted_labels),multioutput='raw_values'))
        s_ress=[compute_s_ress(predicted_labels[:,i],np.asarray(test_labels)[:,i]) for i in range(len(root_mean_squared_errors))]
        s_tot=[compute_s_tot(np.asarray(test_labels)[:,i]) for i in range(len(root_mean_squared_errors))]
        expected_r2=1.-(np.asarray(s_ress)/np.asarray(s_tot))
        stv=[compute_stdv(predicted_labels[:,i], predicted_mean[i]) for i in range(len(root_mean_squared_errors))]


        round_results = np.stack((predicted_mean,stv,root_mean_squared_errors,error_stdv,s_ress,s_tot,r_squares,expected_r2))
        result_names=["Predicted Mean", "Predicted stdv", "RootMeanSquareError" ,"Error stdv", "Residual Sum Squares (SSres)", "Data variance (SStot)", "R^2", "Computed R^2"]
        print_results(round_results, result_names,fold)

        train_histograms=data_historgrams(np.asarray(train_labels))
        test_histograms=data_historgrams(np.asarray(test_labels))
        plot_histograms(train_histograms,test_histograms, fold+1,folds, histograms_boolean)

    return (round_results,multi_regressor.estimators_)



def load_gmm(gaussian_mixtures_folder,descriptor_info,number_of_gaussians,norm_type):

    file_to_open_end = descriptor_info+'.npz'
    possible_files = [(int(f.split('_')[0].split('.')[0]),int(f.split('_')[3].replace('s',''))) for f in os.listdir(gaussian_mixtures_folder) if f.endswith('.npz') and  int(f.split('_')[2].replace('N',''))==number_of_gaussians and descriptor_info == f.split('_')[-2] and f.split('_')[-1] == str(norm_type+'.npz')]
    
    if len(possible_files)==0:
        print("No Gaussian Mixture Model found with these parametters, computing it")

    try:
        sorted_indexes=np.argmax(np.asarray(possible_files)[:,1])#agafem el model que sha fet amb mes samples
    except:
        return False
    models_with_max_samples=np.asarray(possible_files)[:,0][sorted_indexes]#ordenem els sites amb els index

    file_name=(([f for f in os.listdir(gaussian_mixtures_folder) if f.endswith('.npz') and int(f.split('_')[0].split('.')[0]) == models_with_max_samples ]))
    filepath = gaussian_mixtures_folder +'/'+ str(file_name[-1])
    npzfile = np.load(filepath)
    print("Got mixture model for " + descriptor_info + " from " + filepath)



    model = sklearn.mixture.GaussianMixture(n_components=number_of_gaussians, covariance_type='diag')
    model.means_ = npzfile["means"]
    model.covariances_ = npzfile["covs"]
    model.weights_ = npzfile["weights"]
    model.precisions_cholesky_ = npzfile["precisions_cholesky"]


    return model
def save_gmm(save_to_google_drive_boolean,folder_to_save,gmm: sklearn.mixture.GaussianMixture,descriptor_info,number_samples,number_gaussians,fold,norm_type):

    number_of_gmm = np.sum(np.asarray([1 for f in os.listdir(folder_to_save) if f.endswith('.npz')]))
    gaussian_name = str(int(number_of_gmm))+'_gmm_N'+str(number_gaussians)+'_'+str(number_samples)+'s_k'+str(fold+1)+'_'+descriptor_info+'_'+norm_type+'.npz'
    print('Saving GMM ' + gaussian_name + ' into folder ' + folder_to_save)
    means   = np.float32(gmm.means_)
    covs    = np.float32(gmm.covariances_)
    weights = np.float32(gmm.weights_)
    precisions_cholesky = np.float32(gmm.precisions_cholesky_)

    filepath = folder_to_save+'/'+gaussian_name
    np.savez(filepath, means=means, covs=covs, weights=weights, precisions_cholesky=precisions_cholesky)

    if save_to_google_drive_boolean:
        save_file_google_drive_path(filepath,'*/*','TFG/files/gmm')
def retrieve_or_compute_gmm(gaussian_mixtures_folder,feature_info,number_of_gaussians,norm_type,gmm_booleans,samples_from_descriptor,fold,save_to_google_drive_boolean):
    got_model=False
    if gmm_booleans[0]:#if we want to use an already existing gaussian
        got_model=load_gmm(gaussian_mixtures_folder,feature_info,number_of_gaussians,norm_type)
    if got_model:#if there is an existing model
        return got_model
    else:#if we did not find a stored gm model
        #NORMALIZE // SCALE
        rescaled_samples=normalize_data(samples_from_descriptor,norm_type)
        print(feature_info)
        print("     Computing gaussian mixture for descriptor "+feature_info  + " with " + str(number_of_gaussians) + " components")
        #Compute gmm
        got_model = generate_gaussian_mixture_model(rescaled_samples,number_of_gaussians)
        #Save GMM if wanted
        if gmm_booleans[1]:#if it must be saved
            check_folder_and_create(gaussian_mixtures_folder)
            save_gmm(save_to_google_drive_boolean,gaussian_mixtures_folder,got_model,feature_info, len(rescaled_samples),number_of_gaussians,fold,norm_type)
    
    return got_model

def print_results(results, result_names,round_):
    r_count=0
    for result in results:
        if round_ < 0:
            round_string = " mean of all rounds"
        else:
            round_string = " of " + str(round_+1) + " round= "
        print('{:->35}'.format(result_names[r_count] + ': ')+round_string + str([str('{:06.4f}'.format(r)) for r in result]))
        r_count+=1
def save_results(google_drive_boolean,folder_to_save,class_type,regresor,folds,number_of_gaussians,total_samples,samples_per_aqi_index,camera,descriptors,patch_sizes,multi_modal_weights,gaussian_mixture_keep_portion,result_list,result_names,normalization_type,pca_comp):
    todays_date=str(datetime.datetime.now()).split(' ')[0]
    hour_minute=str(datetime.datetime.now()).split(' ')[1]  
    file_name=folder_to_save.split('/')[-1]+\
            '_'+class_type+'_'+str(regresor).split('(')[0]+'_k'+str(folds)+'_ngm'+str(number_of_gaussians)+\
            '_'+str(total_samples)+'_'+str(samples_per_aqi_index)+'.txt'


    check_folder_and_create(folder_to_save)
    print("Results saved in: " + str(os.path.join(folder_to_save,file_name)))

    results_folder=folder_to_save.replace(folder_to_save.split('/')[-1],'')

    print(folder_to_save)
    print(folder_to_save)
    print(folder_to_save)
    print(folder_to_save)
    print(folder_to_save.replace(folder_to_save.split('/')[-1],''))
    print(folder_to_save.split('/'))
    '''
    #MOVE HISTOGRAMS TO THE SAME FOLDER
    for f in os.listdir(results_folder):
        if f.endswith('.png'):
            os.rename(results_folder+'/'+f, folder_to_save+'/'+f)
    '''
    myfile=open(os.path.join(folder_to_save,file_name),"w")

    myfile.write('Camera: ' + str(camera)+'\n')
    myfile.write('Descriptors: ' + str(descriptors)+'\n')
    myfile.write('Patch sizes: ' + str(patch_sizes)+'\n')
    myfile.write('Multimodal weights: ' + str(multi_modal_weights)+'\n')
    myfile.write('Keept portion of samples to do the GMM: ' + str(gaussian_mixture_keep_portion)+'\n')
    myfile.write('Normalization used: ' + str(normalization_type)+'\n')
    myfile.write('PCA coponents: ' + str(pca_comp)+'\n')
    round_count=0
    for results in result_list:#per cada ronda
        round_count+=1
        myfile.write("----------------------------------------------------------------------------------------" + '\n')
        myfile.write("Round " + str(round_count) + " results" + '\n')
        myfile.write('\n')
        metric_count=0
        myfile.write('{:>35}'.format(' '))
        [myfile.write(str('{:8}'.format(p))) for p in parameters_for_saving]
        myfile.write('\n')
        for metric in results:
            myfile.write('{:>35}'.format(result_names[metric_count] + ': '))
            [myfile.write(str('{:06.4f}'.format(i)) + ', ') for i in metric]
            myfile.write('\n')
            metric_count+=1
    mean_results = np.mean(np.asarray(result_list),axis=0)

    myfile.write("----------------------------------------------------------------------------------------" + '\n')
    myfile.write("Mean results" + '\n')
    myfile.write('{:>35}'.format(' '))
    [myfile.write(str('{:8}'.format(p))) for p in parameters_for_saving]
    myfile.write('\n')
    metric_count=0
    for metric in mean_results:
        myfile.write('{:>35}'.format(result_names[metric_count] + ': '))
        [myfile.write(str('{:06.4f}'.format(i)) + ', ') for i in metric]
        myfile.write('\n')
        metric_count+=1


    myfile.close()
    if google_drive_boolean:
        print("Saving results txt into Google Drive")
        result_folder_id=save_file_google_drive_path(os.path.join(final_folder_to_save,file_name),'text/plain','TFG/files/results/'+file_name.split('_')[0]+'_'+file_name.split('_')[1])
        for f in os.listdir(final_folder_to_save):
            if f.endswith('.png'):#if the file is a png file, i.e. the samples histogram
                print("Saving histogram " + f + " into Google Drive")
                save_file_google_drive_id(os.path.join(final_folder_to_save,f),'image/png',result_folder_id)
def compare_results(folder_with_results_txt,metric_used,regresors,fited_regressors):
    

    results=[]
    result_file_names=[]
    #get mean results of metric_used for each file
    for file in os.listdir(folder_with_results_txt):
        if file.endswith('.txt'):#if its a txt file
            myfile = open(os.path.join(folder_with_results_txt,file), "r")
            lines = myfile.readlines()
            for line_number in range(len(lines)-1,0,-1):#read it backwards (we will find the mean results first)
                line = lines[line_number]
                letters=line.split(':')[0]
                numbers=line.split(':')[-1]
                #print(letters.strip() + ' ' + metric_used)
                if letters.strip() == metric_used:
                    #print(str(line_number) +'   ' + str(numbers))
                    num_array=np.asarray(numbers.replace(' \n','').split(',')[0:-1])
                    results.append(num_array)
                    name=file.replace('_pollution','').split('_k')[0:1]
                    result_file_names.append(name)
                if letters.strip() == 'Mean results':#if we have passed the Mean results line, from that everything are the results for each round
                    break
    #results.append([10,-10,10,-10,10,-10])
    #result_file_names.append('fghjkl')
    
    max_strip=np.amax(np.asarray([(len(name[0])) for name in result_file_names]))
    half_strip_formatter="'{:^" + str(max_strip) + "}'"
    result_array=np.asarray(results,dtype=np.float64)
    max_index=[np.argmax(result_array[:,column]) for column in range(result_array.shape[1])]



    comparison_file=open(os.path.join(folder_with_results_txt,'comparison_file.txt'),"w")
    
    #parametter names
    [comparison_file.write(half_strip_formatter.format(p)) for p in parameters_for_saving]
    comparison_file.write('\n')    
    #values list
    [comparison_file.write(half_strip_formatter.format(result_array[max_index[i],i])) for i in range(len(max_index))]
    comparison_file.write('\n')
    #file name list
    [comparison_file.write(half_strip_formatter.format(result_file_names[i][0])) for i in max_index]
    comparison_file.write('\n')
    
    best_reg_names=[result_file_names[i][0] for i in max_index]
    used_regressor_names=[str(r).split('(')[0] for r in regresors]
    count=0
    print(len(best_reg_names))
    print(best_reg_names)
    print(used_regressor_names)

    for regressor in best_reg_names:#per cada parametre regressor es el nom del regressor utilitzat
    
        #per saber l'index del regresor bo dintre de fited_regressors
        used_reg_count=0
        for used_regressor in used_regressor_names:#recorrer els que hem fet servir i buscar el que te el mateix nom que el maxim
            if used_regressor == regressor.split('_')[-1]:#si es el maxim per aquest parametre pues retorna el numero
                print(used_regressor + '   ' + regressor.split('_')[-1])
                break
            else:
                used_reg_count+=1
        max_regressor=fited_regressors[used_reg_count]#take the regressor corresponding to the best for this parametter
        regresor=max_regressor[0]#regressor trained in first round
        print(str(regressor[0]).split('(')[0])
        # [print(psr.coef_) for psr in regresor]
        # [print(np.mean(r.coef_)) for r in regresor]

        
        # save the model to disk
        reg_name=str(regresor[count]).split('(')[0]
        model_folder=folder_with_results_txt+'/linear_models'
        check_folder_and_create(model_folder)
        print("Saving regressor "+ str(reg_name)+ " for parameter " + parameters_for_saving[count])
        
        filename = parameters_for_saving[count]+'_'+reg_name+'.sav'
        pickle.dump(regresor[count], open(model_folder+'/'+filename, 'wb'))
        count+=1


    return [result_file_names[i][0] for i in max_index]





def data_historgrams(data,bins=50):
    histograms=[]
    for i in range(data.shape[1]):
        (n,b)  = np.histogram(data[:,i],bins=bins)
        histograms.append((n,b))
    return histograms
def plot_histograms(train_histograms,test_histograms,round,max_round,boolean):
    #plt.ion()

    for i in range(len(train_histograms)):
        #plt.ioff()
        '''
        if round==max_round and i==len(train_histograms):#just if its the final round and the final histogram
            print('hgfdsdfghjkjhgfds')
            plt.ion()
        else:
            print('AAAAAAAAAAAAAAASASASASADAEADAFADASASASA')
            plt.ioff()
            '''
        plt.figure()
        plt.plot(train_histograms[i][1][:-1],train_histograms[i][0],label='Train samples',color='darkgreen')
        plt.plot(test_histograms[i][1][:-1],test_histograms[i][0],label='Test samples',color='firebrick')
        x_step=1
        #print(test_histograms[i][1][0:x_step:50])
        #plt.xticks(test_histograms[i][1][0:x_step:-1])
        plt.xlabel(switcher_xlabel_plot.get(i,'Invalid parameter name'))
        plt.title(title_switcher.get(i,'hgfds') + " histogram for round " + str(round))
        plt.ylabel('Number of samples')
        plt.legend()

        if boolean[1]:#save them
            plt.savefig(boolean[1]+'/'+title_switcher.get(i,'hgfds') + " histogram for round " + str(round)+'.png', bbox_inches='tight')
        if bool(boolean[0]):#print them
            if round==max_round and i==len(train_histograms)-1:#just if its the final round and the final histogram
                plt.show()
            else:
                plt.draw()
        else:
            plt.close()


def plot_histograms_just_report(data,boolean,folder_to_save,bins=50):
    #plt.ion()
    histograms=[]
    print(np.asarray(data).shape)
    # #for data taken with the get_fields_from_table
    # for d in data:
    #     (n,b)  = np.histogram(d,bins=bins)
    #     histograms.append((n,b))
    #     print(np.asarray(d).shape)

    #for data taken with the get names mathod
    histograms=[]
    for i in range(data.shape[1]):
        (n,b)  = np.histogram(data[:,i],bins=bins)
        histograms.append((n,b))


    check_folder_and_create(folder_to_save)
    for i in range(len(histograms)):

        plt.figure()

        plt.plot(histograms[i][1][:-1],histograms[i][0],label='Train samples',color='darkgreen')
        x_step=1
        #print(test_histograms[i][1][0:x_step:50])
        #plt.xticks(test_histograms[i][1][0:x_step:-1])
        plt.xlabel(switcher_xlabel_plot.get(i,'Invalid parameter name'))
        plt.title(title_switcher.get(i,'hgfds') + " histogram")
        plt.ylabel('Number of samples')

        if boolean[1]:#save them
            plt.savefig(folder_to_save+'/'+title_switcher.get(i,'hgfds') + " histogram " + str(bins) +" bins.png", bbox_inches='tight')
        if bool(boolean[0]):#print them
            if i==len(histograms)-1:#just if its the final round and the final histogram
                plt.show()
            else:
                plt.draw()
        else:
            plt.close()

def visualize_residual_plots(plot_boolean,estimators, test_samples, test_labels,round):
    count=0
    for estimator in estimators:
        temp_test_labels=np.asarray(test_labels)[:,count]
        
        fig = plt.figure()

        visualizer = ResidualsPlot(estimator)
        visualizer.score(test_samples, temp_test_labels)  # Evaluate the model on the test data
        if plot_boolean[1]:#save them
            save_path=plot_boolean[1]+'/residual plots of regressor '+str(estimator).split('(')[0]+' for parameter'+str(parameters_for_saving[count]) + " and round " + str(round)+'.png'
            fig.savefig(save_path, bbox_inches='tight')
        if plot_boolean[0]:
            visualizer.poof()
        else:
            plt.close()
        count+=1




###GOOGLE DRIVE 
def get_google_drive_credentials(credential, credentials_file_name):
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credential = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not credential or not credential.valid:
        if credential and credential.expired and credential.refresh_token:
            credential.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file_name, SCOPES)
            credential = flow.run_local_server()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credential, token)
    return credential
def drive_folder_id_from_path(service,path,parent_id,count):
    folder =  path.split('/')[0]#for each folder in the drive path, starting for the root one
    if count==0:#first swarch we don't know the parent id
        results = service.files().list(#get folders with name = ' '
            q="mimeType = 'application/vnd.google-apps.folder'  and name = '" + str(folder)+ "'",
            spaces='drive',
            fields="files(id, name, mimeType,parents)").execute()
    else:
        results = service.files().list(#get folders with name = ' '
            q="mimeType = 'application/vnd.google-apps.folder'  and name = '" + str(folder)+ "' and '" + parent_id+"' " + "in parents",
            spaces='drive',
            fields="files(id, name, mimeType,parents)").execute()


    if len(results['files'])==0:#si la carpeta no existeix
        folder_metadata={'name': path.split('/')[-1], 'parents' : [parent_id],'mimeType' :'application/vnd.google-apps.folder'}
        new_folder = service.files().create(body=folder_metadata,
                                    fields='id').execute()

        return new_folder['id']


    if path.replace((folder+'/'),'')!=path:
        final_id = drive_folder_id_from_path(service,path.replace((folder+'/'),''),results['files'][0]['id'],count+1)
    else: 
        final_id=results['files'][0]['id']#si es l'ultim tornem la seva

    return final_id#return first element with condition
def save_file_google_drive_path(file_path,mimeType,google_drive_path):
    #mimeType must be: 'image/jpeg', image/png, text/plain, etc
    #drive parameters to create service
    creds= None
    creds_name='credentials.json'
    creds = get_google_drive_credentials(creds, creds_name)
    service = build('drive', 'v3', credentials=creds)
    

    parent_folder_id = drive_folder_id_from_path(service,google_drive_path,0,0)#parent_id==0 because we dont know it, wont be used in the first round 
    file_metadata = {'name': file_path.split('/')[-1], 'parents' : [parent_folder_id]}
    media = MediaFileUpload(file_path)
    file = service.files().create(body=file_metadata,
                                    media_body=media,
                                    fields='id').execute()

    return parent_folder_id
def save_file_google_drive_id(file_path,mimeType,google_drive_folder_id):
    #mimeType must be: 'image/jpeg', image/png, text/plain, etc
    #drive parameters to create service
    creds= None
    creds_name='credentials.json'
    creds = get_google_drive_credentials(creds, creds_name)
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': file_path.split('/')[-1], 'parents' : [google_drive_folder_id]}
    media = MediaFileUpload(file_path)
    file = service.files().create(body=file_metadata,
                                    media_body=media,
                                    fields='id').execute()

def copy_matlab_features_and_pollution_data_from_folder_to_folder(from_folder, to_folder):
    check_folder_and_create(to_folder)#create destiny folder
    for sub_folder in os.listdir(from_folder):#for each folder in the
        destiny_subfolder=to_folder+'/'+sub_folder
        check_folder_and_create(destiny_subfolder)
        names_file = open(os.path.join(to_folder,sub_folder,'image_names.txt'),"w")
        if not sub_folder.endswith('.DS_Store'):
            for file in os.listdir(os.path.join(from_folder,sub_folder)):
                if os.path.isdir(os.path.join(from_folder,sub_folder,file)) and file == 'features':#si es la carpeta de features
                    check_folder_and_create(os.path.join(to_folder,sub_folder,file))#crear carpeta
                    for feature in os.listdir(os.path.join(from_folder,sub_folder,file)):#copiar cada fitxer de la carpeta
                        copyfile(os.path.join(from_folder,sub_folder,file,feature), os.path.join(to_folder,sub_folder,file,feature))
                elif file.endswith('.txt'):
                    copyfile(os.path.join(from_folder,sub_folder,file), os.path.join(to_folder,sub_folder,file))
                elif file.endswith('.jpg'):
                    names_file.write(str(file + '\n'))




if __name__ == '__main__':
    (db_connection,db_cursor)=pp.initialise_db('pollution')
    cameras_closer_than=pp.get_cameras_with_sites_closer_than(db_cursor, 10,'O2_DISTANCE, SO2_DISTANCE, CO_DISTANCE, NO2_DISTANCE, PM25_DISTANCE, PM10_DISTANCE')#all camera paramters
    [print(c) for c in cameras_closer_than]
    print(len(cameras_closer_than))
    pp.close_db_cursor(db_connection)
    #compare_results('/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/results/20190522_1912', 'R^2')

#[print(s) for s in struct]
#r=count_samples_for_aqi_index('/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/pollution_data')
