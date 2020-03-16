import pollution_parser as pp
import monitor_parser as mp

import argparse
import os
import sys
import scipy.io
import time
import utils as u
import numpy as np
from sklearn import svm
from sklearn import linear_model
from datetime import date
import datetime
import pickle

param_switcher = ["gas_ozone", "gas_so2", "gas_co", "gas_no2","poll_25","poll_10"]

def main(args): 

    regresor = svm.SVR(gamma='scale')#ElasticNet(random_state=0)#

    if args.image_folder:
        folder=args.image_folder
    

    #if len(args.patch_sizes)>1:
    #    patch_sizes=args.patch_sizes[1]
    #else:
    multi_modal_weights=args.multi_modal_weights
    patch_sizes=args.patch_sizes
    print(patch_sizes)

    #if len(args.descriptors)>1:
    #    descriptors=args.descriptors[1]
    #else:
    descriptors=args.descriptors
    regresors=[]
    for reg in args.regresor:
        print(reg)
        if reg == 'SVM':
            regresor = svm.SVR(gamma='scale')#ElasticNet(random_state=0)#
        elif reg == 'ELN':
            regresor = linear_model.ElasticNet(random_state=0)#
        elif reg == 'BAY':
            regresor = linear_model.BayesianRidge()
        elif reg == 'LIN':
            regresor = linear_model.LinearRegression()
        elif reg == 'LASSLAR':
            regresor=linear_model.LassoLars()
        elif reg == 'LASS':
            regresor=linear_model.Lasso(alpha=0.1)
        elif reg == 'THE':
            regresor=linear_model.TheilSenRegressor()
        elif reg == 'ARDR':
            regresor = linear_model.ARDRegression()
        regresors.append(regresor)
    print(args.regresor)
    print(regresors)


    '''
    for item in classifiers:
        print(item)
        clf = item
        clf.fit(trainingData, trainingScores)
        print(clf.predict(predictionData),'\n')
    '''


    gaussian_mixture_keep_portion=args.gmm_keep_portion
    pca_comp=args.pca_comp
    folds=args.folds
    number_of_gaussians=args.number_of_gaussians
    samples_per_class=args.samples_per_calss
    
    print_hist=args.print_hist
    save_gmm=args.save_gmm
    
    
    
    todays_date=str(datetime.datetime.now()).split(' ')[0]
    hour_minute=str(datetime.datetime.now()).split(' ')[1]
    sub_folder_name=str(todays_date.split('-')[0])+str(todays_date.split('-')[1])+str(todays_date.split('-')[2])+'_' +str(hour_minute.split(':')[0])+str(hour_minute.split(':')[1])
    if args.run_number:
        sub_folder_name=sub_folder_name+'-'+str(args.run_number)
    count=0
    while os.path.exists(args.save_hist_folder+'/'+sub_folder_name):
        count+=1
        sub_folder_name=sub_folder_name.split('-')[0]+'-'+str(count)

    hist_folder=u.check_folder_and_create(args.save_hist_folder+'/'+sub_folder_name)
    print(hist_folder)
    print(hist_folder)
    print(hist_folder)
    print(hist_folder)
    save_results_boolean=args.save_results
    normalizaton_type=args.normalizaton_type
    use_latest_gmm=args.use_saved_gmm
    save_to_drive_boolean=args.save_to_drive

    hist_bool=[print_hist,hist_folder]#first argument is for showing them on screen, second for saving them''
    gmm_booleans=(use_latest_gmm,save_gmm)




    src_path='/Volumes/Data_HD/Users/Carles/TFG/codes/'
    #[print(ss) for ss in argv]

    (db_connection,db_cursor)=pp.initialise_db('pollution')


    t=time.time()
    print('Getting all cameras closer than' )
    # cameras_closer_than=pp.get_cameras_with_sites_closer_than(db_cursor, 100,'camera_id,O2, SO2, CO, NO2, PM25, PM10, WIND, TEMP, PRES, RH')
    # #cameras_closer_than=[pp.get_fields_from_table(db_cursor,'camera_sites','camera_id,O2, SO2, CO, NO2, PM25, PM10, WIND, TEMP, PRES, RH',str('where camera_id='+ (f.split('_')[1].replace('.jpg','')).lstrip("0") ))[0]    for f in os.listdir('/Volumes/Data_HD/Users/Carles/TFG/codes/images_closer_than_10/for_real') if f.endswith('.jpg')]

    #cameras_closer_than=pp.get_fields_from_table(db_cursor,'camera_sites','camera_id,O2, SO2, CO, NO2, PM25, PM10, WIND, TEMP, PRES, RH','where camera_id=23971')
    u.delete_last_lines()
    print('Got all cameras in ' + '{:4.2f}'.format((time.time()-t)))

    #print(cameras_closer_than)

    '''
    t=time.time()
    print('Starting Matlab engine')
    #eng=matlab.engine.start_matlab()
    u.delete_last_lines()
    print('Engine started in ' + '{:4.2f}'.format((time.time()-t)))
    '''
    #mp.make_image_patches(150,src_path+'images_closer_than_10','2017_00032984.jpg',src_path+'images_closer_than_10/patches',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

    t=time.time()
    print('Gettin pollution data')
    #all_pollution_data=pp.get_all_pollution_data(db_cursor)
    #all_pollution_data=pp.get_all_pollution_data_for_cam(db_cursor,398)
    u.delete_last_lines()
    print('Pollution data got in ' + '{:4.2f}'.format((time.time()-t)))

    '''
    #agafar_possibles
    ff=os.listdir('/Volumes/Data_HD/Users/Carles/TFG/codes/images_closer_than_10/for_real')
    listOfNum=[f.split('_')[1].replace('.jpg','') for f in ff if f.endswith('.jpg')]
    get_list=[int(s.lstrip("0")) for s in listOfNum]
    '''



    #u.count_poll_data('/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/pollution_data/00000398')
    #cameras_to_get=[c for c in cameras_closer_than if c[0] == 398]
    #pp.save_pollution_data_for_cam(db_cursor,cameras_to_get,src_path+'final_data/pollution_data')
    #cameras_to_get=[c for c in cameras_closer_than if [True for cc in get_list if c[0]==cc]]
    #mat = scipy.io.loadmat('/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/image_data/11/raw_descriptors'+'/20170108_193201_patch1_12_color.mat')

    #(s,c)=u.create_classification_data('/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/image_data')
    #print(mat)

    #clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    #clf.fit(s, c)
    
    
    folder_met_data=folder.replace(os.path.join(folder.split('/')[-2],folder.split('/')[-1]),'pollution_data')

    clases_to_get=[int(i) for i in os.listdir(folder) if i != '.DS_Store']    
    # if args.not_all_data:
    clases_to_get=[32,42,47,52,57,62,67,77]#[32, 36, 42, 47, 52, 57, 62, 67, 71, 77, 81, 87, 90]#
    #clases_to_get=[32]
    class_type="pollution"



    
    #GETTING SAMPLES
    t=time.time()
    print("Getting sample names")
    if samples_per_class<=0:#to get the number of samples that has the folder with less samples
        samples_per_class=min(np.asarray([len(os.listdir(os.path.join(folder,f))) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f))]))
    #(sample_names,sample_classes,sample_path,sample_features)=u.get_sample_names_meateorological(clases_to_get,samples_per_class,folder_met_data,1,[c[0] for c in cameras_closer_than])#u.get_sample_names(clases_to_get,samples_per_class,folder)
    
    #u.get_samples_by_pollution(clases_to_get,-1,'/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database',[398],"pollution",True)
    
    (sample_names,sample_classes,sample_pollution_paths,sample_features)=u.get_samples(clases_to_get,samples_per_class,folder,class_type,descriptors,patch_sizes)
    u.delete_last_lines()
    print('Sample names got in ' + '{:4.2f}'.format((time.time()-t)))
    #PERFORMING CROSSVALIDATION
    print(np.asarray(sample_classes).shape)
    u.cross_validation(sample_names,sample_classes,sample_pollution_paths,sample_features,folder,folds,multi_modal_weights,gaussian_mixture_keep_portion,number_of_gaussians,descriptors,patch_sizes,class_type,samples_per_class,regresors,save_results_boolean,hist_bool,save_to_drive_boolean,gmm_booleans,normalizaton_type,pca_comp)
    
    
    model_folder=hist_folder+'/linear_models'
    linear_models=[]
    for regressor in os.listdir(model_folder):
        reg=pickle.load(open(model_folder+'/'+regressor, 'rb'))
        linear_models.append(reg)

    #[print(reg.coef_) for reg in linear_models]
    


    '''
    #TO GET HISTOGRAMS
    u.plot_histograms_just_report(np.asarray(sample_classes),(1,1),'/Volumes/Data_HD/Users/Carles/TFG/codes/histograms/all_samples/398',50)
    u.plot_histograms_just_report(np.asarray(sample_classes),(1,1),'/Volumes/Data_HD/Users/Carles/TFG/codes/histograms/all_samples/398',25)
    #all cams closer than 100 histograms
    
    
    
    #FOR ALL CAMERSA CLOSER THAN
    
    cameras_closer_than=pp.get_cameras_with_sites_closer_than(db_cursor, 10,'O2, SO2, CO, NO2, PM25, PM10')#all camera paramters
    #cameras_closer_than=pp.get_fields_from_table(db_cursor,'camera_sites','camera_id,O2, SO2, CO, NO2, PM25, PM10, WIND, TEMP, PRES, RH','where camera_id=23971')
    sites=[[] for p in param_switcher]
    print(len(sites))

    for c in cameras_closer_than:
        count=0
        for site in c:
            sites[count].append(site)
            count+=1
    [print(len(sites[i])) for i in range(len(param_switcher))]
    final_site_lists=[]
    for i in range(len(param_switcher)):
        final_site_lists.append(list(set(sites[i])))

    print("'" +  "'  OR site =  '".join(final_site_lists[0]) + "'")
    pollution_data=[]
    pollution_data=[pp.get_fields_from_table(db_cursor,param_switcher[i],'sample',str(" where site = '" + "' OR site = '".join(final_site_lists[i]) + "'")) for i in range(len(param_switcher))]
    print(np.asarray(pollution_data).shape)
    print(len(pollution_data))
    print(len(pollution_data[0]))

    fucking_data=[]
    [fucking_data.append(p) for p in pollution_data]
    [print(np.asarray(p).shape) for p in pollution_data]
    print(np.asarray(fucking_data).shape)


    u.plot_histograms_just_report(np.asarray(pollution_data),(1,1),'/Volumes/Data_HD/Users/Carles/TFG/codes/histograms/all_samples/all',25)
    u.plot_histograms_just_report(np.asarray(pollution_data),(1,1),'/Volumes/Data_HD/Users/Carles/TFG/codes/histograms/all_samples/all',50)
    u.plot_histograms_just_report(np.asarray(pollution_data),(1,1),'/Volumes/Data_HD/Users/Carles/TFG/codes/histograms/all_samples/all',100)
    # #print(" or ".join(final_site_lists[1]))
    #[print(len(f)) for f in final_site_lists]
    #[print(len(c)) for c in cameras_closer_than]
    '''

    # pollution_data=[ for table in param_switcher]

    #print(np.asarray(cameras_closer_than).shape)
    


    #eng.quit()
    pp.close_db_cursor(db_connection)
    
    pass




if __name__ == "__main__":
    #default parametters
    default_multi_modal_weights=[0.8,1]#[0.8,1]
    pca_comp=0
    folds=3
    number_of_gaussians=[200,10]
    save_to_drive_boolean=0
    save_gmm = 1
    save_hist_folder='/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/results'
    default_image_folder='/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/image_data/00000398'
    default_samples_per_class=-1


    use_latest_gmm = 1#the index says the number of gmm that wants to be used
    save_results=1
    normalizaton_type='else'#
    default_regresor=['LIN','LASS']
    default_patch_sizes=[12]
    default_descriptors=['centrist']#'color', 'centrist'

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help = "specify the folder with the image_data (deafault = " + default_image_folder + ")", default=default_image_folder,type=str)
    parser.add_argument("--multi_modal_weights", help = "weights for multi modal classification (image feature weights, metheorological feature weights, default = " + str(default_multi_modal_weights)+ ")",default=default_multi_modal_weights,type=float, nargs=2,action='store')
    parser.add_argument("--gmm_keep_portion", help = "portion of the data used to compute the Gaussian Mixture Model (default = 0.5)",default=0.5,type=float)
    parser.add_argument("--pca_comp", help = "number of PCA components (default = "+str(pca_comp)+")",default=pca_comp,type=int)
    parser.add_argument("-F","--folds", help = "number of folds performed in cross validation (default = " + str(folds) + ")",default=folds,type=int)
    parser.add_argument("--descriptors", help = "descriptors used for as features (can be: 'sift', 'centrist', 'color' defaults = " + str(default_descriptors) + ")",type=str,default=default_descriptors, nargs='+',action='store')
    



    parser.add_argument("--patch_sizes", help = "patch sizes used for the descriptors (can be: 12, 16, 20, defaults = " + str(default_patch_sizes) + ")",type=int,default=default_patch_sizes, nargs='+',action='store')
    parser.add_argument("-GMM","--number_of_gaussians", help = "number of gaussians used per gaussian mixture (default = " + str(number_of_gaussians) + ")",default=number_of_gaussians,type=int, nargs=2,action='store')
    parser.add_argument("--normalizaton_type", help = "normalizaton used for the data (can be: 'norm', 'scale', if something else no normalization will be applied, default is " + str(normalizaton_type) + ")",default=normalizaton_type,type=str)
    parser.add_argument("-R","--regresor", help = "regresor used (can be: 'SVM'(support vector machine), 'ELN' (elasticnet), 'BAY' (bayesian ridge), 'LIN' (linear regression), 'LASSLAR' (lasso least angle regression), 'LASS'  (lasso regression), 'THE' (theil sen regression), 'ARDR' (bayesian ARD regression), deafult is " + default_regresor[0] + ")",default=default_regresor,nargs='+',action='store')
    parser.add_argument("-N","--samples_per_calss", help = "number of samples per class (AQI index) used (default = " + str(default_samples_per_class) + ")",default=default_samples_per_class,type=int)
    

    parser.add_argument("--use_saved_gmm", help = "use an already saved GMM for the features (if posible) ()",default=use_latest_gmm,type=int)
    parser.add_argument("--save_to_drive", help = "save results and GMM to google drive (deafult = " + str(save_to_drive_boolean) + ")",default=save_to_drive_boolean,type=int)
    parser.add_argument("--save_results", help = "save results (txt and histograms if generated, default = " + str(save_results) + ")",default=save_results,type=int)
    parser.add_argument("--save_gmm", help = "save gaussian mixture models generated (default = " + str(save_gmm) + ")",default=save_gmm,type=int)
    parser.add_argument("--print_hist", help = "Show histograms (deafult = 0)",default=0,type=int)
    parser.add_argument("--save_hist_folder", help = "folder to save histograms (deafult = " + str(save_hist_folder) + ")",default=save_hist_folder,type=str)
    
    parser.add_argument("--not_all_data", help = "to use just some classes of teh data",type=bool)
    parser.add_argument("--run_number", help = "number of the run if it belongs to a multiple run, this is done to create a result folder for each run",type=int)

    args = parser.parse_args()

    print(args)
    #print(args.samples_per_calss)
    #print(args.number_of_gaussians)

    main(args)




