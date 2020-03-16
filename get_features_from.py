import matlab.engine
import argparse
import time
import sys
import os

def get_matlab_features(eng:matlab.engine, image_path,data_path,nom_foto,patch_sizes,stride,max_size):

    t=time.time()
    print("Getting features from: " + nom_foto)
    eng.first_test(image_path,nom_foto,data_path,patch_sizes, stride, max_size)
    print('Saved features into: '+ data_path + '      in ' + '{:4.2f}'.format((time.time()-t)) + ' seconds')

def get_matlab_features_for_cam(folder_with_cam_data,eng:matlab.engine,patch_sizes,stride,max_size):
    for f in os.listdir(folder_with_cam_data):
        if os.path.isdir(folder_with_cam_data+'/'+f) and not os.path.isdir(folder_with_cam_data+'/'+f+'/features'):
            print("Getting Matlab features for cam: " + str(folder_with_cam_data.split('/')[-1]) + ", folder: " + str(f))
                #check_folder_and_create(folder_with_cam_data+'/'+f+'/features'):
            
            for image in os.listdir(folder_with_cam_data+'/'+f):
                final_data_path=folder_with_cam_data+'/'+f+'/features'
                if image.endswith('.jpg'):
                    get_matlab_features(eng, folder_with_cam_data+'/'+f,final_data_path,image,patch_sizes,stride,max_size)
            
def delete_last_lines(n=1):
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


if __name__ == "__main__":
    default_data_folder = '/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/image_data/'
    default_patch_sizes=[12,16,20]
    def_stride=8
    def_max_size=300

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help = "specify the folder with the image data (deafault = " + default_data_folder + ")", default=default_data_folder,type=str)
    parser.add_argument("camera_folder", help="specify the folder with the camera images",type=str)
    parser.add_argument("-P","--patches", help = "patch sizes used for the descriptors (can be: 12, 16, 20, defaults = " + str(default_patch_sizes) + ")",type=int,default=default_patch_sizes, nargs='*',action='append')
    parser.add_argument("-S","--stride", help="specify the stride used for the patches (default = " + str(def_stride) + ")",type=int,default=def_stride)
    parser.add_argument("-M","--max_size", help="specify the maximum size of the patches (default = " + str(def_max_size)+ ")",type=int,default=def_max_size)

    args = parser.parse_args()

    t=time.time()
    print('Starting Matlab engine')
    eng=matlab.engine.start_matlab()
    delete_last_lines()
    print('Engine started in ' + '{:4.2f}'.format((time.time()-t)))


    final_folder=str(args.image_folder)+str(args.camera_folder)
    get_matlab_features_for_cam(final_folder,eng,args.patches,args.stride,args.max_size)



