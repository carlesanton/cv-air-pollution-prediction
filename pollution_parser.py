
from datetime import datetime
import time
import math
#per parsejar
import requests
import csv
import sqlite3
###
import os
#per agrupar cada 2
import re
#utilities
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import json
import sys
import zipfile
#google drive
import pickle

###
import io
import http.client

###MATLAB


# If modifying these scopes, delete the file token.pickle.
SCOPES =['https://www.googleapis.com/auth/drive']
param_switcher = {
    0: "gas_ozone",
    1: "gas_so2",
    2: "gas_co",
    3: "gas_no2",
    4: "poll_25",
    5: "poll_10",
    6: "met_wind",
    7: "met_temperature",
    8: "met_pressure",
    9: "met_humidity",
}
def lat_lon_2_dis(lat1,lat2,lon1,lon2):
    # approximate radius of earth in km
    R = 6373.0

    dlon = (lon2*math.pi/180) - lon1*math.pi/180
    dlat = lat2*math.pi/180 - lat1*math.pi/180

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    if a > 0:
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
    else: #perque sino peta
        c = 1000
    distance = R * c
    return distance
def get_bounding_box(lat,lon,radi):
    # approximate radius of earth in km
    km_grau_lat=0
    km_grau_lon=0
    if abs(lat)<40:
        km_grau_lat=110.57
        km_grau_lon=111.32
    elif abs(lat)<80:
        km_grau_lat=111.03
        km_grau_lon=85.39
    elif abs(lat)>80 and abs(lat)<90:
        km_grau_lat=111.69
        km_grau_lon=19.39

    d_lat = radi / km_grau_lat
    d_lon = radi / km_grau_lon

    return (lat+d_lat,lat-d_lat,lon+d_lon,lon-d_lon)

def in_bounding_box(bbox,lat,lon):
    if bbox[0]>lat and bbox[1]< lat and bbox[2]> lon and bbox[3]<lon:
        return True
    else:
        return False

#check if folder exists and creates it
def check_folder_and_create(path_to_folder, print_b):
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
        if print_b:
            print('Folder ' + path_to_folder + ' created')
    return os.path.exists(path_to_folder)

#POLLUTION PARSE
#GAS PARSER
def parse_pollution_csv(cursor: sqlite3.Cursor,path,table_name):
    #https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_cbsa_2017.zip
    #(sample_id PRIMARY KEY AUTOINCREMENT, site CHARACTER, data datetime, sample FLOAT(23) , unit TINYTEXT, lat FLOAT, lon FLOAT)")
    paraules=path.split('_')
    count=0
    reader = csv.reader(open(path), delimiter=',')
    inster_string="insert into "+table_name+ " values (?,?,?,?,?,?,?)"
    print(inster_string)
    
    for row in reader:
        #
        if reader.line_num !=1:
            site = row[0]+'-'+row[1]+'-'+row[2]
            date_time = row[11]+' '+row[12]
            sample=row[13]
            unit=row[14]
            lat=row[5]
            lon=row[6]
            cursor.execute(inster_string,(count,site, date_time, sample, unit, lat, lon) )
            if count%10000==0:
                print('Inserted sample: '+str(count) + ' into ' +table_name)
        count+=1 
#parse aqi
def parse_aqi_csv(cursor: sqlite3.Cursor,path):
    paraules=path.split('_')
    count=0
    reader = csv.reader(open(path), delimiter=',')
    for row in reader:
        #print((row[6]+'/'+row[2]))
        if reader.line_num !=1:
            site = row[8]
            date_time = row[4]
            sample=row[5]
            category=row[6]
            def_param=row[7]
            cursor.execute("insert into poll_aqi values (?,?,?,?,?,?)", (count, site,date_time, sample,category,def_param))
            if count%10000==0:
                print('Inserted sample: '+str(count) + ' into poll_aqi: ' + str(site) + '   ||  ' + str(date_time) + '   ||  '+ str(sample) + '   ||  '+ str(category) + '   ||  '+ str(def_param))
        count+=1
def parse_gas_folder(cursor: sqlite3.Cursor,path):
    switcher = {
        44201: "gas_ozone",
        42401: "gas_so2",
        42101: "gas_co",
        42602: "gas_no2"
    }
    for f in os.listdir(path):
        if f.endswith(".csv"):
            table_name = switcher.get(int(f.split('_')[1]), "Invalid parameter code")
            parse_pollution_csv(cursor,str(path+'/'+f),table_name)
#parse meteorolgical
def parse_meteorological_folder(cursor: sqlite3.Cursor,path):
    switcher = {
        'WIND': "met_wind",
        'TEMP': "met_temperature",
        'PRESS': "met_pressure",
        'RH': "met_humidity"
    }
    for f in os.listdir(path):
        if f.endswith(".csv"):
            table_name = switcher.get((f.split('_')[1]), "Invalid parameter code")
            parse_pollution_csv(cursor,str(path+'/'+f),table_name)


#GOOGLE DRIVE FUNCTIONS
def unzip_file_to(zip_path,destiny_path,delete):
    try:
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(destiny_path)
        zip_ref.close()
        if delete:
            os.remove(zip_path)
        return 1
    except zipfile.BadZipFile:
        print("BadZipFile exception with folder: " + zip_path)
        return 0
                
#get just one zip file for camera and put one image from it into the sample folder
def eliminar_carpeta(folder_to_delete):
    for f in os.listdir(folder_to_delete):
        os.remove(folder_to_delete+'/'+f)
    os.rmdir(folder_to_delete)

def filenames_to_txt(folder_path):
    file_name='/file_names.txt'
    for f in os.listdir(folder_path):
        if not os.path.exists(folder_path+file_name):
            myfile = open(folder_path+file_name,"w")
        else:
            myfile = open(folder_path+file_name,"a")
        myfile.write(f+'\n')
        myfile.close()
def list_from_txt(file_path):
    myfile = open(file_path, "r")
    names = []
    for line in myfile:
        new_name=line.split()[0].split('_')[1].lstrip('0').replace('.jpg','')
        names.append(new_name)

    return names

def print_table(cursor: sqlite3.Cursor,tab_name):
    query = "SELECT * FROM " + tab_name
    cursor.execute(query)
    totes=db_cursor.fetchall()
    count=0
    for r in totes:
        print(r)
def get_fields_from_table(cursor:sqlite3.Cursor,table_name,fields,condition):
    cursor.execute("SELECT "+fields+" FROM "+table_name+" " + condition)
    tots=cursor.fetchall()
    sit=[]
    return [s for s in tots]
def get_sites_for_parameter(cursor:sqlite3.Cursor,k,switcher,avoid):
    if np.sum([b+1 for b in avoid if b==k],dtype=bool):#si l'hem d'evitar
        return []
    else:#si no la hem d'evitar
        print('Getting sites from table: ' + switcher.get(k, "Invalid parameter code"),end='\r')
        return get_fields_from_table(cursor,switcher.get(k, "Invalid parameter code"),'distinct site, lat, lon'," ")


def create_camera_sites_relation_table(cursor:sqlite3.Cursor,cam_cursor:sqlite3.Cursor):
    radi_threshold=20000#mig perimetre de la terra
    cursor.execute('''CREATE TABLE IF NOT EXISTS camera_sites (camera_id INTEGER PRIMARY KEY, 
                    O2 CHARACTER, O2_DISTANCE INTEGER,
                    SO2 CHARACTER, SO2_DISTANCE INTEGER,
                    CO CHARACTER, CO_DISTANCE INTEGER,
                    NO2 CHARACTER, NO2_DISTANCE INTEGER,
                    PM25 CHARACTER, PM25_DISTANCE INTEGER,
                    PM10 CHARACTER, PM10_DISTANCE INTEGER,
                    WIND CHARACTER, WIND_DISTANCE INTEGER,
                    TEMP CHARACTER, TEMP_DISTANCE INTEGER,
                    PRES CHARACTER, PRES_DISTANCE INTEGER,
                    RH CHARACTER, RH_DISTANCE INTEGER)''')

    cams=get_fields_from_table(cam_cursor,'cameras','id,lat,long',"where country = 'USA'")#
    radi_original=100
    Numero_param=10
    valors=[["??" for j in range(2)] for i in range(Numero_param)]
    query_taules=[get_sites_for_parameter(cursor,n,param_switcher,[4,5]) for n in range(Numero_param)]#diu els sites on hi ha una mostra d'aquell contaminant
    for c in cams:#per cada camara a USA
        get_sites=[]
        get_fields='distinct site,lat,lon'
        radi=radi_original
        need=np.ones((Numero_param), dtype=bool)
        
        while np.sum(need,dtype=bool):#mentre no haguem trobat un site per cada parametre
            bbox=get_bounding_box(c[1],c[2],radi)#potser va fora del while pero no te pinta
            for n in [x for x in range(0,len(need)) if need[x]]:#per cada paramametre que calgui comprovar
                print('Getting cam ' + str(c[0])+ ' parameter number ' + str(n) + ' actual radius= ' + str(radi),end='\r')
                if n == 4 or n ==5:
                    valors[n][0]="??"
                    valors[n][1]="??"
                    need[n]=False
                    continue
                #get_condition="where lat < "+ str(bbox[0]) +" AND lat > " +str(bbox[1])+" AND lon < " +str(bbox[2])+" AND lon > " +str(bbox[3])           
                print('Getting cam ' + str(c[0])+ ' parameter number ' + str(n)+ ' doing query'+ ' actual radius= ' + str(radi),end='\r')

                sites=[q for q in query_taules[n] if in_bounding_box(bbox,q[1],q[2])]#per cada possible site del contaminant mira si esta a la bbox i si ho esta el posa a la llista


                #get_fields_from_table(cursor,parameter_table,get_fields,get_condition)
                #print('Getting cam ' + str(c[0])+ ' parameter number ' + str(n)+ ' query done'+ ' actual radius= ' + str(radi),end='\r')
                if len(sites)!=0:#SI HEM TROBAT SITES EN AQUELL RADI
                    distances=[lat_lon_2_dis(c[1],site[1],c[2],site[2]) for site in sites]#calculem la distancia a cada un dells
                    sorted_indexes=np.argsort(distances)#agafem els index de les distancies ordenades
                    sorted_sites=np.array(sites)[sorted_indexes]#ordenem els sites amb els index
                    valors[n][0]=str(sorted_sites[0][0])#guardem el site que sigui 
                    valors[n][1]=math.floor(distances[sorted_indexes[0]])#amb la seva corresponent distancia
                    need[n]=False
                elif radi>radi_threshold:
                    valors[n][0]="??"
                    valors[n][1]="> " + str(radi_threshold)
                    need[n]=False
            radi+=radi
        valors_1d=np.reshape(valors,Numero_param*2)
        insert_values=[]
        for n in range(0,len(valors_1d)+1):
            if n==0:
                insert_values.append(c[0])
            else:
                insert_values.append(valors_1d[n-1])
        final_values=tuple(insert_values)
        cursor.execute("insert into camera_sites values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",final_values)
        print("Camera: "+'{:>6}'.format(final_values[0])+''.join(['|||'+'{:11}'.format(valors_1d[n])+' '+ '{:>7}'.format(valors_1d[n+1]) for n in range(0,len(valors_1d),2)])+ '||| with maximum radi: '+str(radi))
def update_camera_sites_table(cursor:sqlite3.Cursor,cam_cursor:sqlite3.Cursor):
    radi_threshold=20000#mig perimetre de la terra
    cursor.execute('''CREATE TABLE IF NOT EXISTS camera_sites (camera_id INTEGER PRIMARY KEY, 
                    O2 CHARACTER, O2_DISTANCE INTEGER,
                    SO2 CHARACTER, SO2_DISTANCE INTEGER,
                    CO CHARACTER, CO_DISTANCE INTEGER,
                    NO2 CHARACTER, NO2_DISTANCE INTEGER,
                    PM25 CHARACTER, PM25_DISTANCE INTEGER,
                    PM10 CHARACTER, PM10_DISTANCE INTEGER,
                    WIND CHARACTER, WIND_DISTANCE INTEGER,
                    TEMP CHARACTER, TEMP_DISTANCE INTEGER,
                    PRES CHARACTER, PRES_DISTANCE INTEGER,
                    RH CHARACTER, RH_DISTANCE INTEGER)''')

    cams=get_fields_from_table(cam_cursor,'cameras','id,lat,long',"where country = 'USA'")#
    radi_original=100
    Numero_param=10
    valors=[["??" for j in range(2)] for i in range(Numero_param)]
    query_taules=[get_sites_for_parameter(cursor,n,param_switcher,[0,1,2,3,6,7,8,9]) for n in range(Numero_param)]
    for c in cams:
        get_sites=[]
        get_fields='distinct site,lat,lon'
        radi=radi_original
        need=np.ones((Numero_param), dtype=bool)
        

        while np.sum(need,dtype=bool):#mentre no haguem trobat un site per cada parametre
            bbox=get_bounding_box(c[1],c[2],radi)#potser va fora del while pero no te pinta
            for n in [x for x in range(0,len(need)) if need[x]]:#per cada paramametre que calgui comprovar
                print('Getting cam ' + str(c[0])+ ' parameter number ' + str(n) + ' actual radius= ' + str(radi),end='\r')
                if n != 4 and n !=5:
                    valors[n][0]="??"
                    valors[n][1]="??"
                    need[n]=False
                    continue
                print('Getting cam ' + str(c[0])+ ' parameter number ' + str(n)+ ' doing query'+ ' actual radius= ' + str(radi),end='\r')
                
                sites=[q for q in query_taules[n] if in_bounding_box(bbox,q[1],q[2])]
                #get_fields_from_table(cursor,parameter_table,get_fields,get_condition)
                #print('Getting cam ' + str(c[0])+ ' parameter number ' + str(n)+ ' query done'+ ' actual radius= ' + str(radi),end='\r')
                if len(sites)!=0:#SI HEM TROBAT SITES EN AQUELL RADI
                    distances=[lat_lon_2_dis(c[1],site[1],c[2],site[2]) for site in sites]#calculem la distancia a cada un dells
                    sorted_indexes=np.argsort(distances)#agafem els index de les distancies ordenades
                    sorted_sites=np.array(sites)[sorted_indexes]#ordenem els sites amb els index
                    valors[n][0]=str(sorted_sites[0][0])#guardem el site que sigui 
                    valors[n][1]=math.floor(distances[sorted_indexes[0]])#amb la seva corresponent distancia
                    need[n]=False
                elif radi>radi_threshold:
                    valors[n][0]="??"
                    valors[n][1]="> " + str(radi_threshold)
                    need[n]=False
            radi+=radi
        valors_1d=np.reshape(valors,Numero_param*2)
        insert_values=[]
        for n in range(0,len(valors_1d)+1):
            if n==0:
                insert_values.append(c[0])
            else:
                insert_values.append(valors_1d[n-1])
        final_values=tuple(insert_values)
        cursor.execute("update camera_sites set PM25 = ?, PM25_DISTANCE = ?, PM10 = ?, PM10_DISTANCE = ? WHERE camera_id= ?",(valors[4][0],valors[4][1],valors[5][0],valors[5][1],c[0]))
        print("Camera: "+'{:>6}'.format(final_values[0])+''.join(['|||'+'{:11}'.format(valors_1d[n])+' '+ '{:>7}'.format(valors_1d[n+1]) for n in range(0,len(valors_1d),2)])+ '||| with maximum radi: '+str(radi))

def get_cameras_with_sites_closer_than(cursor:sqlite3.Cursor, distancia_maxima,field):
    table_name='camera_sites'
    if field:
        fields = field
    else:
        fields = 'O2, SO2, CO, NO2, PM25, PM10, WIND, TEMP, PRES, RH'
    condition = 'where O2_DISTANCE< ' + str(distancia_maxima) + ' AND SO2_DISTANCE < ' + str(distancia_maxima) + ' AND CO_DISTANCE < ' + str(distancia_maxima) + ' AND NO2_DISTANCE < ' + str(distancia_maxima) +' AND WIND_DISTANCE < ' + str(distancia_maxima) +' AND TEMP_DISTANCE < ' + str(distancia_maxima) +' AND PRES_DISTANCE < ' + str(distancia_maxima) +' AND RH_DISTANCE < ' + str(distancia_maxima) + ' AND PM25_DISTANCE < ' + str(distancia_maxima) +' AND PM10_DISTANCE < ' + str(distancia_maxima) + " AND O2 != '??' AND SO2 NOT LIKE '??' AND CO !='??' AND NO2 !='??' AND WIND != '??' AND TEMP !='??' AND PRES !='??' AND RH !='??' AND PM25 !='??' AND PM10 !='??'" 
    cameras = get_fields_from_table(cursor,table_name,fields,condition)
    return cameras

def get_data_from_cameras_in_list(cam_ids_list,all_pollution_data,date_string,destiny_path):

    timee=time.time()
    cam_data=filter_cam_data(all_pollution_data,cam_ids_list,date_string)

    #un cop sabem quin dia i hora toca guardar (perque gaurdem la imatge)
    check_folder_and_create(destiny_path,True)
    save_pollution_data_to(cam_data,'2017-01-01 14:00',destiny_path+'/pollution_data') 

    print('Total camera filter time= '+str(time.time()-timee))



def filter_cam_data(all_data,par_ids,date_string):
    #data follows site_id,date,sample_value structure
    final_data=[]
    for i in range(len(all_data)):
        #print('Filtering parameter: ' + str(i))
        filtered_data=[s[2] for s in all_data[i] if check_cam_date(s,par_ids[i],date_string)]
        if [True for f in filtered_data if not (str(f)) ]:
            print(date_string)
            print(filtered_data)
        final_data.append(filtered_data)

    return (final_data)


def check_cam_date(data_sample,site_id,date):
    if data_sample[0] == site_id and date ==data_sample[1]:
        return True
    else:
        return False

def get_all_pollution_data(cursor:sqlite3.Cursor):
    all_pollution_data=[]
    s=time.time()
    #agafar totes dades contaminacio
    for i in range(0,10):
        n=i
        print('Getting data from '+ param_switcher.get(n,'Invalid parameter name'))
        parameter_data=get_fields_from_table(cursor,param_switcher.get(n,'Invalid parameter name'),'site, data, sample','')
        all_pollution_data.append(parameter_data)
        print([len(s) for s in all_pollution_data])
    print('Got all data in: '+ str(time.time()-s)+ ' seconds')
    return all_pollution_data 

    #for i in range(len(all_data)) 

def get_all_pollution_data_for_cam(cursor:sqlite3.Cursor,cam_id):
    all_pollution_data=[]
    s=time.time()
    cam_parameters=get_fields_from_table(cursor,'camera_sites','O2, SO2, CO, NO2, PM25, PM10, WIND, TEMP, PRES, RH',str('where camera_id='+str(cam_id)))[0]
    #print(cam_parameters)
    #agafar totes dades contaminacio
    for i in range(0,10):
        n=i
        print('Getting data from '+ param_switcher.get(n,'Invalid parameter name') + ' for camera nº: '+ '{:0>8}'.format(cam_id))
        get_condition=" where site= '"+str(cam_parameters[i]) + "'"
        parameter_data=get_fields_from_table(cursor,param_switcher.get(n,'Invalid parameter name'),'site, data, sample',str(get_condition))
        all_pollution_data.append(parameter_data)
        #print([len(s) for s in all_pollution_data])
    print('Got all data in: '+ str(time.time()-s)+ ' seconds')
    return all_pollution_data 
def save_pollution_data_to(cam_data,date_string,destiny_folder):
    save_string=[]
    print(cam_data)
    final_string=",".join(map(str, cam_data))
    print(final_string)
    myfile = open(destiny_path+'/aaa.txt',"w")
    myfile.write(final_string)



def get_data_from_camera(cursor:sqlite3.Cursor,site_ids):
    super_data_array=[]

    s=time.time()
    for i in range(0,len(site_ids)):
        n=i
        if n==4 or n==5:
            n+=2
        get_condition = "where site = '" + str(site_ids[i])+"'"
        print(get_condition)
        a=get_fields_from_table(cursor,param_switcher.get(n,'Invalid parameter name'),'data, sample',get_condition)
        #aa=[k for k in a if k[0]=='2017-01-02 14:00']
        super_data_array.append(a)
        print([len(s) for s in super_data_array])
    print(str(time.time()-s))
    #a super data array estan tots els samples per cada un dels parameters
def get_cameras_data_from_date(super_data_array,cam_id,date):
    #data must be in format YYYY-MM-DD hh:mm:ss
    print(cam_id)
    print(camera_sites[0])
    get_condition="where site = '" + camera_sites[0][0]  + "' AND data = '" + date + "'"
    print(get_condition)
    
    start_time=time.time()
    write_results=[]
    '''
    for i in range(10):
        if i==4 or i==5:
            continue
        a=[]
        a=get_fields_from_table(cursor,param_switcher.get(i,'Invalid parameter name'),'sample',get_condition)
        print(a[0][0])
        write_results.append(a[0][0])
    print('Total time = ' + str(time.time()-start_time))
    '''
    
def initialise_db(database_name):
    db_connection = sqlite3.connect(str(database_name+'.db'))
    return (db_connection,db_connection.cursor())
def close_db_cursor(connection):
    connection.commit()



def save_pollution_data_for_cam_images(cam_images_folder,all_pollution_data,cam_parameter_ids):
    par_ids=[cam_parameter_ids[0][i] for i in range(len(cam_parameter_ids[0])) if i!=0]
    cam_id=cam_images_folder.split('/')[-1]
    for f in os.listdir(cam_images_folder):#per cada carpeta on esta guardat tot
        if os.path.isdir(cam_images_folder+'/'+f):
            image_names =  os.listdir(cam_images_folder+'/'+f)
            image_days=list(set([ff.replace(str('_'+ff.split('_')[-1]),'') for ff in image_names if ff.endswith('.jpg')]))
            for image_name in image_days:

                date_string=format_image_name_to_datetime(image_name)
                filtered=filter_cam_data(all_pollution_data,par_ids,date_string)
                final_txt_path=cam_images_folder+'/'+f+'/'+image_name+'_pollution.txt'
                cc=0
                for fff in filtered:
                    if len(fff)==0:
                        print(filter_cam_data(all_pollution_data,par_ids,date_string)) 
                        print(date_string)
                    cc+=1

                #print(filtered,end='')
                #if not os.path.exists(final_txt_path):#si no hem guardat ja les dades
                myfile = open(final_txt_path,"w")
                [myfile.write(str((s))) for s in filtered]
                myfile.close()
                #print("Pollution data saved for camera: " + '{:0>8}'.format(cam_id) + ", date: " + date_string + " saved into: " + final_txt_path)

def save_pollution_data_for_cam(db_cursor,cam_parameter_ids,save_to_folder):
    par_ids=[cam_parameter_ids[i] for i in range(len(cam_parameter_ids)) if i!=0]
    aqi_query_condition="where site LIKE '"+ cam_parameter_ids[1].replace(cam_parameter_ids[1].split('-')[-1],'') + "%'" #que almenys estigui en el mateix condat
    all_aqi_site_data=get_fields_from_table(db_cursor,'poll_aqi','site,data,sample',aqi_query_condition)
    all_pollution_data=get_all_pollution_data_for_cam(db_cursor,cam_parameter_ids[0])
    for aqi_index in all_aqi_site_data:#per cada
        dest_folder= save_to_folder+'/'+str(aqi_index[2])
        check_folder_and_create(dest_folder,1)
        date_string=aqi_index[1]

        datetime_strings = [date_string + ' ' + '{:0>2}'.format(i)+':00' for i in range(24)]
        for date in datetime_strings:
            date_split=date.split('-')
            txt_name=date_split[0]+date_split[1]+date_split[2].split(' ')[0]+'_'+date_split[2].split(' ')[1].replace(':','')+'00_pollution.txt'
            final_txt_path=dest_folder+'/'+ txt_name
            filtered=filter_cam_data(all_pollution_data,par_ids,date)
            myfile = open(final_txt_path,"w")
            [myfile.write(str((s))) for s in filtered]
            myfile.close()
        print("Data for day: " + date_string + " saved")
        '''
        final_txt_path=dest_folder+'/'+image_name+'_pollution.txt'
        cc=0
        for fff in filtered:
            if len(fff)==0:
                print(filter_cam_data(all_pollution_data,par_ids,date_string)) 
                print(date_string)
            cc+=1

        #print(filtered,end='')
        #if not os.path.exists(final_txt_path):#si no hem guardat ja les dades
        myfile = open(final_txt_path,"w")
        [myfile.write(str((s))) for s in filtered]
        myfile.close()
        #print("Pollution data saved for camera: " + '{:0>8}'.format(cam_id) + ", date: " + date_string + " saved into: " + final_txt_path)
'''

def get_day_par_info(all_data,par_ids,date_string,par_num):
    filtered_data=[s for s in all_data[par_num] if check_cam_date(s,par_ids[par_num],date_string)]
    return (filtered_data)


def format_image_name_to_datetime(image_name):
    #datetime output must be of type 2017-03-01 08:00
    year=image_name[:4]
    month=image_name[4:6]
    day=image_name[6:8]
    hour=image_name[9:11]

    datetime_string = str(year) + '-' + str(month) + '-' +str(day) + ' ' + str(hour) +':00'

    return datetime_string
#def download_camera_samples(service, camera_id, year):    
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#pip3 install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


#src_path='/home/u124223/Documents'




'''
a=get_cameras_with_sites_closer_than(db_cursor, 10,'camera_id,O2, SO2, CO, NO2, WIND, TEMP, PRES, RH')
get_data_from_cameras_in_list(db_cursor,a)
'''
#get_cameras_data_from_date(db_cursor,list(a[0]),src_path+'sample_images/data','2017-01-02 14:00')

#print_table(db_cursor,'camera_sites')
#db_cursor.execute("DROP TABLE gas_ozone")

#b=get_fields_from_table(db_cursor,'met_wind','distinct site, lat, lon'," ")
#a[1]=get_fields_from_table(db_cursor,'met_wind','site',"where site = '16-029-0031'")
'''
a=[[]for n in range(10)]
a[1]=get_fields_from_table(db_cursor,'met_wind','distinct site, lat, lon'," ")
#print(b)
print(a)


db_cursor.execute("DROP TABLE gas_ozone")
db_cursor.execute("DROP TABLE gas_so2")
db_cursor.execute("DROP TABLE gas_co")
db_cursor.execute("DROP TABLE gas_no2")

db_cursor.execute("CREATE TABLE IF NOT EXISTS gas_ozone (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , s TINYTEXT, lat FLOAT, lon FLOAT)")
db_cursor.execute("CREATE TABLE IF NOT EXISTS gas_so2 (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)")
db_cursor.execute("CREATE TABLE IF NOT EXISTS gas_co (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)")
db_cursor.execute("CREATE TABLE IF NOT EXISTS gas_no2 (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)")


db_cursor.execute("CREATE TABLE IF NOT EXISTS met_wind (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)")
db_cursor.execute("CREATE TABLE IF NOT EXISTS met_temperature (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)")
db_cursor.execute("CREATE TABLE IF NOT EXISTS met_pressure (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)"),
db_cursor.execute("CREATE TABLE IF NOT EXISTS met_humidity (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample FLOAT(23) , unit_mesurament TINYTEXT, lat FLOAT, lon FLOAT)")
db_cursor.execute("CREATE TABLE IF NOT EXISTS poll_aqi (sample_id INTEGER PRIMARY KEY, site CHARACTER, data datetime, sample INTEGER,quality TINYTEXT, defining_parameter TINYTEXT)")
'''

#db_cursor.execute('''DROP TABLE aqi_index''')
#db_cursor.execute("CREATE TABLE IF NOT EXISTS aqi_index (date CHARACTER PRIMARY KEY, aqi INTEGER , defining_parameter TINYTEXT, site CHARACTER)")

#db_cursor.execute('''CREATE TABLE IF NOT EXISTS monitors (whole_id CHARACTER PRIMARY KEY, monitor_id CHARACTER,parameter_codes TEXT , lat FLOAT, long FLOAT , state TINYTEXT, county TEXT, city TEXT, full_address TEXT, scale_definition TEXT,  FOREIGN KEY (monitor_id) REFERENCES sites(monitor_id))''')
#iterate_csv_monitors('/Volumes/Data_HD/Users/Carles/TFG/codes/aqs_monitors.csv')
#db_cursor.execute('''DROP TABLE sites''')
#db_cursor.execute('''CREATE TABLE IF NOT EXISTS sites (monitor_id CHARACTER PRIMARY KEY, lat FLOAT, long FLOAT , state TINYTEXT, county TEXT, city TEXT, closing_date TEXT, met_stie_type TEXT)''')
#iterate_csv_sites('/home/u124223/Documents/aqs_sites.csv',db_cursor)
#db_cursor.execute('''DROP TABLE camera_sites''')
#create_camera_site_table(db_cursor)

#iterate_aqi_folder(db_cursor,src_path+'/csv')
#parse_aqi(db_cursor,src_path+'/daily_aqi_by_cbsa_2017.csv')



if __name__ == '__main__':
    db_connection,db_cursor=initialise_db('pollution')
    db_connection2,db_cursor2=initialise_db('camera_location')
    

    db_cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    
    noms=db_cursor.fetchall()
    noms2=db_cursor2.fetchall()
    #noms2=db_cursor2.fetchall()
    '''
    print("pollution.db")
    [print(n) for n in noms]
    print()
    print()
    print()
    print()
    '''
    #[print(n) for n in noms2]
    for n in noms2:
        if n!='sqlite_sequence':
            db_cursor2.execute('select * from ' + n[0])
            column_names = list(map(lambda x: x[0], db_cursor2.description))
            print(n)
            print(column_names)
            print()
    for n in noms:
        if n!='sqlite_sequence':
            db_cursor.execute('select * from ' + n[0])
            column_names = list(map(lambda x: x[0], db_cursor.description))
            print(n)
            print(column_names)
            print()
    # db_cursor2.execute('select * from met_wind')
    # column_names2 = list(map(lambda x: x[0], db_cursor2.description))
    # db_cursor.execute("select * from met_wind where data = '2017-01-10 10:00' and site= '01-073-0023'")
    
    # print("___")
    # rr=db_cursor.fetchall()
    # print()
    # [print(r) for r in rr]
    # print("camera_location.db")
    # print(column_names2)
    '''
    db_cursor.execute("select * from gas_ozone where data = '2017-01-10 10:00'")
    column_names = list(map(lambda x: x[0], db_cursor.description))
    #db_cursor2.execute('select * from cameras ')
    #column_names2 = list(map(lambda x: x[0], db_cursor2.description))
    nn=db_cursor.fetchall()
    #nn2=db_cursor2.fetchall()
    print(nn)
    print(nn2[0])
    print(rr[0])
    print(column_names)
    print(column_names2)
    db_cursor.execute('select COUNT(*) from camera_sites')
    db_cursor2.execute('select COUNT(*) from camera_sites')
    nn=db_cursor.fetchall()
    nn2=db_cursor2.fetchall()
    print(nn)
    print(nn2)
    #nn=db_connection2.fetchall()
    #print(nn[0])
    #[print(n) for n in noms]
    # print()
    # print()
    # print()
    # print()
    # [print(db_cursor2.execute('select * from ' + n[0]).fetchall()[0]) for n in noms2]
    # #rr = db_cursor2.fetchall()
    # [print(r) for r in rr]
    '''
    
    close_db_cursor(db_connection)

















