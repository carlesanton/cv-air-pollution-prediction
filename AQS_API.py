from datetime import datetime
import time
import requests
import sqlite3
import xlrd
import csv
import pandas as panda

class camara:
    def __init__(self, id, lat, long, number, street, city, district,county,state, country, full_address):
        self.id = id
        self.lat = lat
        self.long = long
        self.number = number
        self.street = street
        self.city = city
        self.district = district
        self.county = county
        self.state = state
        self.country = country
        self.full_address = full_address
    def __str__(self):
        return str(self.id) +'||'+ str(self.lat) +', '+ str(self.long)+'||' + str(self.number)+', ' + self.street+', ' + self.city+', ' + self.district+', ' + self.county+', ' + self.state+', ' + self.country
        
    def add_to_database(self, cursor: sqlite3.Cursor):
        cursor.execute('SELECT * FROM cameras where ID = ?', (int(self.id),))
        objectes = cursor.fetchall()
        #mira si no hi ha cap objete amb aquella ID, si no hi ha el posa
        if(len(objectes)==0):
            cursor.execute("insert into cameras values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (self.id, self.lat, self.long, self.country,self.state,self.county ,self.district ,self.city ,self.street ,self.number ,self.full_address))
    def get_closer_monitor(self):
        cursor.execute('SELECT * FROM monitors where CITY = ?', self.city)
        objectes = cursor.fetchall()
        if(len(objectes)==0):#si no hi ha cap en aquell poble
            cursor.execute('SELECT * FROM monitors where DISTRICT = ?', self.district)
            objectes = cursor.fetchall()
            if (len(objectes)==0):
                cursor.execute('SELECT * FROM monitors where COUNTY = ?', self.county)
                objectes = cursor.fetchall()
                if (len(objectes)==0):
                    cursor.execute('SELECT * FROM monitors where STATE = ?', self.state)
                    objectes = cursor.fetchall()
                    if (len(objectes)==0):
                        cursor.execute('SELECT * FROM monitors where COUNTRY = ?', self.country)
                        objectes = cursor.fetchall()
                        if (len(objectes)!=0):
                            self.closer_monitor = cursor[0][0]
                    else:
                        self.closer_monitor = cursor[0][0]
                else:
                    self.closer_monitor = cursor[0][0]
            else:
                self.closer_monitor = cursor[0][0]
        else:
            self.closer_monitor = cursor[0][0]
            
            
#monitor functions
def get_usa_monitor_data():
	#rawData query parameters
	raw_parameters = {'user' : API_USER,'pw' : API_PASSWORD,'format':'DMCSV','param' : '44201', 'bdate' : '20110501', 'edate' : '20110501', 'state' : '37', 'county' : '063'};  
	r=requests.get('https://aqs.epa.gov/api/rawData?',params=raw_parameters)
	print(r.headers)
def get_usa_monitors_list(begin_date,finish_date):
	'''
	bdate (Begin Date): required (string)
	The begin date of data in YYYYMMDD format. The system will use full years when extracting data (the month and day will be ignored).
	example: 20141231

	
	csa (CSA Code): (string - minLength: 3 - maxLength: 3)
	The 3 digit (leading zeroes required) Census code for the Consolidated Statistical Area you would like to filter the query results on.
	example: 290
	
	cbsa (CBSA Code): (string - minLength: 5 - maxLength: 5)
	The 5 digit (leading zeroes required) Census code for the Core Based Statistical Area you would like to filter the query results on.
	example:26620
	
	site (Site Number): (string - minLength: 4 - maxLength: 4)
	The 4 digit (leading zeroes required) AQS ID number for the site within the county that you would like to filter the query results on.
	example:12
	
	county (County Code): (string - minLength: 3 - maxLength: 3)
	The 3 digit (leading zeroes required) FIPS code for the county you would like to filter the query results on.
	example:089
	
	state (State Code): (string - minLength: 2 - maxLength: 2)
	The 2 digit (leading zeroes required) FIPS code for the state you would like to filter the query results on.
	example:01
	
	minlat
	maxlat
	minlon
	maxlon
	'''
	#profile parameters
	prof_parameters = {'user' : API_USER,'pw' : API_PASSWORD,'formar':'CSV', 'bdate':begin_date[2]+begin_date[1]+begin_date[0], 'edate':finish_date[2]+finish_date[1]+finish_date[0]}
	pp={'user':API_USER,'pw':API_PASSWORD,'format':'CSV','param':'44201','bdate':'2010','edate':'2015','state':'37','county':'063'}
	r=requests.get('https://aqs.epa.gov/api/profile?',params=prof_parameters)

	print(r.headers)

#get cameras functions	            
def coordinates_to_camera(id,lat, long,radi):
    parametres = {'app_id' : '4XvFG0rmkdyODpDpjaew','app_code' : 'K-2J4tZ8pDd1uKMJRTMAag','mode':'retrieveAddresses','prox' : str(lat) + ','+ str(long) + ',' + str(radi)};#  
    r=requests.get('https://reverse.geocoder.api.here.com/6.2/reversegeocode.json',params=parametres)
    data = r.json()
    if (len(data['Response']['View']) != 0):
        if ('HouseNumber' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            number = data['Response']['View'][0]['Result'][0]['Location']['Address']['HouseNumber']
        else :
            number = 'empty'
        
        if ('Street' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            street = data['Response']['View'][0]['Result'][0]['Location']['Address']['Street']
        else :
            street = 'empty'

        if ('City' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            city = data['Response']['View'][0]['Result'][0]['Location']['Address']['City']
        else :
            city = 'empty'
    
        if ('District' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            district = data['Response']['View'][0]['Result'][0]['Location']['Address']['District']
        else :
            district = 'empty'
    
        if ('County' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            county = data['Response']['View'][0]['Result'][0]['Location']['Address']['County']
        else :
            county = 'empty'
    
        if ('State' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            state = data['Response']['View'][0]['Result'][0]['Location']['Address']['State']
        else :
            state = 'empty'
    
        if ('Country' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            country = data['Response']['View'][0]['Result'][0]['Location']['Address']['Country']
        else :
            country = 'empty'

        if ('Label' in data['Response']['View'][0]['Result'][0]['Location']['Address']):
            full_address = data['Response']['View'][0]['Result'][0]['Location']['Address']['Label']
        else :
            full_address = 'empty'
    else:
        number = street = city = district = county=state = country = full_address = 'empty'

    return camara( id, lat, long, number, street, city, district,county,state, country, full_address)
def iterate_excel_cameras(path):
    book = xlrd.open_workbook(path)
    #first sheet
    first_sheet = book.sheet_by_index(0)
    for i in range(1, first_sheet.nrows):   
        support_values = first_sheet.row_values(i)
        print(int(support_values[0]))
        camara_support = coordinates_to_camera(support_values[0],support_values[1],support_values[2],100)
        camara_support.add_to_database(db_cursor)
def get_camera_list(imprimir: bool):
    camera_list=[]
    with db_connection:
        db_cursor.execute("SELECT * FROM cameras")
        totes_cameres=db_cursor.fetchall()
        for cam in totes_cameres:
            support_cam=camara( cam[0], cam[1], cam[2], cam[9], cam[8], cam[7], cam[6],cam[5],cam[4], cam[3], cam[10])
            camera_list.append(support_cam)
            if imprimir:
                print(camera_list[-1])
    return camera_list

#get table            
def get_table(table):
    with db_connection:
        db_cursor.execute("SELECT * FROM "+table)
        totes_cameres=db_cursor.fetchall()
        for cam in totes_cameres:
            print(cam)


#######################################################################################################

#API USER AND PASSWORD
# user=carles.anton01@estudiant.upf.edu&pw=mauvemouse32
API_USER = 'carles.anton01@estudiant.upf.edu'
API_PASSWORD = 'mauvemouse32'

# Create database and table

db_connection = sqlite3.connect('camera_location.db')
db_cursor = db_connection.cursor()
# db_cursor.execute('''CREATE TABLE IF NOT EXISTS monitors (id INTEGER PRIMARY KEY, lat FLOAT, long FLOAT,  country TINYTEXT , state TINYTEXT, county TEXT, district TEXT, city TEXT, street TEXT, number INTEGER, full_address TEXT)''')
'''
begin_date=['01','10','2015']
finish_date=['01','10','2016']
get_monitors_list(begin_date[2]+begin_date[1]+begin_date[0],finish_date[2]+finish_date[1]+finish_date[0])

#CSV to SQL
# load data
df = panda.read_csv('FL_insurance_sample.csv')
# strip whitespace from headers
df.columns = df.columns.str.strip()
# drop data into database
df.to_sql("MyTable", db_connection,if_exists='replace')
get_table("MyTable")
'''
#get_usa_monitor_data()
get_usa_monitors_list(['01','10','2015'],['01','10','2016'])

db_connection.close()





