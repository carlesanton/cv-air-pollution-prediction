import csv
import http.client
import io
import json
import math
import os
import pickle
import re
import sqlite3
import sys
import time
import zipfile
from datetime import datetime
from math import atan2
from math import cos
from math import radians
from math import sin
from math import sqrt
from shutil import copyfile

import numpy as np
import requests
from PIL import Image
##IMAGES
# per parsejar
###
# per agrupar cada 2
# utilities
# google drive
###


# If modifying these scopes, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/drive"]


class camara:
    def __init__(
        self,
        id,
        lat,
        long,
        number,
        street,
        city,
        district,
        county,
        state,
        country,
        full_address,
    ):
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

    def add_to_database(self, cursor: sqlite3.Cursor):
        cursor.execute("SELECT * FROM cameras where ID = ?", (int(self.id),))
        objectes = cursor.fetchall()
        # mira si no hi ha cap objete amb aquella ID, si no hi ha el posa
        if len(objectes) == 0:
            cursor.execute(
                "insert into cameras values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.id,
                    self.lat,
                    self.long,
                    self.country,
                    self.state,
                    self.county,
                    self.district,
                    self.city,
                    self.street,
                    self.number,
                    self.full_address,
                ),
            )

    def __str__(self):
        return (
            "ID: "
            + str(self.id)
            + "    LAT: "
            + str(self.lat)
            + "    LON: "
            + str(self.long)
            + "    "
            + self.full_address
        )


class monitor:
    def __init__(
        self,
        id,
        lat,
        long,
        state,
        county,
        city,
        full_address,
        parameter_code,
        scale_definition,
    ):
        self.id = id
        self.lat = lat
        self.long = long
        self.city = city
        self.county = county
        self.state = state
        self.full_address = full_address
        self.parameter_code = parameter_code
        self.scale_definition = scale_definition

    def add_to_database(self, cursor: sqlite3.Cursor):
        monitor_id = self.id.replace("-" + str(self.parameter_code), "")
        cursor.execute("SELECT * FROM monitors where whole_id = ?", (self.id,))
        objectes = cursor.fetchall()
        # mira si no hi ha cap objete amb aquella ID, si no hi ha el posa
        if len(objectes) == 0:
            cursor.execute(
                "insert into monitors values (?,?,?,?,?,?,?,?,?,?)",
                (
                    self.id,
                    monitor_id,
                    self.parameter_code,
                    self.lat,
                    self.long,
                    self.state,
                    self.county,
                    self.city,
                    self.full_address,
                    self.scale_definition,
                ),
            )
        """
        else:
            cursor.execute('SELECT parameter_codes FROM monitors where monitor_id = ?', (int(self.id),))
            codis_actuals = cursor.fetchall()
            cursor.execute('UPDATE monitors SET parameter_codes = ? where monitor_id = ?', ("".join(map(str, codis_actuals[0]))+' '+self.parameter_code,int(self.id),))
        """


class site:
    def __init__(self, id, lat, lon):
        self.id = id
        self.lat = lat
        self.long = lon

    def __str__(self):
        return (
            "site ID: "
            + str(self.id)
            + "    LAT: "
            + str(self.lat)
            + "    LON: "
            + str(self.long)
        )

    def __repr__(self):
        return str(self.id)


def coordinates_to_camera(id, lat, long, radi):
    parametres = {
        "app_id": "4XvFG0rmkdyODpDpjaew",
        "app_code": "K-2J4tZ8pDd1uKMJRTMAag",
        "mode": "retrieveAddresses",
        "prox": str(lat) + "," + str(long) + "," + str(radi),
    }
    #
    r = requests.get(
        "https://reverse.geocoder.api.here.com/6.2/reversegeocode.json",
        params=parametres,
    )
    data = r.json()
    if len(data["Response"]["View"]) != 0:
        if (
            "HouseNumber"
            in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]
        ):
            number = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "HouseNumber"
            ]
        else:
            number = "empty"

        if "Street" in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]:
            street = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "Street"
            ]
        else:
            street = "empty"

        if "City" in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]:
            city = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "City"
            ]
        else:
            city = "empty"

        if (
            "District"
            in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]
        ):
            district = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "District"
            ]
        else:
            district = "empty"

        if "County" in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]:
            county = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "County"
            ]
        else:
            county = "empty"

        if "State" in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]:
            state = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "State"
            ]
        else:
            state = "empty"

        if "Country" in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]:
            country = data["Response"]["View"][0]["Result"][0]["Location"]["Address"][
                "Country"
            ]
        else:
            country = "empty"

        if "Label" in data["Response"]["View"][0]["Result"][0]["Location"]["Address"]:
            full_address = data["Response"]["View"][0]["Result"][0]["Location"][
                "Address"
            ]["Label"]
        else:
            full_address = "empty"
    else:
        number = (
            street
        ) = city = district = county = state = country = full_address = "empty"

    return camara(
        id,
        lat,
        long,
        number,
        street,
        city,
        district,
        county,
        state,
        country,
        full_address,
    )


def get_cameras(cursor: sqlite3.Cursor, print_bool=False):
    cams = []
    with db_connection:
        db_cursor.execute("SELECT * FROM cameras")
        totes_cameres = db_cursor.fetchall()
        count = 0
        for cam in totes_cameres:
            ccc = camara(
                cam[0],
                cam[1],
                cam[2],
                cam[9],
                cam[8],
                cam[7],
                cam[6],
                cam[5],
                cam[4],
                cam[3],
                cam[10],
            )
            cams.append(ccc)
            if print_bool:
                print(ccc)
            count += 1

    return cams


def get_cameras_condition(condition, cursor: sqlite3.Cursor):
    cams = []
    cursor.execute("SELECT * FROM cameras " + condition)
    totes_cameres = cursor.fetchall()
    count = 0
    for cam in totes_cameres:
        cams.append(
            camara(
                cam[0],
                cam[1],
                cam[2],
                cam[9],
                cam[8],
                cam[7],
                cam[6],
                cam[5],
                cam[4],
                cam[3],
                cam[10],
            )
        )
        count += 1
    return cams


def iterate_csv_monitors(path):
    reader = csv.reader(open(path), delimiter=",")
    count = 0
    for row in reader:
        if count != 0:
            lat = 0
            lon = 0
            if len(row[6]) != 0:
                lat = float(row[6])
            if len(row[7]) != 0:
                lon = float(row[7])
            final_id = (
                row[0] + "-" + row[1] + "-" + row[2].replace('"', "") + "-" + row[3]
            )
            suport_monitor = monitor(
                final_id, lat, lon, row[26], row[27], row[28], row[25], row[3], row[21]
            )
            suport_monitor.add_to_database(db_cursor)

            if count % 100 == 0:
                print(count)

        count += 1


def iterate_csv_sites(path, cursor: sqlite3.Cursor):
    reader = csv.reader(open(path), delimiter=",")
    count = 0
    for row in reader:
        if count != 0:
            lat = 0
            lon = 0
            if len(row[3]) != 0:
                lat = float(row[3])
            if len(row[4]) != 0:
                lon = float(row[4])
            final_id = row[0] + "-" + row[1] + "-" + row[2].replace('"', "")

            cursor.execute(
                "insert into sites values (?,?,?,?,?,?,?,?)",
                (final_id, lat, lon, row[22], row[23], row[24], row[10], row[14]),
            )

            if count % 100 == 0:
                print(count)

        count += 1


def lat_lon_2_dis(lat1, lat2, lon1, lon2):
    # approximate radius of earth in km
    R = 6373.0

    dlon = (lon2 * math.pi / 180) - lon1 * math.pi / 180
    dlat = lat2 * math.pi / 180 - lat1 * math.pi / 180

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    if a > 0:
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
    else:  # perque sino peta
        c = 1000
    distance = R * c
    return distance


def get_monitors_condition(condition, cursor: sqlite3.Cursor):
    monit = []
    cursor.execute("SELECT * FROM monitors " + condition)
    tots_monitors = db_cursor.fetchall()
    count = 0
    for mon in tots_monitors:
        monit.append(
            monitor(
                mon[1], mon[3], mon[4], mon[5], mon[6], mon[7], mon[8], mon[2], mon[9]
            )
        )
        # print(count)
        count += 1
    return monit


def get_sites_condition(fields, condition, cursor: sqlite3.Cursor):
    sit = []
    cursor.execute("SELECT " + fields + " FROM sites " + condition)
    tots = db_cursor.fetchall()
    for s in tots:
        sit.append(site(s[0], s[1], s[2]))
    return sit


def get_bounding_box(lat, lon, radi):
    # approximate radius of earth in km
    km_grau_lat = 0
    km_grau_lon = 0
    if abs(lat) < 40:
        km_grau_lat = 110.57
        km_grau_lon = 111.32
    elif abs(lat) < 80:
        km_grau_lat = 111.03
        km_grau_lon = 85.39
    elif abs(lat) > 80 and abs(lat) < 90:
        km_grau_lat = 111.69
        km_grau_lon = 19.39

    d_lat = radi / km_grau_lat
    d_lon = radi / km_grau_lon

    return (lat + d_lat, lat - d_lat, lon + d_lon, lon - d_lon)


def create_camera_site_table(cursor: sqlite3.Cursor):

    db_cursor.execute(
        """CREATE TABLE IF NOT EXISTS camera_sites (camera_id INTEGER PRIMARY KEY, whole_site_id CHARACTER,distance_km INTEGER)"""
    )
    cams = get_cameras_condition("where country = 'USA'", cursor)
    radi_original = 100
    get_sites = []
    print(len(cams))
    for c in cams:
        get_sites = []
        get_fields = "monitor_id,lat,long"
        radi = radi_original
        while len(get_sites) == 0:
            bbox = get_bounding_box(c.lat, c.long, radi)
            get_condition = (
                "where closing_date = '' AND lat < "
                + str(bbox[0])
                + " AND lat > "
                + str(bbox[1])
                + " AND long < "
                + str(bbox[2])
                + " AND long > "
                + str(bbox[3])
                + " ORDER BY lat DESC"
            )
            get_sites = get_sites_condition(get_fields, get_condition, cursor)
            radi += radi_original
            """
            if len(get_sites)!=0:
                print (str(radi) + ": " + str(len(get_sites)))
            """
        distances = []
        for site in get_sites:
            # print(str(c.lat)+" "+ str(site.lat)+" "+ str(c.long)+" "+ str(site.long))
            distances.append(lat_lon_2_dis(c.lat, site.lat, c.long, site.long))

        sorted_indexes = np.argsort(distances)
        sorted_sites = np.array(get_sites)[sorted_indexes]
        # print (str(c.id) +" "+ str(sorted_sites[0].id)+ " "+str(math.floor(distances[sorted_indexes[0]])))
        cursor.execute(
            "insert into camera_sites values (?,?,?)",
            (c.id, sorted_sites[0].id, math.floor(distances[sorted_indexes[0]])),
        )
        print(
            "Camera: "
            + str(c.id)
            + " paired with site: "
            + str(sorted_sites[0].id)
            + " at distance "
            + str(distances[sorted_indexes[0]])
        )


def get_cameras_with_sites(cursor: sqlite3.Cursor):
    relations = []
    cursor.execute("SELECT * FROM camera_sites")
    totes = db_cursor.fetchall()
    count = 0
    for r in totes:
        relations.append([r[0], r[1], r[2]])
        # print([r[0],r[1],r[2]])
    return relations


def get_usa_monitor_data(
    format, param, bdate, edate, state, county, site, destiny_path
):
    # rawData query parameters

    # raw_parameters = {'user' : API_USER,'pw' : API_PASSWORD,'format':'DMCSV','param' : '44201', 'bdate' : '20110501', 'edate' : '20110501', 'state' : '37', 'county' : '063'};
    raw_parameters = {
        "user": API_USER,
        "pw": API_PASSWORD,
        "format": format,
        "param": param,
        "bdate": bdate,
        "edate": edate,
        "state": state,
        "county": county,
        "site": site,
    }
    with requests.Session() as s:
        download = s.get("https://aqs.epa.gov/api/rawData?", params=raw_parameters)
        decoded_content = download.content.decode("utf-8")
        cr = csv.reader(decoded_content.splitlines(), delimiter=",")
        anymes = "nosesae"
        camera_id_string = destiny_path.split("/")[-1].zfill(5)
        myfile = open(destiny_path + "/aaa.txt", "w")
        count = 0
        for row in cr:
            if (
                not cr.line_num == 1 and row[0] != "END OF FILE"
            ):  # si no es la primera ni esta al final
                data = str(row[12]).split("-")  # agafar la data
                anymes = data[0] + data[1]  # posarho en YYYYMM format
                file_name = (
                    camera_id_string + "_" + anymes + "_" + str(param) + ".txt"
                )  # idcam_anymes_param
                final_file_path = (
                    destiny_path + "/" + file_name
                )  # src/idcam/idcam_anymes_param
                if not os.path.exists(final_file_path):  # si no existeix
                    myfile = open(final_file_path, "w")
                    myfile.write(
                        str(row[7])
                        + "    "
                        + str(row[9])
                        + " ("
                        + str(row[17])
                        + ")"
                        + "\n"
                    )  # linea de info
                    myfile.write(
                        str(data[2] + "/" + row[13]) + "    " + str(row[16]) + "\n"
                    )
                elif os.path.exists(final_file_path):  # si existeix
                    myfile = open(final_file_path, "a")
                    myfile.write(
                        str(data[2] + "/" + row[13]) + "    " + str(row[16]) + "\n"
                    )
                myfile.close()
        print(" " + str(cr.line_num - 2) + " lines read")


def get_site_data(
    site_id, data_inici, data_final, cursor: sqlite3.Cursor, destiny_path
):
    check_folder_and_create(destiny_path, 1)
    monitors_lloc = get_monitors_condition(
        "where monitor_id = '" + site_id + "'", cursor
    )
    cc = 1
    for m in monitors_lloc:
        site_number = m.id.split("-")[2]
        print(
            "Downloading data from site "
            + str(site_id)
            + ": monitor "
            + str(cc)
            + "/"
            + str(len(monitors_lloc))
            + ", parameter: "
            + str(m.parameter_code),
            end="",
        )
        sys.stdout.flush()
        get_usa_monitor_data(
            "DMCSV",
            m.parameter_code,
            data_inici,
            data_final,
            m.id.split("-")[0],
            m.id.split("-")[1],
            site_number,
            destiny_path,
        )
        cc += 1


def get_sites_list_data(
    relations_lists, data_inici, data_final, cursor: sqlite3.Cursor
):
    for s in relations_lists:
        get_site_data(
            s[1],
            data_inici,
            data_final,
            cursor,
            str(src_path + "/data/" + str(s[0]).zfill(8)),
        )


# GET ALL DATA
def get_all_data(cursor: sqlite3.Cursor, inici):
    relations = get_cameras_with_sites(cursor)
    end_year = str(int(inici) + 1)
    id = []
    id.append("30001".zfill(8))
    # id.append('20001'.zfill(8))
    r = np.array(relations)
    # get_all_images_in_list(r,inici)
    cam_list = r[6968:, 0]
    get_sample_image_in_list(cam_list, inici)
    # get_sites_list_data(relations,data_inici,data_final,cursor: sqlite3.Cursor)
    # get_sites_list_data(relations_lists,data_inici,data_final,cursor: sqlite3.Cursor)
    # for r in relations:
    # destiny_path=str(src_path+'/data/'+str(r[0]).zfill(8))
    # get_site_data(r[1], inici,final,cursor,destiny_path+'/pollution_data')


# check if folder exists and creates it
def check_folder_and_create(path_to_folder, print_b):
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
        if print_b:
            print("Folder " + path_to_folder + " created")
    return os.path.exists(path_to_folder)


# save a csv AQI year file into the db
def parse_aqi(cursor: sqlite3.Cursor, path):
    # https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_cbsa_2017.zip
    paraules = path.split("_")
    count = 0
    reader = csv.reader(open(path), delimiter=",")
    for row in reader:
        # print((row[6]+'/'+row[2]))
        if reader.line_num != 1:

            cursor.execute(
                "insert into aqi_index values (?,?,?,?)",
                ((row[6] + "/" + row[2]), row[3], row[5], row[6]),
            )
            if count % 100 == 0:
                print(count)
        count += 1


# save AQI indices into db
def iterate_aqi_folder(cursor: sqlite3.Cursor, path):
    for f in os.listdir(path):
        nom_fitxer = str(f)
        print("Geting AQI index from file: " + nom_fitxer)
        parse_aqi(cursor, path + "/" + nom_fitxer)


# parse meteorological data
def parse_rh_dewpoint(cursor: sqlite3.Cursor, path):
    # https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_cbsa_2017.zip
    paraules = path.split("_")
    count = 0
    reader = csv.reader(open(path), delimiter=",")
    for row in reader:
        # print((row[6]+'/'+row[2]))
        if reader.line_num != 1:

            cursor.execute(
                "insert into wind_speed values (?,?,?,?)",
                ((row[6] + "/" + row[2]), row[3], row[5], row[6]),
            )
            if count % 100 == 0:
                print(count)
        count += 1


# GOOGLE DRIVE FUNCTIONS
# get credntials
def get_google_drive_credentials(credential, credentials_file_name):
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            credential = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not credential or not credential.valid:
        if credential and credential.expired and credential.refresh_token:
            credential.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file_name, SCOPES
            )
            credential = flow.run_local_server()
        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(credential, token)
    return credential


# download zip file
def download_google_drive_zip(service, item, destiny_path):
    request = service.files().get_media(fileId=item["id"])
    nom_original = item["name"]
    # final_file_name=nom_original+'_downloaded'+'.zip'
    fh = io.FileIO((destiny_path + "/" + nom_original), "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    tried = 0
    while done is False and tried == 0:
        try:
            status, done = downloader.next_chunk()
            print("Downloading " + nom_original + "  " + str(status.progress() * 100))
        except:
            print("HTTP Exception")
            tried = 1
    return str(fh.name)


# get folders with folder_id as parent folder
def get_drive_folders_in_id(service, folder_id):
    query = (
        "mimeType = 'application/vnd.google-apps.folder'  and '"
        + folder_id
        + "' "
        + "in parents"
    )
    results = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)")
        .execute()
    )
    mini_items = results.get("files", [])
    return mini_items


# get files with folder_id as parent folder
def get_drive_files_in_id(service, folder_id):
    query = "'" + folder_id + "' " + "in parents"
    try:
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name, mimeType)")
            .execute()
        )
        mini_items = results.get("files", [])
    except:
        mini_items = []
    return mini_items


# get folders with target_folder_name as name folder_id as parent folder
def get_drive_folders_in_id_with_name(service, folder_id, target_folder_name):
    query = (
        "mimeType = 'application/vnd.google-apps.folder'  and '"
        + folder_id
        + "' "
        + "in parents and name = "
        + "'"
        + target_folder_name
        + "'"
    )
    try:
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        mini_items = results.get("files", [])
    except:
        mini_items = []
    return mini_items


# get zip files ID in year/cam_id folder
def get_cam_year_files_list(service, camera_id, year, images_folder_id):
    desired_year_folder = get_drive_folders_in_id_with_name(
        service, images_folder_id, str(year)
    )
    cam_id = re.findall("..", str(camera_id))
    zip_files_list = []
    if desired_year_folder:

        half_id_folder = get_drive_folders_in_id_with_name(
            service, desired_year_folder[0]["id"], cam_id[-1]
        )
        if half_id_folder:
            next_half = get_drive_folders_in_id_with_name(
                service, half_id_folder[0]["id"], str(cam_id[-2] + cam_id[-1])
            )
            if next_half:
                final_folder_id = get_drive_folders_in_id_with_name(
                    service, next_half[0]["id"], camera_id
                )
                if final_folder_id:
                    zip_files_list = get_drive_files_in_id(
                        service, final_folder_id[0]["id"]
                    )
                else:
                    print("No " + camera_id + " folder found")
            else:
                print("No " + str(cam_id[-2] + cam_id[-1]) + " folder found")
        else:
            print("No " + cam_id[-1] + " folder found")

    else:
        print("No " + str(year) + " folder found")

    return zip_files_list


# get zip files from year/cam_id, unzip them
def get_all_images_in_list(camera_list, year):
    # drive parameters to create service
    creds = None
    creds_name = "credentials.json"
    creds = get_google_drive_credentials(creds, creds_name)
    service = build("drive", "v3", credentials=creds)

    # comencem per la carpeta src
    results = (
        service.files()
        .list(
            q="mimeType = 'application/vnd.google-apps.folder'  and name = 'AMOS'",
            spaces="drive",
            fields="files(id, name)",
        )
        .execute()
    )

    amos_folder = results.get("files", [])
    image_folder = get_drive_folders_in_id_with_name(
        service, amos_folder[0]["id"], "images"
    )
    for camera in camera_list:
        # referesh creds to avoid timeout
        creds = get_google_drive_credentials(creds, creds_name)
        service = build("drive", "v3", credentials=creds)
        c_id = camera.zfill(8)
        zip_files_list = get_cam_year_files_list(
            service, c_id, year, image_folder[0]["id"]
        )
        if zip_files_list:
            # crear la carpeta per la info de la camara
            drive_destiny_path = src_path + "/data/" + c_id
            check_folder_and_create(drive_destiny_path, 1)
            for z in zip_files_list:
                # baixar zip
                try:
                    file_path = download_google_drive_zip(
                        service, z, drive_destiny_path
                    )
                    final_folder = file_path.replace(".zip", "")
                except:
                    print("No s'ha pogut baixar el zip: " + z["name"])
                    if not os.path.exists(
                        src_path + "/drive_files/" + "download_errors.txt"
                    ):
                        myfile = open(
                            src_path + "/drive_files/" + "download_errors.txt", "w"
                        )
                    else:
                        myfile = open(
                            src_path + "/drive_files/" + "download_errors.txt", "a"
                        )
                    myfile.write(z["name"])
                    myfile.close()
                else:
                    check_folder_and_create(final_folder, 1)
                    unzip_file_to(file_path, final_folder, 1)
        else:
            print("No zip files found in folder " + cam_id)


# unzip zip file
def unzip_file_to(zip_path, destiny_path, delete):
    try:
        zip_ref = zipfile.ZipFile(zip_path, "r")
        zip_ref.extractall(destiny_path)
        zip_ref.close()
        if delete:
            os.remove(zip_path)
        return 1
    except zipfile.BadZipFile:
        print("BadZipFile exception with folder: " + zip_path)
        return 0


# get just one zip file for camera and put one image from it into the sample folder
def get_sample_image_in_list(camera_list, year):
    # drive parameters to create service
    creds = None
    creds_name = "credentials.json"
    creds = get_google_drive_credentials(creds, creds_name)
    service = build("drive", "v3", credentials=creds)
    sample_folder = src_path + "/sample_images"
    check_folder_and_create(sample_folder, 1)
    # comencem per la carpeta src
    results = (
        service.files()
        .list(
            q="mimeType = 'application/vnd.google-apps.folder'  and name = 'AMOS'",
            spaces="drive",
            fields="files(id, name)",
        )
        .execute()
    )

    amos_folder = results.get("files", [])
    image_folder = get_drive_folders_in_id_with_name(
        service, amos_folder[0]["id"], "images"
    )

    for camera in camera_list:
        # referesh creds to avoid timeout
        creds = get_google_drive_credentials(creds, creds_name)
        service = build("drive", "v3", credentials=creds)
        c_id = camera.zfill(8)
        print("Geting sample from: " + c_id)
        zip_files_list = get_cam_year_files_list(
            service, c_id, year, image_folder[0]["id"]
        )
        if zip_files_list:
            # crear la carpeta per guardar les imatges provisionals
            drive_destiny_path = src_path + "/drive_files/" + c_id
            check_folder_and_create(drive_destiny_path, 1)
            z = zip_files_list[0]
            # baixar zip
            try:
                file_path = download_google_drive_zip(service, z, drive_destiny_path)
                final_folder = file_path.replace(".zip", "")
            except:
                print("No s'ha pogut baixar el zip: " + z["name"])
                if not os.path.exists(
                    src_path + "/drive_files/" + "download_errors.txt"
                ):
                    myfile = open(
                        src_path + "/drive_files/" + "download_errors.txt", "w"
                    )
                else:
                    myfile = open(
                        src_path + "/drive_files/" + "download_errors.txt", "a"
                    )
                myfile.write(z["name"])
                myfile.close()
            else:
                # descomprimir zip
                check_folder_and_create(final_folder, 0)
                if unzip_file_to(file_path, final_folder, 1):
                    # copiar imatge a la carpeta de samples amb el nom de la id de la camara
                    imatges_a_la_carpeta = os.listdir(final_folder)
                    fotos_a_les_15 = [
                        f
                        for f in imatges_a_la_carpeta
                        if (re.findall("..", str(f.split("_")[1]))[0] == str(15))
                    ]  # if(re.findall('..', str(f.split('_')[1]))[0]==str(15))
                    if len(fotos_a_les_15) == 0:  # si no i ha fotos a les 15
                        imatge_sample = imatges_a_la_carpeta[0]
                    else:
                        imatge_sample = fotos_a_les_15[0]
                    # imatge_sample=imatges_a_la_carpeta[math.floor(len(imatges_a_la_carpeta)/2)]
                    copy_image_to_folder(
                        final_folder + "/" + imatge_sample,
                        sample_folder,
                        (year + "_" + c_id),
                    )
                    # eliminar carpeta amb totes les imatges
                    eliminar_carpeta(final_folder)
        else:
            print("No zip files found in folder " + c_id)


def copy_image_to_folder(image_path, destiny_folder, new_name):
    img = np.asarray(Image.open(image_path))
    format_imatge = image_path.split(".")[-1]
    try:
        result_image = Image.fromarray((img).astype(np.uint8))
    except:
        print("Problem reading the image: " + image_path)
        if not os.path.exists(src_path + "/drive_files/" + "download_errors.txt"):
            myfile = open(src_path + "/drive_files/" + "download_errors.txt", "w")
        else:
            myfile = open(src_path + "/drive_files/" + "download_errors.txt", "a")
        myfile.write(new_name.split("_")[1])
        myfile.close()
    else:
        final_path = destiny_folder + "/" + new_name + "." + format_imatge
        result_image.convert("RGB").save(final_path)


def eliminar_carpeta(folder_to_delete):
    for f in os.listdir(folder_to_delete):
        os.remove(folder_to_delete + "/" + f)
    os.rmdir(folder_to_delete)


def fer_txt_noms_carpeta(folder_path):
    myfile = open(folder_path + "/file_names.txt", "w")
    for f in os.listdir(folder_path):
        myfile.write(f)
    myfile.close()


def check_night_image(image_path, area_threshold, pixel_threshold):
    # copy_image_to_folder(image_path,destiny_folder,new_name):
    img = np.asarray(Image.open(image_path))
    tamany = img.shape

    total_pixels = tamany[0] * tamany[1]
    grey_image = img[:, :, 0] / 3 + img[:, :, 1] / 3 + img[:, :, 2] / 3
    final_uint8_image = (np.dstack((grey_image, grey_image, grey_image))).astype(
        np.uint8
    )
    grey_pixels_number = np.sum(
        (img[:, :, 0] == img[:, :, 1]) & (img[:, :, 1] == img[:, :, 2])
    )
    pixel_mean_value = np.sum(grey_image) / total_pixels
    grey_area_percent = 100 * grey_pixels_number / total_pixels
    # print(str(pixel_mean_value))
    if (grey_area_percent > area_threshold) or (pixel_mean_value < pixel_threshold):
        # print(str(grey_area_percent)+'%     mean: '+ str(pixel_mean_value)+'    ',end='')
        return True
    else:
        return False


def check_night_folder(folder_to_check, area_th, pixel_th):
    image_list = os.listdir(folder_to_check)
    destiny_folder = folder_to_check + "/night_images"
    check_folder_and_create(destiny_folder, 1)
    for i in image_list:
        if i.endswith(".jpg"):  # si es imatge
            im_path = folder_to_check + "/" + i
            if check_night_image(
                im_path, area_th, pixel_th
            ):  # mirem si es imatge en gris
                copy_image_to_folder(im_path.split(".")[0], destiny_folder, i)
                os.remove(im_path)
                print(i)


def filenames_to_txt(folder_path):
    file_name = "/file_names.txt"
    for f in os.listdir(folder_path):
        if not os.path.exists(folder_path + file_name):
            myfile = open(folder_path + file_name, "w")
        else:
            myfile = open(folder_path + file_name, "a")
        myfile.write(f + "\n")
        myfile.close()


def copy_list_from_to(names_list, original_path, destiny_path):
    count = 0
    files_in_folder = os.listdir(original_path)
    for f in files_in_folder:
        if f.endswith(".jpg"):
            cam_id = f.split("_")[1].split(".")[0]
            # print(['{:0>8}'.format(str(n))+' ' +str(cam_id) for n in names_list if '{:0>8}'.format(str(n)) == cam_id])
            if [
                True for n in names_list if "{:0>8}".format(str(n)) == cam_id
            ]:  # si la camara esta a samples_images (q hauria de)
                count += 1
                check_folder_and_create(destiny_path, 1)
                copyfile(original_path + "/" + f, destiny_path + "/" + f)
                print("Copied " + str(count) + " elements into " + destiny_path)
            # print(cam_id)


def list_from_txt(file_path):
    myfile = open(file_path, "r")
    names = []
    for line in myfile:
        new_name = line.split()[0].split("_")[1].lstrip("0").replace(".jpg", "")
        names.append(new_name)

    return names


def list_from_patches_txt(file_path):
    myfile = open(file_path, "r")
    names = []
    for line in myfile:
        names.append(int(line.replace("\n", "")))

    return names


def make_image_patches(
    patch_size, original_folder_path, image_name, destiny_folder_path, *args
):
    if len(args) > 0:  # si hi ha arguments variables
        if len(args) == 1:  # si hi ha un
            patches_to_do = args[0]
    else:
        patches_to_do = []
    patches_to_do.sort()
    format = image_name.split(".")[-1]
    img = np.asarray(Image.open(original_folder_path + "/" + image_name))
    tamany = img.shape
    x_step = patch_size
    y_step = patch_size
    count = 0
    patches_done_count = 0
    must_do_patch = False
    if not os.path.exists(destiny_folder_path):
        os.makedirs(destiny_folder_path)
    for x in range(0, tamany[0], x_step):
        for y in range(0, tamany[1], y_step):
            p = img[x : (x + x_step), y : (y + y_step), :]
            result = Image.fromarray((p).astype(np.uint8))
            # print(patches_to_do[patches_done_count])
            if len(patches_to_do) == 0:
                must_do_patch = True
            elif (
                patches_to_do[patches_done_count] == count
            ):  # si el patch que anem a fer es el següent que hem de fer
                must_do_patch = True
            if must_do_patch:
                if patches_done_count < len(patches_to_do) - 1:
                    patches_done_count += 1
                # print('done')
                if not os.path.exists(
                    destiny_folder_path
                    + "/"
                    + image_name.replace("." + format, "")
                    + "_patch"
                    + str(count)
                    + "."
                    + format
                ):
                    result.save(
                        destiny_folder_path
                        + "/"
                        + image_name.replace("." + format, "")
                        + "_patch"
                        + str(count)
                        + "."
                        + format
                    )
            must_do_patch = False
            count += 1
    print("Done " + str(patches_done_count) + " patches")


def save_patches_to_do(folder_path):
    for f in os.listdir(folder_path):
        if f.endswith(".jpg"):
            cam_id = f.split("_")[1]

            patch_number = f.split("_")[2].replace("patch", "").replace(".jpg", "")
            file_name = cam_id + ".txt"
            print(
                "Saving patches names of "
                + cam_id
                + " to "
                + folder_path
                + "/"
                + file_name
            )
            if not os.path.exists(folder_path + "/" + file_name):
                myfile = open(folder_path + "/" + file_name, "w")
            else:
                myfile = open(folder_path + "/" + file_name, "a")
            myfile.write(patch_number + "\n")
            myfile.close()


def initialise_db(database_name):
    db_connection = sqlite3.connect(str(database_name + ".db"))
    return (db_connection, db_connection.cursor())


def close_db_cursor(connection):
    connection.commit()


# def download_camera_samples(service, camera_id, year):
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# pip3 install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


API_USER = "carles.anton01@estudiant.upf.edu"
API_PASSWORD = "mauvemouse32"
src_path = "/Volumes/Data_HD/Users/Carles/TFG/codes"
# src_path='/home/u124223/Documents'


# plt.ion() # enables MATPLOTLIB interactive mode


area_t = 40
pix_t = 70
night_check_folder = src_path + "/sample_images"

"""
night_images_list=list_from_txt(src_path+'/sample_images/night_images/file_names.txt')
get_sample_image_in_list(night_images_list,'2017')
"""
# check_night_folder(night_check_folder,area_t,pix_t)
# check_night_image(src_path+'/sample_images/2017_00000011.jpg',40,50)

# get_sample_image_in_list(cam_list,'2017')

# db_cursor.execute('''DROP TABLE aqi_index''')
# db_cursor.execute("CREATE TABLE IF NOT EXISTS aqi_index (date CHARACTER PRIMARY KEY, aqi INTEGER , defining_parameter TINYTEXT, site CHARACTER)")

# db_cursor.execute('''CREATE TABLE IF NOT EXISTS monitors (whole_id CHARACTER PRIMARY KEY, monitor_id CHARACTER,parameter_codes TEXT , lat FLOAT, long FLOAT , state TINYTEXT, county TEXT, city TEXT, full_address TEXT, scale_definition TEXT,  FOREIGN KEY (monitor_id) REFERENCES sites(monitor_id))''')
# iterate_csv_monitors('/Volumes/Data_HD/Users/Carles/TFG/codes/aqs_monitors.csv')
# db_cursor.execute('''DROP TABLE sites''')
# db_cursor.execute('''CREATE TABLE IF NOT EXISTS sites (monitor_id CHARACTER PRIMARY KEY, lat FLOAT, long FLOAT , state TINYTEXT, county TEXT, city TEXT, closing_date TEXT, met_stie_type TEXT)''')
# iterate_csv_sites('/home/u124223/Documents/aqs_sites.csv',db_cursor)
# db_cursor.execute('''DROP TABLE camera_sites''')
# create_camera_site_table(db_cursor)

# iterate_aqi_folder(db_cursor,src_path+'/csv')
# parse_aqi(db_cursor,src_path+'/daily_aqi_by_cbsa_2017.csv')


"""
data_inici='20150101'
data_final='20160101'
relations=get_cameras_with_sites(db_cursor)
fknid=relations[0][1]
destiny_path=str(src_path+'/data/'+str(relations[0][0]).zfill(5))
monitors_lloc=get_monitors_condition("where monitor_id = '" + fknid + "'", db_cursor)
m=monitors_lloc[0]
#check_folder_and_create(destiny_path+'/pollution_data',0)
#get_usa_monitor_data('DMCSV',m.parameter_code,data_inici,data_final,m.id.split("-")[0],m.id.split("-")[1],m.id.split("-")[2],destiny_path+'/pollution_data')

get_site_data(fknid, data_inici, data_final,db_cursor,destiny_path)
"""


# get_all_data(db_cursor,'20140101','20170101')

if __name__ == "__main__":
    db_connection, db_cursor = initialise_db("camera_location")

    totes_cameras = get_cameras(db_cursor)
    cams = get_cameras_condition("where country = 'USA'", db_cursor)

    print("Total cameras = " + str(len(totes_cameras)))

    print("USA cameras = " + str(len(cams)))

    print(
        "USA camera % of total relation = "
        + "{:02.2f}".format(100 * float(len(cams) / len(totes_cameras)))
    )

    close_db_cursor(db_connection)


"""
coordenades=[52.74959372,-155.5664063]
camera1=coordinates_to_camera(7910,coordenades[0], coordenades[1],10)
print (camera1.full_address)
"""
