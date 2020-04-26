import csv
import sqlite3
import time
from datetime import datetime

import requests
import xlrd


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


def iterate_excel(path):
    book = xlrd.open_workbook(path)
    # first sheet
    first_sheet = book.sheet_by_index(0)
    for i in range(1, first_sheet.nrows):
        support_values = first_sheet.row_values(i)
        print(int(support_values[0]))
        camara_support = coordinates_to_camera(
            support_values[0], support_values[1], support_values[2], 100
        )
        camara_support.add_to_database(db_cursor)


def get_cameras():
    with db_connection:
        db_cursor.execute("SELECT * FROM cameras")
        totes_cameres = db_cursor.fetchall()
        for cam in totes_cameres:
            print(cam)


def iterate_csv(path):
    reader = csv.reader(open(path, "rt"), delimiter="|", quotechar=" ")
    print(reader.__next__())


# Create database and table

# db_connection = sqlite3.connect('site_monitor_location.db')
# db_cursor = db_connection.cursor()
# db_cursor.execute('''CREATE TABLE IF NOT EXISTS sites (site_id INTEGER PRIMARY KEY, lat FLOAT, long FLOAT,  country TINYTEXT , state TINYTEXT, county TEXT, district TEXT, city TEXT, street TEXT, number INTEGER, full_address TEXT)''')

iterate_csv("aqs_monitors.csv")

"""
coordenades=[52.74959372,-155.5664063]
camera1=coordinates_to_camera(7910,coordenades[0], coordenades[1],10)
print (camera1.full_address)
"""
