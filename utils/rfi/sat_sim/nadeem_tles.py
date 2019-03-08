#!/usr/bin/env python

import os
import argparse
from functools import partial

import sys

import katdal
import katpoint

import numpy as np

from datetime import datetime
import time as Time

# Starting function for TLE extract

def TLE_Extract(StartDate, EndDate, SatFile):
    '''

    This function will use the start and end date to find the appropriate TLE at L-band

    Requires:
        A login with password and
        the file "satcat_xls.xlsx"

    Return:
        A TLE file for date used

    '''
    print(StartDate,EndDate,SatFile)

    import spacetrack.operators as op
    from spacetrack import SpaceTrackClient
    import pandas as pd

    #theurl = 'API  https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID//orderby/TLE_LINE1\ ASC/format/tle'
    # if you want to run this example you'll need to supply
    # a protected page with your username and password
    username = input('Enter your username for space-track within quote ')
    password = input('Enter space track password within quote ')

    #
    # Using my login
    #
    st = SpaceTrackClient(username, password)

    #
    # Selecting the date range for  TLE request
    #

    drange = op.inclusive_range(StartDate,EndDate)
    #
    # extracting the TLEs here
    #
    lines = st.tle(iter_lines=True, epoch=drange, orderby='TLE_LINE1', format='tle')

    my_tle = []
    for line in lines:
        my_tle.append(line)

    # The name of satellite is given by the col 3-8 as from the ref
    # https://en.wikipedia.org/wiki/Two-line_element_set

    my_tle[0][2:7]

    ##### Filtering for know L-band satellites
    ### List of known GPS satellites

    #In L band (800 - 2150 MHz)

    #Global Positioning System (GPS) carriers and also satellite mobile phones, such as Iridium;
    #Inmarsat providing communications at sea, land and air; WorldSpace satellite radio.

    #
    # This is a file that has the known satellite number and their respective names
    #
    #satcat_df = pd.read_excel('TLE/satcat_xls.xlsx',header=None)
    satcat_df = pd.read_excel(SatFile,header=None)
    satcat_df.columns = ['Internl_Designator','NORAD_No','some_flags','Satellite_Name','Ownership','Launch_Date',\
              'Launch_site','Decay_date','Orb_period_min','Inclination_deg','Apogee_alt_Perigee_alt',\
             'Radar_cross','Orbital_status_code']
    # Filter on L-band satellites only now
    # Any other filter can be used at any time

    L_band = True

    if L_band:
        sat_are = ['GLONASS','Inmarsat','IRIDIUM','BEIDOU','BIIR','GALILEO','IRNSS','NAVSTAR','ALPHASAT','QZS']
        pattern = '|'.join(sat_are)

    print ("No of TLEs found ",np.sum(satcat_df['Satellite_Name'].str.contains(pattern)))

    filtered_df = satcat_df[satcat_df['Satellite_Name'].str.contains(pattern,na = False)]

    filtered_df = filtered_df.reset_index()

    del filtered_df['index']

    #filtered_df.head()

    print('\n ### Writing everything in a file format that is useable by katpoint ### \n')

    TLEOutputFile = 'MyTLE_'+StartDate.strftime('%Y-%m-%d')+'.tle'

    print('\n ### Will write TLE file to', TLEOutputFile,'### \n')


    for i in range(0,len(my_tle)-1,2):

        line1 = my_tle[i]
        line2 = my_tle[i+1]
        Sat_Name = filtered_df['Satellite_Name'][filtered_df.index[int(line1[2:7]) == filtered_df['NORAD_No']]]

        try:
            with open(TLEOutputFile, 'a') as fp:
                fp.write(str(Sat_Name.values[0])+'\n')
                fp.write(str(line1)+'\n')
                fp.write(str(line2)+'\n')

            fp.close()
        except:
            # This is in case there are some issues for code not to stop
            # print("No Sat_Name")
            pass

    print('Done')

    return(TLEOutputFile)


def get_argparser():
    "Get argument parser"
    parser = argparse.ArgumentParser(description="This method is specifically for TLE archive extract")

    argument = partial(parser.add_argument)
    argument('-f', '--file',  dest='file', help='Name of the rdb file to use')
    argument('-S', '--start-date',  dest='Sdate', help='Start date for TLE [2018-11-30 10:30:51.693]')
    argument('-E', '--End-date',  dest='Edate', help='End date for TLE [2018-11-30 10:30:51.693]')
    argument('-c', '--SatCat',  dest='SatCat', help='Full path with location of satcat_xls.xlsx')
    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()


    #ToUseH5 = input('Are you using rdb files [Y/N]')

    #
    # This function will check if we are extracting the datetime from rdb file or inputted by the user
    #

    if args.file is not None:

        filename = input('Enter the file to use')
        h5 = katdal.open(filename)
        Start_time = katpoint.Timestamp(h5.timestamps[0])
        End_time = katpoint.Timestamp(h5.timestamps[-1])
    else:

        Sdate_time_obj = datetime.strptime(args.Sdate, '%Y-%m-%d %H:%M:%S.%f')
        Edate_time_obj = datetime.strptime(args.Edate, '%Y-%m-%d %H:%M:%S.%f')


    TLEOutputFile = 'MyTLE_'+Sdate_time_obj.strftime('%Y-%m-%d')+'.tle'

    #
    # Will check if TLE exist will not re-create new one
    #

    if os.path.isfile(TLEOutputFile) and os.access(TLEOutputFile, os.R_OK):
        print ("File %s exists and is readable" %TLEOutputFile)
        TLEFileToUse = TLEOutputFile
    else:
        print("Either the file is missing or not readable, will fetch function now ...")
        TLEFileToUse = TLE_Extract(Sdate_time_obj,Edate_time_obj,args.SatCat)

if __name__ == '__main__':
    main()
