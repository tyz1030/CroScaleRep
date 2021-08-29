# This script samples images from Google Map Static API

import numpy as np
import urllib.request as request
import os
import cv2 as cv
from .urlsigner import sign_url
import socket
socket.setdefaulttimeout(30)

URL_TEST = "https://maps.googleapis.com/maps/api/staticmap?center=31.1034243,121.236696&zoom=20&size=640x640&scale=2&maptype=satellite&format=png&key="
PLAYGROUND = "/home/tyz/static_map_api-data/playground/test.png"
API_KEY = "AIzaSyDCYqIfRg-kLDEPDKFckzGIwn0nnP-__Sc"

class GMapRequester():
    URL_BASE = "https://maps.googleapis.com/maps/api/staticmap?"
    def __init__(self, apikey, secret = None, scale=2, zoom=20, size=(224, 224)) -> None:
        self.__secret = secret # url signing secret
        self.apikey = apikey
        self.scale = scale
        self.center = "Brooklyn+Bridge" # [latitiude, longtitude]
        self.zoom = zoom
        self.size = str(size[0])+"x"+str(size[1]) # max 640*640
        self.maptype = "satellite"  # "roadmap", "satellite", "terrain", or "hybrid"
        self.format = "png"
        pass

    def set_apikey(self, api):
        self.apikey = api
        return

    def set_center(self, center):
        self.center = str(center[0]) + "," + str(center[1]) #in the order of latitude, longtitude
        return

    def set_size(self, size):
        self.size = str(size[0])+"x"+str(size[1])
        return

    def set_zoom(self, zoom):
        self.zoom = zoom
        return

    def set_scale(self, scale):
        if scale != 1 or scale != 2:
            print("Fail to set scale. Scale needs to be 1 or 2. Given " + str(scale))
            return False
        else:
            self.scale = scale
            return True

    def form_full_url(self):
        url = self.URL_BASE
        url = url+"center="+self.center + "&"
        url = url+"zoom="+str(self.zoom) + "&"
        url = url+"size="+self.size + "&"
        url = url+"scale="+str(self.scale) + "&"
        url = url+"maptype="+str(self.maptype) + "&"
        url = url+"format="+str(self.format) + "&"
        url = url+"key="+self.apikey
        # print("url: " + url)
        return url

    def request_from_url(self, center = None, viz = False):
        # full_url = os.path.join(url, api_key)
        if center is not None:
            self.set_center(center)
        full_url = self.form_full_url()
        # sign url
        full_url = sign_url(full_url, self.__secret)

        #############
        count = 1
        while count <= 5:
            try:
                local_filename, headers= request.urlretrieve(full_url)                                                
                break
            except socket.timeout:
                err_info = 'Reloading for %d time'%count if count == 1 else 'Reloading for %d times'%count
                print(err_info)
                count += 1
        if count > 5:
            print("downloading picture fialed!")
        #############
        img = cv.imread(local_filename)
        if viz:
            cv.imshow('test', cv.resize(img, (640, 640)))
            cv.waitKey()
        # print(img.shape)
        return img


def main():
    api_key = API_KEY

    g_map_sampler = GMapRequester(api_key, secret = "zgdrECVG2zvvHU8W5fz4RJKpUSg=", scale = 2, zoom = 20, size=(640, 640))
    g_map_sampler.request_from_url(center = (47.708887, 10.303193), viz = True)


if __name__ == '__main__':
    main()