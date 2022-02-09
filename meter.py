import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import requests
import json

def init_data():
    #Laster inn energidata fra Statsbygg EOS
    power = pd.read_excel('Timeforbruk av energi for perioden 01.01.2021 - 21.11.2021 (1).xlsx')
    # Laster inn utetemperaturer
    utetemp = pd.read_excel('table3.xlsx')

    power = power[3:len(power)-4]
    col = ['date', 'meter','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21','22', '23', '24', 'sum']
    drops = ['sum']
    if len(power.columns) > 27:
        col = ['date', 'meter','type', 'unit', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21','22', '23', '24', 'sum']
        drops = ['sum', 'type', 'unit']
    power.columns = col
    power = power.drop(labels =drops, axis=1)
    power['date'] =pd.to_datetime(power['date'], format='%d.%m.%Y')
    power = power.set_index(['meter', 'date'])
    power = power.stack().reset_index()
    power.columns = ['meter', 'date', 'hour', 'power']
    power.date = power.date + pd.to_timedelta(power.hour+':00:00')
    power = power.reset_index()
    #power = power.set_index(['date'])
    meters = power.pivot(index='index', columns='meter', values='power').columns.tolist()
    if 'Snitt' in meters:
        meters.remove('Snitt')
    if 'Snitt ukedager' in meters:
        meters.remove('Snitt ukedager')

    utecols = ['Navn', 'Stasjon', 'Time', 'Rain', 'Temp']
    utetemp.columns = utecols
    utetemp = utetemp[utetemp.columns.tolist()[2:]]
    utetemp['Time'] = pd.to_datetime(utetemp['Time'], format='%d.%m.%Y %H:%M')

    power = power.merge(utetemp, left_on='date', right_on='Time', how='inner')
    power['weekday'] = power.date.dt.weekday
    power['month'] = power.date.dt.month
    power['hour'] = power.date.dt.hour
    powerNatt = power[power['hour'] == '4']
    power = power.set_index('date')
    return meters, power

class Building():
    def __init__(self, name, area, year):
        self.name = name
        self.area = area
        self.constructionYear = year
        self.weather_data = None
        self.meters = []

    def setOneMeter(self, meter_data):
        self.meters.append(Meter(meter_data, self.weather_data))

    def setHeatingAndMainMeter(self, main_meter, heating_meter):
        self.meters.append(Meter(main_meter, self.weather_data))
        self.meters.append(Meter(heating_meter, self.weather_data))

    def set_weather_data(self, place):
        location = self.get_location(place)
        self.weather_data = self.get_weather_data(location['id'])

    def get_location(self, place):
        endpoint = 'https://frost.met.no/sources/v0.jsonld'
        client_id = 'b7bd7c16-d236-457c-a285-d43dad15ed79'
        parameters = {'name': place}
        r = requests.get(endpoint, parameters, auth=(client_id,''))
        json = r.json()

        if r.status_code == 200:
            data = json['data']
            print('Data retrieved from frost.met.no!')
        else:
            print('Error! Returned status code %s' % r.status_code)
            print('Message: %s' % json['error']['message'])
            print('Reason: %s' % json['error']['reason'])

        return data[0]

    def get_data(self, source_id):
        endpoint = 'https://frost.met.no/observations/availableTimeSeries/v0.jsonld'
        client_id = 'b7bd7c16-d236-457c-a285-d43dad15ed79'
        parameters = {
            'sources': source_id,
        }
        r = requests.get(endpoint, parameters, auth=(client_id,''))
        json = r.json()

        if r.status_code == 200:
            data = json['data']
            print('Data retrieved from frost.met.no!')
        else:
            print('Error! Returned status code %s' % r.status_code)
            print('Message: %s' % json['error']['message'])
            print('Reason: %s' % json['error']['reason'])

        return data

    def get_weather_data(self, source_id, start_date='2020-12-31', end_date='2022-01-01'):
        endpoint = 'https://frost.met.no/observations/v0.jsonld'
        client_id = 'b7bd7c16-d236-457c-a285-d43dad15ed79'
        parameters = {
            'sources': source_id,
            'elements': 'air_temperature, sum(precipitation_amount PT1H)',
            'referencetime': start_date + '/' + end_date,
            'timeresolutions': 'PT1H'
        }
        r = requests.get(endpoint, parameters, auth=(client_id,''))
        json = r.json()

        if r.status_code == 200:
            data = json['data']
            print('Data retrieved from frost.met.no!')
            df = pd.json_normalize(data,  record_path =['observations'], meta=['referenceTime'])
            df['referenceTime'] = pd.to_datetime(df['referenceTime'])
            df = df[['referenceTime', 'value', 'unit']]
            rain = df[df['unit'] == 'mm']
            temp = df[df['unit'] == 'degC']
            df = rain.merge(temp, on='referenceTime')
            df.columns = ['time', 'rain', 'mm', 'temperature', 'unit']
            df = df[['time', 'rain', 'temperature']]
            #df = df.set_index('time')
        else:
            print('Error! Returned status code %s' % r.status_code)
            print('Message: %s' % json['error']['message'])
            print('Reason: %s' % json['error']['reason'])
            df = None
        
        return df    

class Meter():

    def __init__(self, meter_data, weather_data):
        #self.meter_data.index = pd.to_datetime(self.meter_data.index).tz_localize('Etc/UCT')
        self.meter_data = weather_data.merge(meter_data.rename('power'), left_index=True, right_index=True, how='inner')
        self.constant_load = None
        self.night_load = None
        self.day_load = None
        self.weekend_load = None
        #self.set_constant_load()
        #self.set_variable_loads()

    def set_hollidays(self):
        self.meter_data['weekday'] = self.meter_data.date.dt.weekday
        self.meter_data['month'] = self.meter_data.date.dt.month
        self.meter_data['hour'] = self.meter_data.date.dt.hour
        
    def set_constant_load(self):
        self.constant_load = min(self.meter_data['power'])

    def set_variable_loads(self):
        meterDataNoConstant = self.meter_data
        meterDataNoConstant.power = meterDataNoConstant.power-self.constantLoad
        self.night_load = MeterTemperatureRelation(meterDataNoConstant.between_time('01:00', '04:00'))
        self.day_load = MeterTemperatureRelation(meterDataNoConstant.between_time('07:00', '16:00'))
    

class MeterTemperatureRelation():

    def __init__(self, meter_data):
        self.meter_data = meter_data
        self.heatingLoss = None # Heatingloss [kW/k]
        self.outdoorHeating = None # OutDoor heating [kW]
        self.outdoorHeatingWithRainSensor = None # OutDoor heating [kW]
        self.heatingLossAt0 = None # Predicted heatingpower at 0 degrees C [kW]
        self.heatingStartTemp = None # Temperature where heating == 0 [degrees C]

        #VentilationHeating
        self.ventilationHeating = None  # kW heating power for heating of outdoor air
        self.ventilationHeatingAt0 = None 
        self.startTempForVentilationHeating = None

        #ventilationCooling
        self.ventilationCooling = None  # kW heating power for heating of outdoor air
        self.ventilationCoolingAt0 = None 
        self.startTempForVentilationHeating = None

    def set_constant_load(self):
        self.constant_load = min(self.meter_data['power'])
        
    def linearRegressionNight(self):
        linear_regressor = LinearRegression()
        pow = self.meter_data['power']-self.constant_load
        temp = self.meter_data[['temperature', 'rain']]
        temp['vk'] = 0
        temp['vk2'] = 0
        temp.loc[(temp.temperature<0)&(temp.rain>0.2), 'vk']=1
        temp.loc[(temp.temperature<0), 'vk2']=1
        temp = temp[['temperature', 'vk', 'vk2']]
        linear_regressor.fit(temp, pow)
        self.heatingLoss = linear_regressor.coef_[0]
        self.outdoorHeating = linear_regressor.coef_[2]
        self.outdoorHeatingWithRainSensor = linear_regressor.coef_[1]
        self.heatingLossAt0 = linear_regressor.intercept_
        self.heatingStartTemp = -self.heatingLossAt0/self.heatingLoss


    def linear2(self):
        power_min = self.meter_data.resample('D')['power'].min()
        power_max = self.meter_data.resample('D')['power'].max()
        varme2 = self.meter_data.merge(power_min.rename('powerMin'), left_index=True, right_index=True, how='outer')
        varme2 = varme2.merge(power_max.rename('powerMax'), left_index=True, right_index=True, how='outer')
        varme2['pDiff'] = varme2.powerMax-varme2.powerMin
        varme2.dropna(inplace=True)
        
        #Heatingdependend ventilation
        linear_regressor = LinearRegression()
        pow = varme2[varme2.Temp<10].pDiff
        temp = varme2.Temp
        linear_regressor.fit(temp, pow)
        self.ventilationHeating = linear_regressor.coef_[0]  # kW heating power for heating of outdoor air
        self.ventilationHeatingAt0 = linear_regressor.intercept_ 
        self.startTempForVentilationHeating = -self.ventilationHeatingAt0/self.ventilationHeating

        #Coolingdependend ventilation
        linear_regressor = LinearRegression()
        pow = varme2[varme2.Temp>15].pDiff
        temp = varme2.Temp
        linear_regressor.fit(temp, pow)
        self.ventilationCooling = linear_regressor.coef_[0]  # kW heating power for heating of outdoor air
        self.ventilationCoolingAt0 = linear_regressor.intercept_ 
        self.startTempForVentilationHeating = self.ventilationCoolingAt0/self.ventilationCooling
        
    

class MeterMetaData():
    def __init__(self):
        self.heatPump=MeterMetaDataOperations()
        self.districtHeating=MeterMetaDataOperations()
        self.electricHeating=MeterMetaDataOperations()
        self.outdoorHeating=MeterMetaDataOperations()
        self.light=MeterMetaDataOperations()
        self.server =MeterMetaDataOperations()
        self.ventilationFans=MeterMetaDataOperations()
        self.generalConsumption=MeterMetaDataOperations()
        self.cooling=MeterMetaDataOperations()
        
    def getMetaDict(self):
        d = self.__dict__
        for key in d:
            d[key] = vars(d[key])
        return d

class MeterMetaDataOperations():
    def __init__(self):
        self.constant=False
        self.daytime=False
        self.night=False
        
    def getMetaOperationsDict(self):
        return self.__dict__

    def setConstant(self):
        self.__init__()
        self.constant = True

    def setDaytime(self):
        self.__init__()
        self.daytime = True
    
    def setNight(self):
        self.__init__()
        self.night = True
        
