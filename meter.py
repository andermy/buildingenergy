import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import requests
import json
#import holidays
from dotenv import load_dotenv
load_dotenv()
import os

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
    power = power.set_index('date')
    power.index = pd.to_datetime(power.index).tz_localize('Etc/UCT')
    return power

class Building():
    def __init__(self, name, area, year):
        self.name = name
        self.area = area
        self.constructionYear = year
        self.weather_data = None
        self.meters = []
        self.total_consumption = None

    def setOneMeter(self, meter_data):
        met = Meter(meter_data, self.weather_data)
        self.meters.append(met)
        self.total_consumption = met

    def setHeatingAndMainMeter(self, main_meter, heating_meter):
        total_power = main_meter + heating_meter
        total_power.dropna(inplace=True)
        self.total_consumption = Meter(total_power, self.weather_data)
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
        client_id = os.getenv('CLIENT_ID')
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
        client_id = os.getenv('CLIENT_ID')
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
            df = df.set_index('time')
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
        self.day_load = None
        self.weekend_load = None
        self.set_holidays()
        self.set_variable_loads()

    def set_holidays(self):
        self.meter_data['weekday'] = self.meter_data.index.weekday
        self.meter_data['month'] = self.meter_data.index.month
        self.meter_data['hour'] = self.meter_data.index.hour
        #h = holidays.Norway()
        #self.meter_data['holiday'] = self.meter_data.apply()
        
    def set_variable_loads(self):
        self.day_load = PeriodLoad(self.meter_data[self.meter_data.weekday < 5])
        self.weekend_load = PeriodLoad(self.meter_data[self.meter_data.weekday >= 5])
        

class PeriodLoad():

    def __init__(self, meter_data):
        self.meter_data = meter_data
        self.day_difference_meter_data = None
        self.night_meter_data = None

        self.constant_load = None # Constant loss [kW]
        self.constant_load_day = None # Constant loss [kW]
        self.constant_load_night = None # Constant loss [kW]

        self.heating_loss = None # Heatingloss [kW/k]
        self.outdoor_heating = None # OutDoor heating [kW]
        self.outdoor_heating_with_rainsensor = None # OutDoor heating [kW]
        self.heating_loss_at0 = None # Predicted heatingpower at 0 degrees C [kW]
        self.heating_start_temp = None # Temperature where heating == 0 [degrees C]

        #VentilationHeating
        self.ventilation_heating = None  # kW heating power for heating of outdoor air
        self.ventilation_heating_at0 = None 
        self.start_temp_for_ventilation_heating = None

        #ventilationCooling
        self.ventilation_cooling = None  # kW heating power for heating of outdoor air
        self.ventilation_cooling_at0 = None 
        self.start_temp_for_ventilation_cooling = None

        # Collect statistics
        self.set_constant_load()
        self.linearRegressionNight()
        self.linearRegressionDay()

    def set_constant_load(self):
        self.constant_load = min(self.meter_data['power'])
        self.constant_load_day = min(self.meter_data['power'].between_time('07:00', '16:00'))
        self.constant_load_night = min(self.meter_data['power'].between_time('01:00', '04:00'))
        
    def linearRegressionNight(self):
        self.night_meter_data = self.meter_data.between_time('01:00', '04:00')
        linear_regressor = LinearRegression()
        pow = self.night_meter_data['power']-self.constant_load_night
        temp = self.night_meter_data[['temperature', 'rain']]
        temp.loc[:, 'vk'] = 0
        temp.loc[:,'vk2'] = 0
        temp.loc[(temp.temperature<0)&(temp.rain>0.2), 'vk']=1
        temp.loc[(temp.temperature<0), 'vk2']=1
        temp = temp[['temperature', 'vk','vk2']]
        linear_regressor.fit(temp, pow)
        self.heating_loss = linear_regressor.coef_[0]
        self.outdoor_heating = linear_regressor.coef_[2]
        self.outdoor_heating_with_rainsensor = linear_regressor.coef_[1]
        self.heating_loss_at0 = linear_regressor.intercept_
        self.heating_start_temp = -self.heating_loss_at0/self.heating_loss


    def linearRegressionDay(self):
        power_min = self.meter_data.resample('D')['power'].min()
        power_max = self.meter_data.resample('D')['power'].max()
        varme2 = self.meter_data.merge(power_min.rename('powerMin'), left_index=True, right_index=True, how='outer')
        varme2 = varme2.merge(power_max.rename('powerMax'), left_index=True, right_index=True, how='outer')
        varme2['pDiff'] = varme2.powerMax-varme2.powerMin
        varme2.dropna(inplace=True)
        confidence = 0.95
        cInterval = np.percentile(varme2.pDiff,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])
        varme2 = varme2[varme2.pDiff>cInterval[0]]
        self.day_difference_meter_data = varme2
        
        #Heatingdependend ventilation
        linear_regressor = LinearRegression()
        pow = varme2[varme2.temperature<10].pDiff.values.reshape(-1, 1)
        temp = varme2[varme2.temperature<10].temperature.values.reshape(-1, 1)
        linear_regressor.fit(temp, pow)
        self.ventilation_heating = linear_regressor.coef_[0][0]  # kW heating power for heating of outdoor air
        self.ventilation_heating_at0 = linear_regressor.intercept_[0] 
        self.start_temp_for_ventilation_heating = -self.ventilation_heating_at0/self.ventilation_heating

        #Coolingdependend ventilation
        linear_regressor = LinearRegression()
        pow = varme2[varme2.temperature>10].pDiff.values.reshape(-1, 1)
        temp = varme2[varme2.temperature>10].temperature.values.reshape(-1, 1)
        linear_regressor.fit(temp, pow)
        self.ventilation_cooling = linear_regressor.coef_[0][0]  # kW heating power for heating of outdoor air
        self.ventilation_cooling_at0 = linear_regressor.intercept_[0]
        self.start_temp_for_ventilation_cooling = self.ventilation_cooling_at0/self.ventilation_cooling
        
    def view_plot_night(self):
        plt.scatter(self.night_meter_data.temperature, self.night_meter_data.power)
        tMin = min(self.night_meter_data.temperature)
        tMax = max(self.night_meter_data.temperature)
        Y_pred_min = self.heating_loss_at0 + self.heating_loss * tMin + self.constant_load_night
        Y_pred_max = self.heating_loss_at0 + self.heating_loss * tMax + self.constant_load_night

        Y_pred_min2_outdoor_heat = self.heating_loss_at0 + self.outdoor_heating + self.outdoor_heating_with_rainsensor  + self.constant_load_night
        Y_pred_max2_outdoor_heat = Y_pred_min2_outdoor_heat + self.heating_loss * tMin

        plt.plot([tMin, tMax], [Y_pred_min, Y_pred_max], color='red')
        plt.plot([0, tMin], [Y_pred_min2_outdoor_heat, Y_pred_max2_outdoor_heat], color='orange')

        plt.xlabel('Outdoor temperature (C)')
        plt.ylabel('Energy consumption [kWh/h]')
        
        # Tittel
        plt.title('Temperature dependence peakLoad-lowLoad each day')
        plt.legend(['Heating power night [kWh/h]', 'Heatloss coefficent: ' + str(round(self.heating_loss,2)) + ' kW/K', 'Outdoor heating: ' + str(round(self.outdoor_heating + self.outdoor_heating_with_rainsensor,2)) + ' kW'])

        plt.show()

    def view_plot_day(self):
        plt.scatter(self.day_difference_meter_data.temperature, self.day_difference_meter_data.pDiff)
        tMin = min(self.day_difference_meter_data.temperature)
        tMax = 10
        Y_pred_min = self.ventilation_heating_at0 + self.ventilation_heating * tMin
        Y_pred_max = self.ventilation_heating_at0 + self.ventilation_heating * tMax

        tMin2 = 10
        tMax2 = max(self.day_difference_meter_data.temperature)
        Y_pred_min2 = self.ventilation_cooling_at0 + self.ventilation_cooling * tMin2
        Y_pred_max2 = self.ventilation_cooling_at0 + self.ventilation_cooling * tMax2

        plt.plot([tMin, tMax], [Y_pred_min, Y_pred_max], color='red')
        plt.plot([tMin2, tMax2], [Y_pred_min2, Y_pred_max2], color='orange')

        plt.xlabel('Outdoor temperature (C')
        plt.ylabel('Energy consumption [kWh/h]')
        
        # Tittel
        plt.title('Temperature dependence peakLoad-lowLoad each day')
        plt.legend(['Daytime heating power [kWh/h]', 'Heating coefficent: ' + str(round(self.ventilation_heating,2)) + ' kW/K', 'Cooling coefficent: ' + str(round(self.ventilation_cooling,2)) + ' kW/K'])

        plt.show()
    

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
        
