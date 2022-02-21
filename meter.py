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
pd.options.mode.chained_assignment = None

def init_data():
    #Laster inn energidata fra Statsbygg EOS
    power = pd.read_excel('./aalesund/G-bygg.xlsx')
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

    p = []
    for m in meters:
        p.append(power[power.meter==m].power)
    return p

class Building():
    def __init__(self, name, area, floors, year):
        self.name = name
        self.area = area
        self.floors = floors
        self.construction_year = year
        self.weather_data = None
        self.meters = []
        self.total_consumption = None
        self.day_difference_meter_data = []
        self.set_standard_u_value()
        self.scale_building_size()
        self.set_expected_heating_loss()
        self.set_excpected_airflow_exchange()
        self.set_optimal_airflow_exchange()

        self.has_cooling = False
        self.has_heating = False
        self.has_outdoor_heating = False
        self.has_ventilation_heating = False
        self.has_ventilation_cooling = False

        self.constant_load = 0 # Constant loss [kW]
        self.constant_load_day = 0 # Constant loss [kW]
        self.constant_load_night = 0 # Constant loss [kW]

        self.heating_loss = 0 # Heatingloss [kW/k]
        self.outdoor_heating = 0 # OutDoor heating [kW]
        self.outdoor_heating_with_rainsensor = 0 # OutDoor heating [kW]
        self.heating_loss_at0 = 0 # Predicted heatingpower at 0 degrees C [kW]
        self.heating_start_temp = 0 # Temperature where heating == 0 [degrees C]

        self.cooling_loss = 0 # Heatingloss [kW/k]
        self.cooling_loss_at0 = 0 # Predicted heatingpower at 0 degrees C [kW]
        self.cooling_start_temp = 0 # Temperature where heating == 0 [degrees C]

        #VentilationHeating
        self.ventilation_heating = 0  # kW heating power for heating of outdoor air
        self.ventilation_heating_at0 = 0
        self.start_temp_for_ventilation_heating = 0

        #ventilationCooling
        self.ventilation_cooling = 0  # kW heating power for heating of outdoor air
        self.ventilation_cooling_at0 = 0
        self.start_temp_for_ventilation_cooling = 0
        
    def set_standard_u_value(self):

        years = [1949, 1969, 1987, 1997, 2007, 2010, 2017]
        roof = [1, 0.58, 0.2, 0.15, 0.13, 0.13, 0.13]
        floor = [0.8, 0.6, 0.3, 0.15, 0.15, 0.15, 0.15]
        wall = [1.1, 0.6, 0.3, 0.22, 0.18, 0.18, 0.15]
        window = [3.5, 2.7, 2.4, 1.6, 1.2, 1.2, 0.8]

        r = np.interp(self.construction_year, years, roof)
        f = np.interp(self.construction_year, years, floor)
        wa = np.interp(self.construction_year, years, wall)
        wi = np.interp(self.construction_year, years, window)

        self.floor_u_value = f
        self.roof_u_value = r
        self.facade_u_value = wa
        self.window_u_value = wi

    def scale_building_size(self):
        self.floor_area = self.area/self.floors
        self.roof_area = self.floor_area
        self.facade_area = np.sqrt(self.floor_area)*3*4*self.floors*0.65
        self.window_area = np.sqrt(self.floor_area)*3*4*self.floors*0.35

    def set_expected_heating_loss(self):
        self.standard_heating_loss = self.floor_area*self.floor_u_value + self.roof_area*self.roof_u_value + self.facade_area*self.facade_u_value + self.window_area*self.window_u_value

    def set_excpected_airflow_exchange(self):
        self.airflow = self.area * 8
        
    def set_optimal_airflow_exchange(self):
        self.optimal_airflow_heating_loss = self.airflow*0.35*(1-0.85)/1000

    def set_one_meter(self, meter_data):
        met = Meter(meter_data, self.weather_data)
        self.meters.append(met)
        self.total_consumption = met

    def set_heating_and_main_meter(self, main_meter, heating_meter):
        total_power = main_meter + heating_meter
        total_power.dropna(inplace=True)
        self.total_consumption = Meter(total_power, self.weather_data)
        self.meters.append(Meter(main_meter, self.weather_data))
        self.meters.append(Meter(heating_meter, self.weather_data))

    def set_multiple_meters(self, meters):
        total_power = pd.Series()
        for meter in meters:
            total_power = total_power + meter
            self.meters.append(Meter(meter, self.weather_data))
        total_power.dropna(inplace=True)
        self.total_consumption = Meter(total_power, self.weather_data)
        
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
            if len(rain) > 0:
                df = rain.merge(temp, on='referenceTime')
                df.columns = ['time', 'rain', 'mm', 'temperature', 'unit']
                df = df[['time', 'rain', 'temperature']]
            else:
                df = temp
                df.columns = ['time', 'temperature', 'unit']
                df = df[['time', 'temperature']]
            df = df.set_index('time')
        else:
            print('Error! Returned status code %s' % r.status_code)
            print('Message: %s' % json['error']['message'])
            print('Reason: %s' % json['error']['reason'])
            df = None
        
        return df

    def map_building_loss_parameters(self):
        for m in self.meters:
            if m.day_load.has_cooling:
                self.has_cooling = True
                self.cooling_loss = self.cooling_loss + m.day_load.cooling_loss
                self.cooling_loss_at0 = self.cooling_loss_at0 + m.day_load.cooling_loss_at0

            if m.day_load.has_outdoor_heating:
                self.has_outdoor_heating = True
                self.outdoor_heating = self.outdoor_heating + m.day_load.outdoor_heating
                self.outdoor_heating_with_rainsensor = self.outdoor_heating_with_rainsensor + m.day_load.outdoor_heating_with_rainsensor

            if m.day_load.has_ventilation_cooling:
                self.has_ventilation_cooling = True
                self.ventilation_cooling = self.ventilation_cooling + m.day_load.ventilation_cooling
                self.ventilation_cooling_at0 = self.ventilation_cooling_at0 + m.day_load.ventilation_cooling_at0

            self.constant_load = self.constant_load + m.day_load.constant_load
            self.constant_load_day = self.constant_load_day + m.day_load.constant_load_day
            self.constant_load_night = self.constant_load_night = m.day_load.constant_load_night

            self.heating_loss = self.heating_loss + m.day_load.heating_loss
            self.heating_loss_at0 = self.heating_loss_at0 + m.day_load.heating_loss_at0

            self.ventilation_heating = self.ventilation_heating + m.day_load.ventilation_heating
            self.ventilation_heating_at0 = self.ventilation_heating_at0 + m.day_load.ventilation_heating_at0
        
        self.heating_start_temp = self.heating_loss_at0/ self.heating_loss
        self.start_temp_for_ventilation_heating = self.ventilation_heating_at0/self.ventilation_heating
        if self.cooling_loss > 0:
            self.cooling_start_temp = self.cooling_loss_at0/self.cooling_loss
        if self.ventilation_cooling > 0:
            self.start_temp_for_ventilation_cooling = self.ventilation_cooling_at0/self.ventilation_cooling

        self.day_difference_meter_data.append(m.day_load.day_difference_meter_data)
    
    def view_plot_night(self):
        plt.scatter(self.total_consumption.day_load.meter_data.temperature, self.total_consumption.day_load.meter_data.power)
        tMin = min(self.total_consumption.day_load.meter_data.temperature)
        tMax = max(self.total_consumption.day_load.meter_data.temperature)
        Y_pred_min = self.heating_loss_at0 + self.heating_loss * tMin + self.constant_load_night
        Y_pred_max = self.heating_loss_at0 + self.heating_loss * tMax + self.constant_load_night

        Y_pred_min2_outdoor_heat = self.heating_loss_at0 + self.outdoor_heating + self.outdoor_heating_with_rainsensor  + self.constant_load_night
        Y_pred_max2_outdoor_heat = Y_pred_min2_outdoor_heat + self.heating_loss * tMin

        plt.plot([tMin, tMax], [Y_pred_min, Y_pred_max], color='red')
        if self.has_outdoor_heating:
            plt.plot([0, tMin], [Y_pred_min2_outdoor_heat, Y_pred_max2_outdoor_heat], color='orange')

        plt.xlabel('Outdoor temperature (C)')
        plt.ylabel('Energy consumption [kWh/h]')
        
        # Tittel
        plt.title('Temperature dependence peakLoad-lowLoad each day')
        if self.has_outdoor_heating:
            plt.legend(['Heating power night [kWh/h]', 'Heatloss coefficent: ' + str(round(self.heating_loss,2)) + ' kW/K', 'Outdoor heating: ' + str(round(self.outdoor_heating + self.outdoor_heating_with_rainsensor,2)) + ' kW'])
        else:
            plt.legend(['Heating power night [kWh/h]', 'Heatloss coefficent: ' + str(round(self.heating_loss,2)) + ' kW/K'])

        plt.show()

    def view_plot_day(self):
        plt.scatter(self.total_consumption.day_load.day_difference_meter_data.temperature, self.total_consumption.day_load.day_difference_meter_data.pDiff)
        tMin = min(self.total_consumption.day_load.day_difference_meter_data.temperature)
        tMax = max(self.total_consumption.day_load.day_difference_meter_data.temperature)
        if self.has_ventilation_cooling:
            tMax = 10
            tMin2 = 10
            tMax2 = max(self.total_consumption.day_load.day_difference_meter_data.temperature)
            Y_pred_min2 = self.ventilation_cooling_at0 + self.ventilation_cooling * tMin2
            Y_pred_max2 = self.ventilation_cooling_at0 + self.ventilation_cooling * tMax2
        Y_pred_min = self.ventilation_heating_at0 + self.ventilation_heating * tMin
        Y_pred_max = self.ventilation_heating_at0 + self.ventilation_heating * tMax

        
        plt.plot([tMin, tMax], [Y_pred_min, Y_pred_max], color='red')
        if self.has_ventilation_cooling:
            plt.plot([tMin2, tMax2], [Y_pred_min2, Y_pred_max2], color='orange')

        plt.xlabel('Outdoor temperature (C')
        plt.ylabel('Energy consumption [kWh/h]')
        
        # Tittel
        plt.title('Temperature dependence peakLoad-lowLoad each day')
        if self.has_ventilation_cooling:
            plt.legend(['Daytime heating power [kWh/h]', 'Heating coefficent: ' + str(round(self.ventilation_heating,2)) + ' kW/K', 'Cooling coefficent: ' + str(round(self.ventilation_cooling,2)) + ' kW/K'])
        else:
            plt.legend(['Daytime heating power [kWh/h]', 'Heating coefficent: ' + str(round(self.ventilation_heating,2)) + ' kW/K'])

        plt.show()

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
        self.has_cooling = False
        self.has_heating = False
        self.has_outdoor_heating = False
        self.has_ventilation_heating = False
        self.has_ventilation_cooling = False

        self.constant_load = None # Constant loss [kW]
        self.constant_load_day = None # Constant loss [kW]
        self.constant_load_night = None # Constant loss [kW]

        self.heating_loss = 0 # Heatingloss [kW/k]
        self.outdoor_heating = 0 # OutDoor heating [kW]
        self.outdoor_heating_with_rainsensor = 0 # OutDoor heating [kW]
        self.heating_loss_at0 = 0 # Predicted heatingpower at 0 degrees C [kW]
        self.heating_start_temp = 0 # Temperature where heating == 0 [degrees C]

        self.cooling_loss = 0 # Heatingloss [kW/k]
        self.cooling_loss_at0 = 0 # Predicted heatingpower at 0 degrees C [kW]
        self.cooling_start_temp = 0 # Temperature where heating == 0 [degrees C]

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
        self.set_cooling()
        self.linear_regression_night_outdoor_heating()
        self.linear_regression_day()

    def set_constant_load(self):
        self.constant_load = min(self.meter_data['power'])
        self.constant_load_day = min(self.meter_data['power'].between_time('07:00', '16:00'))
        self.constant_load_night = min(self.meter_data['power'].between_time('01:00', '04:00'))
        
    def set_cooling(self):
        linear_regressor = LinearRegression()
        pow = self.meter_data[self.meter_data.temperature>15].power.values.reshape(-1, 1)
        temp = self.meter_data[self.meter_data.temperature>15].temperature.values.reshape(-1, 1)
        linear_regressor.fit(temp, pow)
        self.cooling_loss = linear_regressor.coef_[0][0]
        if self.cooling_loss < 0:
            self.has_cooling = True
            self.cooling_at0 = linear_regressor.intercept_[0]
            self.cooling_start_temp = self.cooling_at0/self.cooling_loss

    def linear_regression_night_outdoor_heating(self):
        self.night_meter_data = self.meter_data.between_time('01:00', '04:00')
        linear_regressor = LinearRegression()
        pow = self.night_meter_data['power']-self.constant_load_night
        temp = self.night_meter_data.temperature.to_frame()
        temp.loc[:, 'vk'] = 0
        temp.loc[temp.temperature<0, 'vk'] = 1
        temp = temp[['temperature', 'vk']]
        linear_regressor.fit(temp, pow)
        self.heating_loss = linear_regressor.coef_[0]
        self.outdoor_heating = linear_regressor.coef_[1]
        #self.outdoor_heating_with_rainsensor = linear_regressor.coef_[1]
        self.heating_loss_at0 = linear_regressor.intercept_
        self.heating_start_temp = -self.heating_loss_at0/self.heating_loss
        if self.outdoor_heating > 5:
            self.has_outdoor_heating = True
        else:
            self.linear_regression_night()

    def linear_regression_night(self):
        self.night_meter_data = self.meter_data.between_time('01:00', '04:00')
        linear_regressor = LinearRegression()
        pow = self.night_meter_data['power']-self.constant_load_night
        pow = pow.values.reshape(-1, 1)
        temp = self.night_meter_data.temperature.values.reshape(-1, 1)
        linear_regressor.fit(temp, pow)
        self.heating_loss = linear_regressor.coef_[0][0]
        self.heating_loss_at0 = linear_regressor.intercept_[0]
        self.heating_start_temp = -self.heating_loss_at0/self.heating_loss

    def linear_regression_day(self):
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
        if self.ventilation_cooling > 0:
            self.has_ventilation_cooling = True
        
    def view_plot_night(self, excpected_heating_loss=None):
        plt.scatter(self.night_meter_data.temperature, self.night_meter_data.power)
        tMin = min(self.night_meter_data.temperature)
        tMax = max(self.night_meter_data.temperature)
        Y_pred_min = self.heating_loss_at0 + self.heating_loss * tMin + self.constant_load_night
        Y_pred_max = self.heating_loss_at0 + self.heating_loss * tMax + self.constant_load_night

        Y_pred_min2_outdoor_heat = self.heating_loss_at0 + self.outdoor_heating + self.outdoor_heating_with_rainsensor  + self.constant_load_night
        Y_pred_max2_outdoor_heat = Y_pred_min2_outdoor_heat + self.heating_loss * tMin     

        plt.plot([tMin, tMax], [Y_pred_min, Y_pred_max], color='red')
        if self.has_outdoor_heating:
            plt.plot([0, tMin], [Y_pred_min2_outdoor_heat, Y_pred_max2_outdoor_heat], color='orange')

        if excpected_heating_loss:
            Y_pred_min3 = -self.heating_start_temp*excpected_heating_loss + excpected_heating_loss * tMin + self.constant_load_night
            Y_pred_max3 = -self.heating_start_temp*excpected_heating_loss + excpected_heating_loss * tMax + self.constant_load_night
            plt.plot([tMin, tMax], [Y_pred_min3, Y_pred_max3], color='green')

        plt.xlabel('Utetemperatur (C)')
        plt.ylabel('Energiforbruk [kWh/h]')
        
        # Tittel
        plt.title('Kalkulert varmetap')
        if self.has_outdoor_heating:
            plt.legend(['Forbruk oppvarming [kWh/h]', 'Varmetap: ' + str(round(self.heating_loss,2)) + ' kW/K', 'Utendørs varme: ' + str(round(self.outdoor_heating + self.outdoor_heating_with_rainsensor,2)) + ' kW'])
        else:
            plt.legend(['Forbruk oppvarming [kWh/h]', 'Varmetap: ' + str(round(self.heating_loss,2)) + ' kW/K'])

        plt.show()

    def view_plot_day(self, optimal_heating_loss=None):
        plt.scatter(self.day_difference_meter_data.temperature, self.day_difference_meter_data.pDiff)
        tMin = min(self.day_difference_meter_data.temperature)
        tMax = max(self.day_difference_meter_data.temperature)
        if self.has_ventilation_cooling:
            tMax = 10
            tMin2 = 10
            tMax2 = max(self.day_difference_meter_data.temperature)
            Y_pred_min2 = self.ventilation_cooling_at0 + self.ventilation_cooling * tMin2
            Y_pred_max2 = self.ventilation_cooling_at0 + self.ventilation_cooling * tMax2
        Y_pred_min = self.ventilation_heating_at0 + self.ventilation_heating * tMin
        Y_pred_max = self.ventilation_heating_at0 + self.ventilation_heating * tMax
        
        plt.plot([tMin, tMax], [Y_pred_min, Y_pred_max], color='red')
        if self.has_ventilation_cooling:
            plt.plot([tMin2, tMax2], [Y_pred_min2, Y_pred_max2], color='orange')

        if optimal_heating_loss:
            Y_pred_min2 = -self.start_temp_for_ventilation_heating*optimal_heating_loss + optimal_heating_loss * tMin
            Y_pred_max2 = -self.start_temp_for_ventilation_heating*optimal_heating_loss + optimal_heating_loss * tMax
            plt.plot([tMin, tMax], [Y_pred_min2, Y_pred_max2], color='green')

        plt.xlabel('Utendørs temperatur (C')
        plt.ylabel('Energiforbruk [kWh/h]')
        
        # Tittel
        plt.title('Energiforbruk oppvarming uteluft/ ventilasjon')
        if self.has_ventilation_cooling:
            plt.legend(['Energiforbruk oppvarming uteluft [kWh/h]', 'Temperaturkoeffisient luftoppvarming: ' + str(round(self.ventilation_heating,2)) + ' kW/K', 'Temperaturkoeffisient luftkjøling: ' + str(round(self.ventilation_cooling,2)) + ' kW/K'])
        else:
            plt.legend(['Energiforbruk oppvarming uteluft [kWh/h]', 'Temperaturkoeffisient luftoppvarming: ' + str(round(self.ventilation_heating,2)) + ' kW/K'])
        plt.show()

class MeterMetaData():
    def __init__(self):
        self.heatPump=False
        self.districtHeating=False
        self.electricHeating=False
        self.outdoorHeating=False
        self.light=False
        self.server =False
        self.ventilationFans=False
        self.generalConsumption=False
        self.cooling=False
        
    def getMetaDict(self):
        d = self.__dict__
        for key in d:
            d[key] = vars(d[key])
        return d

    def get_meta(self):
        return vars(self)

    def set_meta(self, new_meta):
        for key in new_meta.keys():
            setattr(self, key, new_meta[key])


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
        
