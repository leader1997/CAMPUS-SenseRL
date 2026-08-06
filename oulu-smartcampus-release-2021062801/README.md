Please always refer to up-to-date documentation regarding this release.

---

# Smart Campus IoT data

This dataset contains indoor air-quality, presence and light measurements from the University of Oulu Linnanmaa-campus and botanical garden. Collected by low-power wireless sensors connected to the 5GTN LoRa-network.

The dataset begins on 1.7.2020, this initial release containing data until 1.6.2021. Later the set may be appended as new data is released.



## 'application' and 'lora' .csv-tables

The data is split into two tables, 'application' and 'lora', the former containing a numeric representation of the physical quantities under measurement, and the latter containing values exposed by our LoRa-gateway, including metrics such as signal strength.

A month of this .csv formatted data consumes around 100 MiB's of memory, however the files are distributed in a compressed archive.



## Sensors and measurements

The network consists of 429 deployed sensors. Each of which transmit once every 15 minutes.

The following quantities are included in the 'application'-table.

Description        |Column name     |Unit                 |Only in
-------------------|----------------|---------------------|-------
Air temperature    |temperature     |[°C]                 |
Relative humidity  |humidity        |[% RH]               |
Light              |light           |(linear index)       |
Passive infrared   |motion          |(linear index, pir)  |
Co2                |co2             |[ppm]                |ers-co2
Battery voltage    |battery         |[V]                  |
Average SPL        |sound_avg       |[dB]                 |ers-sound
Peak SPL           |sound_peak      |[dB]                 |ers-sound
Soil moisture      |moisture        |[%]                  |elt-2-with-soil-moisture
Atmospheric        |pressure        |[bar]                |elt-2-with-soil-moisture
Acceleration (X)   |acceleration_x  |                     |elt-2-with-soil-moisture
             (Y)   |acceleration_y  |                     |
             (Z)   |acceleration_z  |                     |


Data from three different kinds of sensors are combined. Some quantites are only measured by some sensors, as indicated by the 'Only in' column above. Latest information on sensor accuracy can be found from the vendors datasheets.

Please always refer to up-to-date documentation regarding this release.


## Sensor metadata

devices.jsonl uses the jsonlines-format. It contains metadata about individual sensor devices, including the geolocation (Web Mercator EPSG:3785) and spoken description of installation location.



## Time format

The time format of the 'time'-column is epoch milliseconds:

```py
example = 1596240013479


import time

time.gmtime(example / 1000)
# time.struct_time(tm_year=2020, tm_mon=8, ...)

time.localtime(example / 1000)
# time.struct_time(tm_year=2020, tm_mon=8, ...)


from datetime import datetime as dt

dt.fromtimestamp(example / 1000)
# datetime.datetime(2020, 8, 1, 3, 0, 13, 479000)
```

## oulu-smartcampus-export

oulu-smartcampus-export was used to create these .csv-files.

---

Aleksi Pirttimaa
Research assistant
Smart Campus
CWC Networks and systems
University of Oulu

