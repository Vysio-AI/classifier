## SPAR Dataset
---
### Source
---
[Github Repository](https://github.com/dmbee/SPAR-dataset)

### Description
---
Consists of 6-axis inertial sensor data (accelerometer and gyroscope) collected using an Apple Watch 2 and Apple Watch 3 from 20 healthy subjects (40 shoulders), as they perform 7 shoulder physiotherapy exercises. 

The activities are:

0. Pendulum (PEN)
1. Abduction (ABD)
2. Forward elevation (FEL)
3. Internal rotation with resistance band (IR)
4. External rotation with resistance band (ER)
5. Lower trapezius row with resistance band (TRAP)
6. Bent over row with 3 lb dumbell (ROW)

The subjects repeat each activity 20 times on each side (left and right).


### Data Format
---
The data is available in csv format in the csv folder. Each file represents a single activity being repeated 20 times. The files are named to convey:
`S1\_E0\_R`: subject 1, activity 0 (PEN), right side

Each file contains 6 axis inertial data collected at 50 Hz. The columns are:
* ax, ay, az: 3-axis accelerometer data measured in G
* wx, wy, wz: 3-axis gyroscope data measured in rad/s
Data reference frame is currently unknown

### Changes Log
---
