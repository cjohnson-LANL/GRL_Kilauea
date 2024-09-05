#PullSeismicWaves.py -o AHUD --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations AHUD --channel EH --parallel 8 --download
#PullSeismicWaves.py -o BYL  --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations BYL  --parallel 8 --download
# PullSeismicWaves.py -o RIMD --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations RIMD --parallel 8 --download
#PullSeismicWaves.py -o OTLD --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations OTLD --channel EH --parallel 8 --download
#PullSeismicWaves.py -o UWE  --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations UWE  --parallel 8 --download


PullSeismicWaves.py -o AHUD --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations AHUD --channel EH --parallel 8 --response
PullSeismicWaves.py -o BYL  --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations BYL  --parallel 8 --response
PullSeismicWaves.py -o RIMD --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations RIMD --parallel 8 --response
PullSeismicWaves.py -o OTLD --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations OTLD --channel EH --parallel 8 --response
PullSeismicWaves.py -o UWE  --datacenter IRIS -t0 2018-05-16 -t1 2018-08-06 --network HV --stations UWE  --parallel 8 --response
