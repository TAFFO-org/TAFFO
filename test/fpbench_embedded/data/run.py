import random

N = 10000

# for i in range(100000):
#     print(random.uniform(-1000000,100000))


def generateValues(benchmark, N):
    for condition in benchmark.values():
        for variable in condition.values():
            for _ in range(N):
                tmp = random.uniform(variable[0][0],variable[0][1])
                variable[1].append(tmp)



def printBench(name, benchmark):
    for condition in benchmark.items():
        if len(benchmark) != 1:
            filename = name + str(condition[0])
        else:
            filename = name
        with open(filename, "w") as file:
            line = ""
            #write csv header
            for varname in condition[1]:
                line += "{}, ".format(varname)
        
            line = line[:-2]
            line += "\n"
            benchRes = [values[1] for values in condition[1].values()]

            for i in range(len(benchRes[0])):
                for j in range(len(benchRes)):
                    line += "{}, ".format(benchRes[j][i])
                line = line[:-2]
                line += "\n"
            
            file.write(line)


            



def simpleRangeBench():
    global N
    benchmarks = { 
        # doppler1 (and (<= -100 u 100) (<= 20 v 20000) (<= -30 T 50))
        # doppler2 (and (<= -125 u 125) (<= 15 v 25000) (<= -40 T 60))
        # doppler3 (and (<= -30 u 120) (<= 320 v 20300) (<= -50 T 30))
        "Doppler" : {
        1 : { "u" :  ((-100, 100), []), "v" : ((20, 20000), []), "T": ((-30, 50), [])   },
        2 : { "u" :  ((-125, 125), []), "v" : ((15, 25000), []), "T": ((-40, 60), [])   },
        3 : { "u" :  ((-30, 120), []), "v" : ((320, 20300), []), "T": ((-50, 30), [])   },
        },
        # turbine1 (and (<= -9/2 v -3/10) (<= 2/5 w 9/10) (<= 19/5 r 39/5))
        # turbine2 (and (<= -9/2 v -3/10) (<= 2/5 w 9/10) (<= 19/5 r 39/5))
        # turbine3 (and (<= -9/2 v -3/10) (<= 2/5 w 9/10) (<= 19/5 r 39/5))
        "Turbine" : {
        1 : { "v" :  ((-9/2, -3/10), []), "w" : ((2/5, 9/10), []), "r": ((19/5, 39/5), [])   },
        2 : { "v" :  ((-9/2, -3/10), []), "w" : ((2/5, 9/10), []), "r": ((19/5, 39/5), [])   },
        3 : { "v" :  ((-9/2, -3/10), []), "w" : ((2/5, 9/10), []), "r": ((19/5, 39/5), [])   },
        },
        #(and (<= 1 radius 10) (<= 0 theta 360))
        "Cx" : {
            1 : {"radius": ((1, 10), []), "theta": ((0, 360) , [])}
        },
        #(and (<= 1 radius 10) (<= 0 theta 360))
        "Cy" : {
            1 : {"radius": ((1, 10), []), "theta": ((0, 360) , [])}
        },
        #(and (<= 0 lat1 2/5) # (<= 1/2 lat2 1) (<= 0 lon1 62831853/20000000) (<= -62831853/20000000 lon2 -1/2))
        "Azimuth" : {
            1 : {"lat1": ((0, 2/5), []), "lat2": ((1/2, 1), []), "lon1": ((0, 62831853/20000000), []), "lon2": ((-62831853/20000000, -1/2), []), }
        },
        # (<= 1/10 v 1/2)
        "CarbonGas": {
            1 : {"v": ((1/10, 1/2), [])}
        },
        # (and (<= 1 x 100) (<= 1 y 100))
        "CRadius": {
            1 : {"x": ((1, 100), []), "y": ((1, 100), [])}
        },
                # (and (<= 1 x 100) (<= 1 y 100))
        "CTheta": {
            1 : {"x": ((1, 100), []), "y": ((1, 100), [])}
        },
        #(and (<= 0 t 300) (<= 1 resistance 50) (<= 1 frequency 100) (<= 1/1000 inductance 1/250) (<= 1 maxVoltage 12))
        "InstantaneousCurrent": {
            1 : {"t": ((0, 300), []), "resistance": ((1, 50), []), "frequency": ((1, 100), []), "inductance": ((1/1000, 1/250), []), "maxVoltage": ((1, 12), [])}
        },  
        #(and (<= -5 x1 5) (<= -20 x2 5))
        "JetEngine": {
            1 : {"x1": ((-5, 5), []), "x2": ((-20, 5), [])}
        },
        #(and (< 0 yd 50) (< 0 y 50))
        "Leadlag": {
            1 : {"yd": ((0, 50), []), "y": ((0, 50), [])}
        },
    }

    for benchmark in benchmarks:
        name = benchmark
        generateValues( benchmarks[name], N )
        printBench(name , benchmarks[name])




simpleRangeBench()