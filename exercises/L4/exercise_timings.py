import timeit
import os
from statistics import mean, stdev


def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing=timeit.default_timer() - starttime
    return timing

def training_timing():
    starttime = timeit.default_timer()
    os.system('python training.py')
    timing=timeit.default_timer() - starttime
    return timing

def measure_and_save_timings():
    ingestion_times = []
    training_times = []
    for i in range(20):
        t = ingestion_timing()
        ingestion_times.append(t)
        t = training_timing()
        training_times.append(t)
    return (
        mean(ingestion_times),
        stdev(ingestion_times),
        min(ingestion_times),
        max(ingestion_times),
        mean(training_times),
        stdev(training_times),
        min(training_times),
        max(training_times)
        )

print(measure_and_save_timings())
