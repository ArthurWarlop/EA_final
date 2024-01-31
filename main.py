import r0812080
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

benchmarks = [
    "tour50",
    "tour100",
    "tour200",
    "tour500",
    "tour750",
    "tour1000",
]

# benchmark = "tour50"
# mean = []
# best = []
# for i in tqdm(range(500)):
#     a = r0812080.r0812080()
#     meanObjective, bestObjective = a.optimize("./Benchmark/" + benchmark + ".csv")
#     file = open("./r0812080.csv")
#     data = np.genfromtxt(file, delimiter=",", usecols=[0, 1, 2, 3])
#     times = data[:, 1]
#     meanValues = data[:, 2]
#     bestValues = data[:, 3]
#     mean.append(meanValues[-1])
#     best.append(bestValues[-1])

# plt.ticklabel_format(useOffset=False)
# plt.hist(mean)
# plt.xlabel("Mean fitness")
# plt.ylabel("Count")
# plt.savefig("./results/HISTOGRAM MEAN.pdf")
# plt.close()

# plt.ticklabel_format(useOffset=False)
# plt.hist(best)
# plt.xlabel("Best fitness")
# plt.ylabel("Count")
# plt.savefig("./results/HISTOGRAM BEST.pdf")

def plot(benchmark, bestObjective):
    # Plotting
    file = open("./r0812080.csv")
    data = np.genfromtxt(file, delimiter=",", usecols=[0, 1, 2, 3])


    shutil.copy("./r0812080.csv", "./results/" + str(bestObjective) + '.csv')
    times = data[:, 1]
    meanValues = data[:, 2]
    bestValues = data[:, 3]

    
    # plt.show() 


    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(times, meanValues, label="Mean fitness")
    plt.plot(times, bestValues, label="Best fitness")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Fitness")
    if benchmark == "tour1000" or benchmark == "tour750":
        plt.axis([0, 300, 150000, 200000])
    elif benchmark == "tour500":
        plt.axis([0, 300, 120000, 155000])

    elif benchmark == "tour200":
        plt.axis([0, 300, 35000, 45000])

    elif benchmark == "tour100":
        plt.axis([0, 300, 75000, 95000])
    elif benchmark == "tour50":
        plt.axis([0, 300, 24000, 28000])
    
    # plt.title(f"Iterations = {int(data[-1,0])+1}, total time = {data[-1,1]: .2f} sec")
    plt.savefig("./results/" + benchmark + str(bestObjective) + ".pdf")
    # plt.show()
    plt.close()
     




for i in range(1):
    for benchmark in benchmarks:
        best_fitness = np.inf
        for i in range(1):
            a = r0812080.r0812080()
            meanObjective, bestObjective = a.optimize("./Benchmark/" + benchmark + ".csv")
            print(
                "Finished" + benchmark,
                "Best objective value:",
                bestObjective,
                "Mean objective value:",
                meanObjective,
            )
            if bestObjective < best_fitness:
                best_fitness = bestObjective
                plot(benchmark, bestObjective)


