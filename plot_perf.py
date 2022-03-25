import matplotlib.pyplot as plt

x = [
    "GPU baseline",
    "GPU\n+ branchless",
    "GPU\n+ branchless + float4", 
    "GPU\n+ branchless + float4\n+ register",
    "GPU\n+ branchless + float4\n+ register + shared"
]
time = [13, 11, 6, 0.8, 0.6]
fps = [80, 90, 160, 1240, 1700]

fig, ax1 = plt.subplots() 
ax1.bar(x, time, color="tab:blue")
ax1.set_ylabel("Time per step (ms)")

ax2 = ax1.twinx()
ax2.plot(x, fps, color="tab:orange")
ax2.set_ylabel("Steps per second")
ax2.set_ylim([0, 1400])

plt.show()