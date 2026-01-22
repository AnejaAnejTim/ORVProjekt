import matplotlib
matplotlib.use("QtAgg")
import serial
import time
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

SERIAL_PORT = ("COM4")
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
time.sleep(2)

print("Connected to STM32 CDC device.")
print("Waiting for data...\n")

temps = deque(maxlen=200)
times = deque(maxlen=200)
start_time = time.time()

temp_pattern = re.compile(r"(-?\d+(?:.\d+)?)")

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_title("STM32 Temperature")
ax.set_xlabel("Time (s)")
ax.setylabel("Temperature (°C)")
ax.grid(True)

def update(frame):
    for _ in range(20):
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw:
            break
        match = temp_pattern.search(raw)
        if match:
            temp = float(match.group(1))
            t = time.time() - start_time
            temps.append(temp)
            times.append(t)

            line.set_data(times, temps)
            ax.relim()
            ax.autoscale_view()
            print(raw)

    return line,

ani = animation.FuncAnimation(
    fig,
    update,
    interval=200,
    cache_frame_data=False
)

try:
    plt.show(block=True)
finally:
    ser.close()
