#tempCodeRunnerFile.pyimport numpy as np
import matplotlib.pyplot as plt
# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)
# Create a plot
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()
save_path = 'sine_wave.png'
