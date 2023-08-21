import matplotlib.pyplot as plt
import numpy as np
import os

LOSS_PLOT_PATH = os.path.sep.join(["output", "t1.png"])
ACC_PLOT_PATH = os.path.sep.join(["output", "t2.png"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20))
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig(LOSS_PLOT_PATH)

# plot the model training history
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 5))
plt.title("Bounding Box Regression Accuracy on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig(ACC_PLOT_PATH)