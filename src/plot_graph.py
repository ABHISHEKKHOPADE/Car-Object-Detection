import matplotlib.pyplot as plt

def load_history(file):
    acc, val = [], []
    with open(file, "r") as f:
        for line in f:
            a, v = line.strip().split(",")
            acc.append(float(a))
            val.append(float(v))
    return acc, val

adam_acc, adam_val = load_history("results/adam.txt")
sgd_acc, sgd_val = load_history("results/sgd.txt")
rms_acc, rms_val = load_history("results/rmsprop.txt")

epochs = range(1, len(adam_acc)+1)

plt.plot(epochs, adam_val, label="Adam")
plt.plot(epochs, sgd_val, label="SGD")
plt.plot(epochs, rms_val, label="RMSprop")

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Optimizer Comparison")
plt.legend()

plt.savefig("results/accuracy_comparison.png")
plt.show()