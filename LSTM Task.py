import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

Wf = 0.7;  Uf = 0.5;  bf = 0.1
Wi = 0.6;  Ui = 0.4;  bi = 0.1
Wc = 0.5;  Uc = 0.6;  bc = 0.2
Wo = 0.8;  Uo = 0.3;  bo = 0.1

Wy = 1.2;  by = 0.5

h = 0.0
c = 0.0

inputs = [1.0, 2.0, 3.0, 4.0]

for t in range(len(inputs)):
    x = inputs[t]
    print("Time step", t+1, "- input =", x)

    f = sigmoid(Wf * x + Uf * h + bf)
    print("  Forget gate:", round(f, 4))

    i = sigmoid(Wi * x + Ui * h + bi)
    print("  Input gate:", round(i, 4))

    c_tilde = tanh(Wc * x + Uc * h + bc)
    print("  Candidate cell:", round(c_tilde, 4))

    c = f * c + i * c_tilde
    print("  Cell state:", round(c, 4))

    o = sigmoid(Wo * x + Uo * h + bo)
    print("  Output gate:", round(o, 4))

    h = o * tanh(c)
    print("  Hidden state:", round(h, 4))
    print()

y = Wy * h + by
print("Predicted next value:", round(y, 4))
