import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

Wf = 0.5;  Whf = 0.1;  bf = 0
Wi = 0.6;  Whi = 0.2;  bi = 0
Wc = 0.7;  Whc = 0.3;  bc = 0
Wo = 0.8;  Who = 0.4;  bo = 0

Wy = 4;  by = 0

h = 0.0
c = 0.0

inputs = [1.0, 2.0, 3.0, 4.0]

for t in range(len(inputs)):
    x = inputs[t]
    print("Time step", t+1, "- input =", x)

    f = round(sigmoid(Wf * x + Whf * h + bf), 3)
    print("  Forget gate:", f)

    i = round(sigmoid(Wi * x + Whi * h + bi), 3)
    print("  Input gate:", i)

    c_tilde = round(tanh(Wc * x + Whc * h + bc), 3)
    print("  Candidate cell:", c_tilde)

    c = round(f * c + i * c_tilde, 3)
    print("  Cell state:", c)

    o = round(sigmoid(Wo * x + Who * h + bo), 3)
    print("  Output gate:", o)

    h = round(o * tanh(c), 3)
    print("  Hidden state:", h)
    print()

h = 0.949
y = Wy * h + by
print("Hidden state (as per problem sheet): 0.949")
print("Predicted next value:", round(y, 1))
