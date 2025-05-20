import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm
from scipy.sparse import spmatrix, csr_matrix, identity, kron, block_diag, lil_matrix
from math import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc, PathPatch
from matplotlib.textpath import TextPath

def dsum(a: np.array, b: np.array) -> np.array:
    sum = np.zeros(np.add(a.shape,b.shape), dtype=np.complex128)
    sum[:a.shape[0], :a.shape[1]] = a
    sum[a.shape[0]:, a.shape[1]:] = b
    return sum

def dsum_sparse(a: spmatrix, b: spmatrix) -> spmatrix:
    return block_diag((a, b), format="csr")

def permutation_matrix(perms: list[int]) -> csr_matrix:
    N = len(perms)
    P = lil_matrix((N, N), dtype=np.complex128)
    
    for perm in perms:
        P[perm[0], perm[1]] = 1

    return P.tocsr()

def round_sparse(M: spmatrix, decimals: int = 10) -> spmatrix:
    M = M.copy()
    M.data = np.round(M.data, decimals=decimals)
    M.eliminate_zeros()
    return M

def sparse_is_equal(A: spmatrix, B: spmatrix) -> bool:
    return np.all(A.data == B.data) and np.all(A.indices == B.indices) and np.all(A.indptr == B.indptr)

HADAMARD = (1.0 / np.sqrt(2)) * np.array([
    [1., 1.],
    [1., -1.]
], dtype=np.complex128)
X = np.array([
    [0., 1.],
    [1., 0.]
], dtype=np.complex128)
Y = np.array([
    [0., complex(0, -1)],
    [complex(0, 1), 0.]
], dtype=np.complex128)
Z = np.array([
    [1., 0.],
    [0., -1.]
], dtype=np.complex128)
S = np.array([
    [1., 0.],
    [0., complex(0, 1)]
], dtype=np.complex128)
T = np.array([
    [1., 0.],
    [0., np.exp(complex(0, pi/4.0))]
], dtype=np.complex128)

def dirac(state: np.array, precision: int = 2) -> str:
    basis = ""
    first = True
    for i in range(len(state)):
        qubits = str(bin(i))[2:]
        if len(qubits) < np.log2(len(state)):
            qubits = ("0" * (int(np.log2(len(state))) - len(qubits))) + qubits

        value = ""
        if state[i] == 0:
            continue
        elif state[i].real == 1.0:
            pass
        elif state[i].real == -1.0:
            value = "-"
        elif state[i].imag == 0.0:
            if (not first) and (state.real[i] > 0):
                value += "+ "
            elif (state.real[i] < 0):
                if not first:
                    value += "- "
                else:
                    value += "-"

            value += str(round(abs(float(state.real[i])), precision))
        else:
            if not first:
                value = "+ "
            value += f"("
            value += str(round(abs(float(state.real[i])), precision))

            if (state.imag[i] < 0):
                value += " - "
            else:
                value += " + "

            value += str(round(abs(float(state.imag[i])), precision)) + "i)"

        if not first:
            basis += " "

        first = False

        basis += f"{value}|{qubits}>"

    if basis == "":
        return "0"
    return basis

class Gate:
    is_measurement = False
    def __init__(self, U: np.array, targets: list = [], controls: list = [], n_wires: int = -1, classical_control: int = None, label: str = None):
        self.U = csr_matrix(U)
        self.U.dtype = np.complex128
        self.targets = deepcopy(targets)
        self.controls = deepcopy(controls)
        self.classical_control = classical_control
        self.P = None
        self.M = None

        if label == None:
            if np.array_equal(U, HADAMARD):
                label = "H"
            elif np.array_equal(U, X):
                label = "X"
            elif np.array_equal(U, Y):
                label = "Y"
            elif np.array_equal(U, Z):
                label = "Z"
            elif np.array_equal(U, S):
                label = "S"
            elif np.array_equal(U, T):
                label = "T"

        self.label = label

        if n_wires < 0:
            if len(targets) == 0 and len(controls) != 0:
                self.n_wires = max(controls) + 1
            elif len(controls) == 0 and len(targets) != 0:
                self.n_wires = max(targets) + 1
            else:
                self.n_wires = max(max(targets), max(controls)) + 1
        else:
            self.n_wires = n_wires

        self.build_permutation_matrix()
        self.build_matrix()

    def build_permutation_matrix(self):
        N = 2**self.n_wires
        n_t = len(self.targets)
        n_c = len(self.controls)
        n_e = self.n_wires - n_t - n_c

        perms = []
        for i in range(N):
            sorted_qb = [0] * self.n_wires

            for c in range(n_c):
                bit = (i >> (self.n_wires - 1 - c)) & 1
                sorted_qb[self.n_wires - 1 - self.controls[c]] = bit

            for t in range(n_t):
                bit = (i >> (self.n_wires - 1 - (n_c + t))) & 1
                sorted_qb[self.n_wires - 1 - self.targets[t]] = bit

            used = []
            for t in self.targets:
                used.append(t)
            for c in self.controls:
                used.append(c)

            fill_i = 0
            for j in range(self.n_wires):
                if j not in used:
                    bit = (i >> (self.n_wires - 1 - (n_c + n_t + fill_i))) & 1
                    sorted_qb[self.n_wires - 1 - j] = bit
                    fill_i += 1
                    if fill_i == n_e:
                        break

            value = 0
            for b in sorted_qb:
                value = (value << 1) | b

            perms.append([value, i])

        self.P = permutation_matrix(perms)

    def build_matrix(self) -> None:

        N = 2**self.n_wires
        n_t = len(self.targets)
        n_c = len(self.controls)
        n_e = self.n_wires - n_t - n_c

        U_eff = kron(self.U, identity(2**n_e, format='csr'), format='csr')
        U_eff = dsum_sparse(identity(N - 2**(n_t + n_e), format='csr'), U_eff)

        self.M = round_sparse(self.P @ U_eff @ self.P.T)


class Measurement(Gate):
    is_measurement = True
    def __init__(self, target: int, control: int):
        self.target = target
        self.control = control


class QC:
    def __init__(self, n_wires: int, n_classical_wires: int = 0):
        self.n_wires = n_wires
        self.n_classical_wires = n_classical_wires
        self.gates: list[Gate] = []

    def get_matrix(self) -> csr_matrix:
        for gate in self.gates:
            if gate.is_measurement:
                return None

        N = 2**self.n_wires
        U = identity(N, format='csr')

        for i in range(len(self.gates)):
            if self.gates[i].classical_control != None:
                continue
            U = self.gates[i].M @ U
            U = round_sparse(U)

        return U
    
    def run(self, state: np.array):
        N = 2**self.n_wires
        ensemble = [[1.0, state, [0 for _ in range(self.n_classical_wires)]]]

        for i in range(len(self.gates)):
            new_ensemble = []
            for pure_state in ensemble:
                if self.gates[i].is_measurement == False:
                    if self.gates[i].classical_control != None:
                        if pure_state[2][self.gates[i].classical_control] == 0:
                            new_ensemble.append(pure_state)
                        else:
                            new_ensemble.append([pure_state[0], self.gates[i].M @ pure_state[1], pure_state[2]])
                    else:
                        new_ensemble.append([pure_state[0], self.gates[i].M @ pure_state[1], pure_state[2]])
                else:
                    proj_0 = lil_matrix((N, N))
                    proj_1 = lil_matrix((N, N))

                    for state_i in range(N):
                        if ((state_i >> self.gates[i].target) & 1 == 0):
                            proj_0[state_i, state_i] = 1.0
                        else:
                            proj_1[state_i, state_i] = 1.0

                    mag0 = np.linalg.norm(proj_0 @ pure_state[1])
                    mag1 = np.linalg.norm(proj_1 @ pure_state[1])

                    p0 = np.abs(mag0)**2
                    p1 = np.abs(mag1)**2

                    if self.gates[i].control != None:
                        classical_wires0 = deepcopy(pure_state[2])
                        classical_wires0[self.gates[i].control] = 0
                        classical_wires1 = deepcopy(pure_state[2])
                        classical_wires1[self.gates[i].control] = 1

                        new_ensemble.append([pure_state[0] * p0, (proj_0 @ pure_state[1]) / mag0, classical_wires0])
                        new_ensemble.append([pure_state[0] * p1, (proj_1 @ pure_state[1]) / mag1, classical_wires1])
                    else:
                        new_ensemble.append([pure_state[0] * p0, (proj_0 @ pure_state[1]) / mag0, pure_state[2]])
                        new_ensemble.append([pure_state[0] * p1, (proj_1 @ pure_state[1]) / mag1, pure_state[2]])

                ensemble = new_ensemble


        simplified_ensemble = []
        while len(ensemble) > 0:
            pure_state = deepcopy(ensemble[0])

            for i in range(1, len(ensemble)):
                if np.array_equal(pure_state[1], ensemble[i][1]) and np.array_equal(pure_state[2], ensemble[i][2]): 
                    pure_state[0] += ensemble[i][0]

            simplified_ensemble.append(pure_state)

            ensemble = [left_over for left_over in ensemble if not (np.array_equal(pure_state[1], left_over[1]) and np.array_equal(pure_state[2], left_over[2]))]

        return simplified_ensemble

    def add_loaded_gate(self, gate: Gate) -> None:
        self.gates.append(gate)

    def add_gate(self, U: np.array, targets: list = [], controls: list = [], classical_control: int = None, label: str = None) -> None:
        self.gates.append(Gate(deepcopy(U), targets, controls, self.n_wires, classical_control, label))
    
    def add_measurement(self, target: int, control: int = None) -> None:
        self.gates.append(Measurement(target, control))
        if control != None:
            if control >= self.n_classical_wires:
                self.n_classical_wires = control + 1

    def remove_gate(self, gate_i: int) -> None:
        self.gates.remove(gate_i)

    def render(self, axis_labels = False, fig_scale = 1.0) -> None:
        GATE_SIZE = .5
        GATE_SPACING = 1
        CIRCLE_RADIUS_1 = .06
        CIRCLE_RADIUS_2 = .1
        LINE_WIDTH = 2
        TEXT_SIZE = .6
        CLASSICAL_GAP = .02

        fig, ax = plt.subplots(figsize = (6.4 * fig_scale, 4.8 * fig_scale))
        ax.set_aspect('equal')

        if not axis_labels:
            ax.axis('off')

        w = GATE_SPACING * (1 + len(self.gates))
        for wire_i in range(self.n_wires):
            ax.plot([0, w], [wire_i, wire_i], color="black", linewidth=LINE_WIDTH, zorder=1)

        for wire_i in range(self.n_classical_wires):
            ax.plot([0, w], [-(wire_i + 1) + CLASSICAL_GAP, -(wire_i + 1) + CLASSICAL_GAP], color="black", linewidth=LINE_WIDTH / 2.0, zorder=1)
            ax.plot([0, w], [-(wire_i + 1) - CLASSICAL_GAP, -(wire_i + 1) - CLASSICAL_GAP], color="black", linewidth=LINE_WIDTH / 2.0, zorder=1)

        for gate_i in range(len(self.gates)):
            if (self.gates[gate_i].is_measurement == True):
                x = GATE_SPACING * (1 + gate_i) - GATE_SIZE / 2.
                y = self.gates[gate_i].target - GATE_SIZE / 2.
                h = GATE_SIZE

                if self.gates[gate_i].control != None:
                    ax.plot([x + GATE_SIZE / 2.0 - CLASSICAL_GAP, x + GATE_SIZE / 2.0 - CLASSICAL_GAP], [-(self.gates[gate_i].control + 1), self.gates[gate_i].target], linewidth=LINE_WIDTH / 2.0, color="black", zorder=1)
                    ax.plot([x + GATE_SIZE / 2.0 + CLASSICAL_GAP, x + GATE_SIZE / 2.0 + CLASSICAL_GAP], [-(self.gates[gate_i].control + 1), self.gates[gate_i].target], linewidth=LINE_WIDTH / 2.0, color="black", zorder=1)
                    circle = Circle((x + GATE_SIZE / 2.0, -(self.gates[gate_i].control + 1)), CIRCLE_RADIUS_1, linewidth=LINE_WIDTH, edgecolor='black', facecolor='black', zorder=2)
                    ax.add_patch(circle)

                rect = Rectangle((x, y), GATE_SIZE, h, linewidth=1, edgecolor='black', facecolor=[.5, .5, 1], zorder=2)
                ax.add_patch(rect)
                
                arc = Arc((x + GATE_SIZE / 2.0, self.gates[gate_i].target - .2 * GATE_SIZE), width=0.6 * GATE_SIZE, height=0.6 * GATE_SIZE, angle=0, theta1=0, theta2=180, linewidth=LINE_WIDTH, edgecolor='black', zorder=3)
                ax.add_patch(arc)
                ax.plot([x + GATE_SIZE / 2.0, x + GATE_SIZE / 2.0 + .3 * GATE_SIZE], [self.gates[gate_i].target - .2 * GATE_SIZE, self.gates[gate_i].target + .1 * GATE_SIZE], linewidth=LINE_WIDTH, color="black", zorder=5)

                # text = f"?"
                # tp = TextPath((0, 0), text, size = GATE_SIZE)
                # bbox = tp.get_extents()

                # scale = (TEXT_SIZE * GATE_SIZE) / max(bbox.width, bbox.height)

                # x_text = x + GATE_SIZE / 2.0
                # y_text = y + GATE_SIZE / 2.0

                # text_patch = PathPatch(tp, transform=plt.matplotlib.transforms.Affine2D().translate(-bbox.width / 2.0, -bbox.height / 2.0).scale(scale).translate(x_text, y_text) + ax.transData, color='black', lw=0, zorder=3)

                # ax.add_patch(text_patch)

            else:
                top = max(self.gates[gate_i].targets)
                bot = min(self.gates[gate_i].targets)

                x = GATE_SPACING * (1 + gate_i) - GATE_SIZE / 2.0
                y = bot - GATE_SIZE / 2.0
                h = (top - bot) + GATE_SIZE

                for c in self.gates[gate_i].controls:
                    ax.plot([x + GATE_SIZE / 2.0, x + GATE_SIZE / 2.0], [c, (top + bot) / 2.0], linewidth=LINE_WIDTH, color="black", zorder=1)
                    circle = Circle((x + GATE_SIZE / 2.0, c), CIRCLE_RADIUS_1, linewidth=LINE_WIDTH, edgecolor='black', facecolor='black', zorder=2)
                    ax.add_patch(circle)

                if self.gates[gate_i].classical_control != None:
                    ax.plot([x + GATE_SIZE / 2.0 - CLASSICAL_GAP, x + GATE_SIZE / 2.0 - CLASSICAL_GAP], [-(self.gates[gate_i].classical_control + 1), (top + bot) / 2.0], linewidth=LINE_WIDTH / 2.0, color="black", zorder=1)
                    ax.plot([x + GATE_SIZE / 2.0 + CLASSICAL_GAP, x + GATE_SIZE / 2.0 + CLASSICAL_GAP], [-(self.gates[gate_i].classical_control + 1), (top + bot) / 2.0], linewidth=LINE_WIDTH / 2.0, color="black", zorder=1)
                    circle = Circle((x + GATE_SIZE / 2.0, -(self.gates[gate_i].classical_control + 1)), CIRCLE_RADIUS_1, linewidth=LINE_WIDTH, edgecolor='black', facecolor='black', zorder=2)
                    ax.add_patch(circle)

                if (self.gates[gate_i].label == "X") and (len(self.gates[gate_i].controls) > 0) and (self.gates[gate_i].is_measurement == False):
                    circle = Circle((x + GATE_SIZE / 2.0, self.gates[gate_i].targets[0]), CIRCLE_RADIUS_2, linewidth=LINE_WIDTH, edgecolor='black', facecolor='none', zorder=2)
                    ax.add_patch(circle)

                    ax.plot([x + GATE_SIZE / 2.0, x + GATE_SIZE / 2.0], [self.gates[gate_i].targets[0] - .9*CIRCLE_RADIUS_2, self.gates[gate_i].targets[0] + .9*CIRCLE_RADIUS_2], linewidth=LINE_WIDTH, color="black", zorder=3)

                else:
                    rect = Rectangle((x, y), GATE_SIZE, h, linewidth=1, edgecolor='black', facecolor='lightgray', zorder=2)
                    ax.add_patch(rect)

                    text = f"${self.gates[gate_i].label}$"
                    tp = TextPath((0, 0), text, size = GATE_SIZE)
                    bbox = tp.get_extents()

                    scale = (TEXT_SIZE * GATE_SIZE) / max(bbox.width, bbox.height)

                    x_text = x + GATE_SIZE / 2.0
                    y_text = y + GATE_SIZE / 2.0

                    text_patch = PathPatch(tp, transform=plt.matplotlib.transforms.Affine2D().translate(-bbox.width / 2.0, -bbox.height / 2.0).scale(scale).translate(x_text, y_text) + ax.transData, color='black', lw=0, zorder=3)

                    ax.add_patch(text_patch)

def simplify_ensemble(ensemble: list[list]) -> list[list]:
    simplified_ensemble = []
    orig = deepcopy(ensemble)

    while len(orig) > 0:
        pure_state = deepcopy(orig[0])

        for i in range(1, len(orig)):
            if np.array_equal(pure_state[1], orig[i][1]): 
                pure_state[0] += orig[i][0]

        simplified_ensemble.append(pure_state)
        orig = [left_over for left_over in orig if not np.array_equal(pure_state[1], left_over[1])]

    return simplified_ensemble

def remove_classical(ensemble: list[list]) -> list[list]:
    output_ensemble = []
    orig = deepcopy(ensemble)

    for pure_state in orig:
        output_ensemble.append([pure_state[0], pure_state[1]])

    return simplify_ensemble(output_ensemble)

def qubit_state(state: np.array, qubit: int) -> np.array:
    output_state = np.zeros((2, 1), dtype=np.complex128)
    for i in range(len(state)):
        if ((i >> qubit) & 1 == 0):
            output_state[0] += state[i]
        else:
            output_state[1] += state[i]

    return output_state

def qubit_ensemble(ensemble: list[list], qubit: int) -> list[list]:
    output_ensemble = []
    orig = deepcopy(ensemble)
    for pure_state in orig:
        pure_state[1] = qubit_state(pure_state[1], qubit)
        output_ensemble.append(pure_state)

    return simplify_ensemble(output_ensemble)

def draw() -> None:
    plt.show()