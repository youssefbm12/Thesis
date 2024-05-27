import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit
import re


class QAOA:
    # Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
    def __init__(self, graph_g, n_qubits):
        """
        The cost Hamiltonian:
        C_alpha = 1/2 * (1 - Z_j Z_k)
        (j, k) is an edge of the graph
        """
        self.n_qubits = n_qubits
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=1)
        self.dev_train = qml.device("lightning.qubit", wires=self.n_qubits)
        self.n_samples = 100
        self.pauli_z = [[1, 0], [0, -1]]
        self.pauli_z_2 = np.kron(self.pauli_z, self.pauli_z, requires_grad=False)
        self.graph = graph_g 
    
    def costFunc(self, params, quantum_circuit=None, ansatz=''):
        @qml.qnode(self.dev_train)
        def circuit_input(parameters, edge=None):
            i = 0
            for instr, qubits, clbits in quantum_circuit.data:
                name = instr.name.lower()
                if name == "rx":
                    if ansatz == 'all':
                        qml.RX(instr.params[0], wires=qubits[0]._index)
                    else:
                        qml.RX(parameters[i], wires=qubits[0]._index)
                        
                elif name == "ry":
                    if ansatz == 'all':
                        
                        qml.RY(instr.params[0], wires=qubits[0]._index)
                    else:
                        qml.RY(parameters[i], wires=qubits[0]._index)
                        
                elif name == "rz":
                    if ansatz == 'all':
                        qml.RZ(instr.params[0], wires=qubits[0]._index)
                    else:
                        qml.RZ(parameters[i], wires=qubits[0]._index)
                        
                elif name == "h":
                    qml.Hadamard(wires=qubits[0]._index)
                elif name == "cx":
                    qml.CNOT(wires=[qubits[0]._index, qubits[1]._index])

            if edge is None:
                return qml.sample()

            return qml.expval(qml.Hermitian(self.pauli_z_2, wires=edge))
        neg_obj = 0
        for edge in self.graph:
            neg_obj -= 0.5*(1-circuit_input(params, edge))
        return neg_obj

    def getReward(self, params, quantum_circuit=None, ansatz=''):
        return -self.costFunc(params, quantum_circuit, ansatz)
    
    def print_probabilities(self, params, quantum_circuit, ansatz=''):
        @qml.qnode(self.dev_train)
        def probability_circuit(parameters):
            i = 0
            for instr, qubits, clbits in quantum_circuit.data:
                name = instr.name.lower()
                if name in ["rx", "ry", "rz"]:
                    if ansatz == 'all':
                        gate = getattr(qml, name.upper())
                        gate(instr.params[0], wires=qubits[0]._index)
                    else:
                        gate = getattr(qml, name.upper())
                        gate(parameters[i], wires=qubits[0]._index)
                        i += 1  # Increment i only if using parametric gates
                elif name == "h":
                    qml.Hadamard(wires=qubits[0]._index)
                elif name == "cx":
                    qml.CNOT(wires=[qubits[0]._index, qubits[1]._index])

            return qml.probs(wires=range(self.n_qubits))

        probabilities = probability_circuit(params)

        # Construct the probability string if the array is valid
        if probabilities is not None:
            result_str = ""
            num_qubits = self.n_qubits
            for state_index, probability in enumerate(probabilities):
                if probability > 0.000001:  # Include only states with non-zero probabilities
                    state = format(state_index, '0' + str(num_qubits) + 'b')
                    if result_str:  # Not the first term, add a plus
                        result_str += " + "
                    result_str += f"{probability:.1f}|{state}>"
            return result_str

        

    def gradient_descent(self, quantum_circuit):
        opt = qml.AdamOptimizer()
        parameters = get_parameters(quantum_circuit)
        theta = np.array(parameters, requires_grad=True)

        # store the values of the cost function

        def prova(params):
            return self.costFunc(params=params, quantum_circuit=quantum_circuit, ansatz='')

        cost = [prova(theta)]
        
      
        # store the values of the circuit parameter
        angle = [theta]

        max_iterations = 100
        conv_tol = 1e-06  # default -06

        for n in range(max_iterations):
            theta, prev_energy = opt.step_and_cost(prova, theta)
            cost.append(prova(theta))
            angle.append(theta)

            conv = np.abs(cost[-1] - prev_energy)

            if n % 10 == 0:
                print(f"Step = {n},  Cost = {cost[-1]:.8f}")


            if conv <= conv_tol:
                print('Landscape is flat')
                break
        
        print(f" Last Step Cost = {cost[-1]:.8f}")
        print(theta)
        print(self.print_probabilities(params=theta, quantum_circuit=quantum_circuit, ansatz=''))

        return cost
def count_qubit(array_of_tuples):
    # Flatten the array of tuples into a single list of numbers
    flattened_list = [item for sublist in array_of_tuples for item in sublist]
    unique_numbers = set(flattened_list)
    return len(unique_numbers)

def get_parameters(quantum_circuit):
        parameters = []
        # Iterate over all gates in the circuit
        for instr, qargs, cargs in quantum_circuit.data:

            # Extract parameters from gate instructions
            if len(instr.params) > 0:
                parameters.append(instr.params[0])
        return parameters

# graph = [(0, 1), (0, 2),  (2, 3), (1, 4), (2, 4), (0, 5),  (3, 6), (1, 6)]  #  Paper graph
# graph = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)] # Cycle graph
# graph [(0, 1), (0, 3), (2, 5), (4, 1), (4, 5), (6, 3)] # Bipartite graph
# graph [(0, 3), (0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (4, 6)] # Connected graph

graph = [(0, 1), (0, 3), (0, 5), (2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (4, 5), (6, 1), (6, 3), (6, 5)] #Bipartite 
number_of_qubits = count_qubit(graph)
qaoa_class = QAOA(graph_g=graph,n_qubits=number_of_qubits)
# Class works - Implement Gradient Descent on the parameters and check if it return the right solution


#save_in_file.run_and_savepkl()
