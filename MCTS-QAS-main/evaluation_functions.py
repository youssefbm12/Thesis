from problems.combinatorial import qaoa_class
from problems.vqls import vqls_demo, vqls_paper
from problems.combinatorial import get_parameters
import heapq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile


def qaoa(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = qaoa_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        params = get_parameters(quantum_circuit)
        return problem.costFunc(params,quantum_circuit=quantum_circuit)
    else:
        return problem.getReward(params=[0.1,0.2],quantum_circuit=quantum_circuit)
        



