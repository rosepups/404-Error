# 404-Error
Files for the quantum chemistry challenge part of Womanium 

1. Firstly need to create the Hatree Fock initial state. In this code the LiH molecule was created with a bond distance of 2.5 angstrom in the single state with no charge

`from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,)
molecule = Molecule(
    geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 2.5]]], charge=0, multiplicity=1)
driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)`
    
 2.
 The below code prints out the Hamiltonian from above into terms of fermionic operators. 
 
 `es_problem = ElectronicStructureProblem(driver)
second_q_op = es_problem.second_q_ops()
print(second_q_op["ElectronicEnergy"])`

Output:

![image](https://user-images.githubusercontent.com/53739684/186053064-24b2a871-898c-4a15-b661-3f62649dd6d5.png)

3. To run on the quantum computer we need to change it from fermionic operators to spin operators. The Jordan-Wigner Mapper was used. 

`qubit_converter = QubitConverter(mapper=JordanWignerMapper())
qubit_op = qubit_converter.convert(second_q_op["ElectronicEnergy"])
print(qubit_op)`

4. To decrease the number of qubits from 12 to 10, the Parity Mapper is used. This is able to remove 2 qubits by exploting known symmetries arising from the mapping

`qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
qubit_op = qubit_converter.convert(
    second_q_op["ElectronicEnergy"], num_particles=es_problem.num_particles)
print(qubit_op)`

5. Then use a set of optimisers for comparison. The iterations were increased until it was shown that all the optimisers converged. 

`optimizers = [COBYLA(maxiter=500), L_BFGS_B(maxiter=300), SLSQP(maxiter=300)]
converge_cnts = np.empty([len(optimizers)], dtype=object)
converge_vals = np.empty([len(optimizers)], dtype=object)
for i, optimizer in enumerate(optimizers):
    print('\rOptimizer: {}        '.format(type(optimizer).__name__), end='')
    algorithm_globals.random_seed = 50
    ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
 counts = []
 values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)`
  
   `vqe = VQE(ansatz, optimizer, callback=store_intermediate_result,
              quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator')))
    result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
    converge_cnts[i] = np.asarray(counts)
    converge_vals[i] = np.asarray(values)
print('\rOptimization complete      ');`

Output:
![image](https://user-images.githubusercontent.com/53739684/186061698-c0efa040-9981-49a5-8816-4afbbde93201.png)

6. NumPyMinimumEigensolver is used to computer a reference value of the LiH

`npme = NumPyMinimumEigensolver()
result = npme.compute_minimum_eigenvalue(operator=H2_op)
ref_value = result.eigenvalue.real
print(f'Reference value: {ref_value:.5f}')`

Output: 
![image](https://user-images.githubusercontent.com/53739684/186062227-e08364d7-ca05-43e4-aa59-f7f38a0ad741.png)

7. The difference between the exact solution and the energy convergence using VQE can then be plotted
`pylab.rcParams['figure.figsize'] = (12, 8)
for i, optimizer in enumerate(optimizers):
    pylab.plot(converge_cnts[i], abs(ref_value - converge_vals[i]), label=type(optimizer).__name__)
pylab.xlabel('Eval count')
pylab.ylabel('Energy difference from solution reference value')
pylab.title('Energy convergence for various optimizers')
pylab.yscale('log')
pylab.legend(loc='upper right');`

Output:
![image](https://user-images.githubusercontent.com/53739684/186062382-135d6701-083c-411a-bc0f-5ae2be336bd0.png)


The next step was using the Qiskit Aer to run a simulation with noise. For a comparison we firstly looked at the results without noise and the looked at the results with noise.

`seed = 170
iterations = 125
algorithm_globals.random_seed = seed
backend = Aer.get_backend('aer_simulator')
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed) 

counts = []
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = SPSA(maxiter=iterations)
vqe = VQE(ansatz, optimizer=spsa, callback=store_intermediate_result, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
print(f'VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}')
print(f'Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}')`

Output:
![image](https://user-images.githubusercontent.com/53739684/186062765-695cb65c-6937-4bf3-a94b-48b214c582cc.png)

The energy values are then graphed during the convergence as shown below
![image](https://user-images.githubusercontent.com/53739684/186062884-b1b4b955-19d9-4c5e-b70c-d84f829346e9.png)

The next step is adding the noise. Mock backends were used in this case as we were not able to access the quantum computers above 7 qubits. However we wanted to see what the result could be if used those backends from real noise data.In this case the FakeSydney backend was used.

`import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeSydney`

`device_backend = FakeSydney()`

`backend = Aer.get_backend('aer_simulator')
counts1 = []
values1 = []
noise_model = None
device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates`

`print(noise_model)
print()`

`algorithm_globals.random_seed = seed
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                     coupling_map=coupling_map, noise_model=noise_model,)`

`def store_intermediate_result1(eval_count, parameters, mean, std):
    counts1.append(eval_count)
    values1.append(mean)`

`var_form = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = SPSA(maxiter=iterations)
vqe = VQE(ansatz, optimizer=spsa, callback=store_intermediate_result1, quantum_instance=qi)
result1 = vqe.compute_minimum_eigenvalue(operator=qubit_op)
print(f'VQE on Aer qasm simulator (with noise): {result1.eigenvalue.real:.10f}')
print(f'Delta from reference energy value is {(result1.eigenvalue.real - ref_value):.10f}')` 


Output:
![image](https://user-images.githubusercontent.com/53739684/186063634-945594e4-9479-4516-90ba-123b0bcf0063.png)

The graph is then drawn showing convergence with noise

![image](https://user-images.githubusercontent.com/53739684/186063768-d5b843f1-5c27-4477-8fef-bb4400ebfde7.png)

# Use real BackEnds
The next step after that was using actual backends. Because the backends were limited to only 5 or 7 qubits the essentialUC was used as the ansatz. Once again the LiH was set up with a bond distance of 2.5. 

'bond_distance = 2.5  # in Angstrom``


`molecule = Molecule(
    geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, bond_distance]]], charge=0, multiplicity=1)`



`driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)``
`properties = driver.run()`

`particle_number = properties.get_property(ParticleNumber)`

`active_space_trafo = ActiveSpaceTransformer(
    num_electrons=particle_number.num_particles, num_molecular_orbitals=3)`


`problem = ElectronicStructureProblem(driver, transformers=[active_space_trafo])`


`qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)'



