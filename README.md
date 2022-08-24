## 404-Error

#Team Members:


**Jessica Jein White:**

email: jessica.white3@griffithuni.edu.au

GitHub: jein55

Discord ID: doublej#8638



**Rose Manakil**

email: rose.manakil@griffithuni.edu.au

GitHub: rosepups

Discord ID: kuro_chan2356#6896


**Peter Collins**

email: peetahjohn@gmail.com

GitHub ID: PeterJC98

Discord ID: PeterJC#5601


**Pitch Presenter: Peter Collins**

# Creating Li-H Hamiltonian

Firstly we need to create the Hatree Fock initial state. In this code the LiH molecule was created with a bond distance of 2.5 angstrom in the single state with no charge

```
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,)
molecule = Molecule(
    geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 2.5]]], charge=0, multiplicity=1)
driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)
   ```
 The below code prints out the Hamiltonian from above into terms of fermionic operators. 
 
 `es_problem = ElectronicStructureProblem(driver)
second_q_op = es_problem.second_q_ops()
print(second_q_op["ElectronicEnergy"])`

Output:

![image](https://user-images.githubusercontent.com/53739684/186053064-24b2a871-898c-4a15-b661-3f62649dd6d5.png)


To run on the quantum computer we need to change it from fermionic operators to spin operators. The Jordan-Wigner Mapper was used. 

```
qubit_converter = QubitConverter(mapper=JordanWignerMapper())
qubit_op = qubit_converter.convert(second_q_op["ElectronicEnergy"])
print(qubit_op)
```

To decrease the number of qubits from 12 to 10, the Parity Mapper is used. This is able to remove 2 qubits by exploiting known symmetries arising from the mapping
```
qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
qubit_op = qubit_converter.convert(
    second_q_op["ElectronicEnergy"], num_particles=es_problem.num_particles)
print(qubit_op)
```

# Hamiltonian Optimization

We compared the optimizers available on qiskit.algorithms.optimizers. The optimizer iterations were increased until it was shown that all the optimisers converged. Many optimisers were investigated including BOBYQA, IMFIL, SNOBFIT, COBYLA, L_BFGS_B, SLSQP, CG, ADAM

Is this code COBYLA, L_BFGS_B, SLSQP and CG were compared. The max iterations was changed until clear convergence could be seen. The other optimisers gained undesirable results therefore were not 

```
optimizers = [COBYLA(maxiter=4000), L_BFGS_B(maxiter=4000), SLSQP(maxiter=1000), CG(maxiter=200)]
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
        values.append(mean)
  
    vqe = VQE(ansatz, optimizer, callback=store_intermediate_result,
              quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator')))
    result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
    converge_cnts[i] = np.asarray(counts)
    converge_vals[i] = np.asarray(values)
print('\rOptimization complete      ');
```

Output:

![image](https://user-images.githubusercontent.com/53739684/186298044-f9b0d646-ac4f-4ad1-a14f-eb3593a13fa2.png)


NumPyMinimumEigensolver is used to computer a reference value of the LiH (this uses our Hamiltonian which was made in the first step)

```
npme = NumPyMinimumEigensolver()
result = npme.compute_minimum_eigenvalue(operator=qubit_op)
ref_value = result.eigenvalue.real
print(f'Reference value: {ref_value:.5f}')
```

Output: 
![image](https://user-images.githubusercontent.com/53739684/186062227-e08364d7-ca05-43e4-aa59-f7f38a0ad741.png)

7. The difference between the exact solution and the energy convergence using VQE can then be plotted
```
pylab.rcParams['figure.figsize'] = (12, 8)
for i, optimizer in enumerate(optimizers):
    pylab.plot(converge_cnts[i], abs(ref_value - converge_vals[i]), label=type(optimizer).__name__)
pylab.xlabel('Eval count')
pylab.ylabel('Energy difference from solution reference value')
pylab.title('Energy convergence for various optimizers')
pylab.yscale('log')
pylab.legend(loc='upper right');
```

Output:
This graph shows the different from the reference value. It can be seem that COBYLA(blue) although converged early did not gain the results closest to the reference. In fact the CG (red) was found to gain the best results with the smallest difference. It should be noted however that COBYLA did take a long time (4000 iterations) to reach that covergence. 

![image](https://user-images.githubusercontent.com/53739684/186298054-19e34ad3-27c6-4a8f-92d0-4e4a94d86556.png)

# Noise

The next step was using the Qiskit Aer to run a simulation with noise on the Hamiltonian. For a comparison we firstly looked at the results without noise and then looked at the results with noise.
```
seed = 170
iterations = 4000
algorithm_globals.random_seed = seed
backend = Aer.get_backend('aer_simulator')
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed) 

counts = []
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

ansatz = EfficientSU2(4, su2_gates=['rx', 'cx'], entanglement='circular', reps=1) 
cobyla = COBYLA(maxiter=iterations)
vqe = VQE(ansatz, optimizer=cobyla, callback=store_intermediate_result, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=qubit_op)
print(f'VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}')
print(f'Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}')
```

Output:
![image](https://user-images.githubusercontent.com/53739684/186299171-35bd6533-ea60-4df7-92d3-5c5413f5837e.png)


The energy values are then graphed during the convergence as shown below
![image](https://user-images.githubusercontent.com/53739684/186299226-15886d6e-466c-4c53-83f7-1ad68cbfe7ee.png)


The next step is adding the noise. Mock backends were used in this case as we were not able to access the quantum computers above 7 qubits. However we wanted to see what the result could be if used those backends from real noise data. In this case the FakeSydney and FakeCairo backend were used.

```
iimport os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeCairo

device_backend = FakeCairo()

backend = Aer.get_backend('aer_simulator')
counts1 = []
values1 = []
noise_model = None
device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates

print(noise_model)
print()

algorithm_globals.random_seed = seed
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                     coupling_map=coupling_map, noise_model=noise_model,)

def store_intermediate_result1(eval_count, parameters, mean, std):
    counts1.append(eval_count)
    values1.append(mean)

var_form = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
cobyla = COBYLA(maxiter=iterations)
vqe = VQE(ansatz, optimizer=cobyla, callback=store_intermediate_result1, quantum_instance=qi)
result1 = vqe.compute_minimum_eigenvalue(operator=qubit_op)
print(f'VQE on Aer qasm simulator (with noise): {result1.eigenvalue.real:.10f}')
print(f'Delta from reference energy value is {(result1.eigenvalue.real - ref_value):.10f}')
```


Output:
![image](https://user-images.githubusercontent.com/53739684/186299652-1313127e-d88a-4d36-bf13-c4b44ab8cb77.png)


The graph is then drawn showing convergence with noise

![image](https://user-images.githubusercontent.com/53739684/186299603-a1278691-e26d-4e62-a151-9f7e96a18bbf.png)


# Use real BackEnds
The next step after that was using actual backends. Because the backends were limited to only 5 or 7 qubits we had to use another method which doesn't consider our Hartree Fock operator. Instead we just used the EssentialSU2 and managed to reduce it to 4 qubits so it could be run on all the free quantum computers. Once again the LiH was set up with a bond distance of 2.5. 

Therefore we went through the process again of testing the optimisers to the reference value. This is because the reference value changed in this case as our Hamiltonian operator is not considered. It was found that SPSA gained the best results in this regard and therefore was used as the main operator.

```
bond_distance = 2.5  # in Angstrom


molecule = Molecule(
    geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, bond_distance]]], charge=0, multiplicity=1)


driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)
properties = driver.run()

particle_number = properties.get_property(ParticleNumber)

active_space_trafo = ActiveSpaceTransformer(
    num_electrons=particle_number.num_particles, num_molecular_orbitals=3)


problem = ElectronicStructureProblem(driver, transformers=[active_space_trafo])


qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
```


Numpy was used to calculate the reference result.
```
import numpy as np

target_energy = np.real(np_result.eigenenergies + np_result.nuclear_repulsion_energy)[0]
print("Energy:", target_energy)
```

Output: ![image](https://user-images.githubusercontent.com/53739684/186067264-80dcf412-c034-476b-99c8-d3dee92f0326.png)


The ansatz was then created using the circuit library EfficientSU2. Different ansatz were investigated including TwoLocal and PauliGate (refer to _Looking at different ansatz and the impact of results on local VQE_)

```
ansatz = EfficientSU2(4, su2_gates=['rx', 'cx'], entanglement='circular', reps=1)
ansatz.decompose().draw("mpl", style="iqx")
```
![image](https://user-images.githubusercontent.com/53739684/186074269-3dd68906-8b5a-4c7d-842c-db299b238b89.png)
![image](https://user-images.githubusercontent.com/53739684/186074287-b911134d-b0e7-473b-8d2c-d7cf92d0efb6.png)

Output:



Here the Unitary Coupled Cluster (UCC) is used. It is in a factory form as it has shown to have fast initiazation of VQE in a chemistry standard. 
```
from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory

quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
vqe_solver = VQEUCCFactory(quantum_instance=quantum_instance)
```

Then optimised using the SPSA

```
from qiskit.algorithms.optimizers import SPSA

optimizer = SPSA(maxiter=100)

np.random.seed(5)  # fix seed for reproducibility
initial_point = np.random.random(ansatz.num_parameters)

Used the local simulator to run VQE

from qiskit.providers.basicaer import QasmSimulatorPy  # local simulator
from qiskit.algorithms import VQE

local_vqe = VQE(
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=initial_point,
    quantum_instance=QasmSimulatorPy(),
)

local_vqe_groundstate_solver = GroundStateEigensolver(qubit_converter, local_vqe)

local_vqe_result = local_vqe_groundstate_solver.solve(problem)

print(
    "Energy:",
    np.real(local_vqe_result.eigenenergies + local_vqe_result.nuclear_repulsion_energy)[0],)
```
    
 Output:
 ![image](https://user-images.githubusercontent.com/53739684/186069130-f057b72a-26a9-4b02-b4a2-62b24dc07dcb.png)
    
Now the code is ran on a real backend. This code was run on 5 different IBM backends, and at this time the ibmq_belem was the closest to the reference value (refer to next section _Comparing different BackEnds with the NumPy reference value_ for more)
    
```
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')  # replace by your runtime provider

backend = provider.get_backend("ibm_oslo")  # select a backend that supports the runtime

from qiskit_nature.runtime import VQEClient
runtime_vqe = VQEClient(
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=initial_point,
    provider=provider,
    backend=backend,
    shots=1024,
    measurement_error_mitigation=True,
)  # use a complete measurement fitter for error mitigation
```


The job was then submitted:

`runtime_vqe_groundstate_solver = GroundStateEigensolver(qubit_converter, runtime_vqe)
runtime_vqe_result = runtime_vqe_groundstate_solver.solve(problem)`

Output:

ibmq_belem: ![Belm-EssentialSU2_2_ANSATZ--SPSA-100](https://user-images.githubusercontent.com/111412305/186310360-be2e5de0-45c5-4441-ae15-f2f702451c13.png)


# Comparing different BackEnds with the NumPy reference value

These backends were investigated. It is interesting to see that Belem gained the best results to the reference value. We noticed that this would change throughout the day due to the error of the CNOTs changing in each of the quantum backends throughout. Therefore we can say at this particular time Belem was the best, however this has be known to change with time (in previous runs Nairobi was better).

Reference Value:

![image](https://user-images.githubusercontent.com/53739684/186071772-d50d9cb0-7c50-4ba6-8a0d-5a7d3ac764fa.png)

ibm_nairobi: ![Nairobi-EssentialSU2_2_ANSATZ--SPSA-100](https://user-images.githubusercontent.com/111412305/186310399-e6f8cb38-ab0e-43ef-ab6d-77606742a1e2.png)

ibmq_belem: ![Belm-EssentialSU2_2_ANSATZ--SPSA-100](https://user-images.githubusercontent.com/111412305/186310360-be2e5de0-45c5-4441-ae15-f2f702451c13.png)

ibm_oslo: ![Oslo-EssentialSU2_2_ANSATZ--SPSA-100](https://user-images.githubusercontent.com/111412305/186310486-6bfa7156-4e44-46ad-9f3c-969d952e63c4.png)

ibmq_quito: ![Quito-EssentialSU2_2_ANSATZ--SPSA-100](https://user-images.githubusercontent.com/111412305/186310545-7c8c406b-1c47-4a62-8a1f-e30b0355945b.png)

ibmq_lima: ![Lima-EssentialSU2_2_ANSATZ--SPSA-100](https://user-images.githubusercontent.com/111412305/186310590-d1a40b7f-13b5-43ad-891c-c2f1643f9d67.png)


# Looking at different ansatz and the impact of results on local VQE

Different ansatz values were investigated against the reference value. It was found that the EssentialSU2_2 gained the best results out of all the ansatz investigated in this study. It is interesting to see the impact of gates on the final value.

Reference value: 
![image](https://user-images.githubusercontent.com/53739684/186073639-23db5e35-d422-477b-8a9d-c0905fd60ed1.png)

RealAmplitude1:

`from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import EvolvedOperatorAnsatz`

`ansatz = RealAmplitudes(4, reps=2)
ansatz.decompose().draw("mpl", style="iqx")`

![image](https://user-images.githubusercontent.com/53739684/186073972-82074235-0acc-46e5-b349-78491b07b38a.png)
![image](https://user-images.githubusercontent.com/53739684/186074016-150587dd-20f0-415b-b8cf-14a2bbede2ac.png)

Real Amplitude2:
`from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import RealAmplitudes`


`#ansatz = EfficientSU2(4, su2_gates=['ry','cx', 'cz'], entanglement='circular', reps=1)
ansatz = RealAmplitudes(4, reps=1, entanglement='circular', insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")`

![image](https://user-images.githubusercontent.com/53739684/186073749-cc6c20fe-53ad-40d8-b247-c4017a5de44d.png)
![image](https://user-images.githubusercontent.com/53739684/186073775-d07cb163-c3e2-478d-a0ff-6043a8143649.png)


EssentialSU2_1:
`ansatz = EfficientSU2(4, su2_gates=['rx', 'y'], entanglement='circular', reps=1)
ansatz.decompose().draw("mpl", style="iqx")`

![image](https://user-images.githubusercontent.com/53739684/186074196-6f5cb097-ad0f-4345-a65f-e14ed1c0a705.png)
![image](https://user-images.githubusercontent.com/53739684/186074223-23231628-9919-46df-8297-cf259f87fb54.png)

EssentialSU2_2:
`ansatz = EfficientSU2(4, su2_gates=['rx', 'cx'], entanglement='circular', reps=1)
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074269-3dd68906-8b5a-4c7d-842c-db299b238b89.png)
![image](https://user-images.githubusercontent.com/53739684/186074287-b911134d-b0e7-473b-8d2c-d7cf92d0efb6.png)

EssentialSU2_3:
`ansatz = EfficientSU2(4, su2_gates=['rz','ry', 'cx'], entanglement='circular', reps=1)
ansatz.decompose().draw("mpl", style="iqx")'
![image](https://user-images.githubusercontent.com/53739684/186074329-c76dbee7-20a6-4a6e-9358-897999bed8d4.png)
![image](https://user-images.githubusercontent.com/53739684/186074352-466c3db6-caac-4d08-a552-b27bea6c9ca8.png)

EssentialSU2_4:
`ansatz = EfficientSU2(4, su2_gates=['ry','cx', 'cz'], entanglement='circular', reps=1)
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074394-a7a46e52-728a-4fb5-975a-e7f726b86a69.png)
![image](https://user-images.githubusercontent.com/53739684/186074414-a640a80a-226f-43c4-8eb5-ed5377ee38d7.png)

TwoLocal_1:
`ansatz = TwoLocal(4, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074479-55783901-cf3e-46d8-bbdd-42518e60d798.png)
![image](https://user-images.githubusercontent.com/53739684/186074495-caaef07d-7498-4484-ac36-80dd7b692ba8.png)

TwoLocal_2:
`ansatz=TwoLocal(4, ['ry','rz'], 'cz', 'full', reps=1, insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074536-bf0bed1a-78d6-4e82-b2c3-0acd92b757cd.png)
![image](https://user-images.githubusercontent.com/53739684/186074553-8cab9e71-31f9-4ebf-a186-f00fed0b902b.png)

TwoLocal_3:
`ansatz=TwoLocal(4, ['ry','rz'], 'cz', 'linear', reps=1, insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074596-b3a0c529-1814-47de-8413-04ac4f4e7b1e.png)
![image](https://user-images.githubusercontent.com/53739684/186074613-09ba3c57-a283-49cc-bf39-b10c483929c8.png)


TwoLocal_4:
`ansatz=TwoLocal(4, ['ry','cx'], 'cz', 'circular', reps=1, insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074672-2fd71516-05e3-4462-a890-4184ec962819.png)
![image](https://user-images.githubusercontent.com/53739684/186074780-cc0a9e99-ad3a-4f18-9dab-8eedeb2fa2e4.png)

TwoLocal_5:
'ansatz=TwoLocal(4, ['ry','rx','rz'], 'cz', 'circular', reps=1, insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")'
![image](https://user-images.githubusercontent.com/53739684/186074873-1ca1e437-0a7d-40b8-85d0-a206e6cf61c8.png)
![image](https://user-images.githubusercontent.com/53739684/186074888-a6dfed21-1274-4681-aa76-a16dfd831bf6.png)

PauliTwo:
`ansatz=PauliTwoDesign(num_qubits=4, reps=2, seed=None, insert_barriers=True, name='PauliTwoDesign')
ansatz.decompose().draw("mpl", style="iqx")`
![image](https://user-images.githubusercontent.com/53739684/186074959-0970e932-ceca-43e9-9a52-60d646a13297.png)
![image](https://user-images.githubusercontent.com/53739684/186074976-ddfb1ac0-a0eb-4a88-925c-407a127bd1f3.png)

`# Ansatz Circuit Optimization
Like with the Hamiltonian Optimizers different optimizers were tested for the Ansatz circuit, and found to be different then the hamiltonian optimizers.
We compared the optimizers available on qiskit.algorithms.optimizers, all with 100 iterations. Many optimisers were investigated including COBYLA, L_BFGS_B, SLSQP, CG, ADAM.

Again the refernce value is: ![image](https://user-images.githubusercontent.com/53739684/186071772-d50d9cb0-7c50-4ba6-8a0d-5a7d3ac764fa.png)

The optimizers that estimated closest to the refernce value were considered, which were:

COBYLA: ![COBYLA-100](https://user-images.githubusercontent.com/111412305/186312146-ef3ed60d-a468-404d-80b3-f266a3d527a7.png)

SPSA: ![SPSA-100](https://user-images.githubusercontent.com/111412305/186312187-113d2bbe-1478-49e2-a6fd-b098a5e114ee.png)


Others weren't as successful, as seen below:

SLSQP: ![SLSQP-100](https://user-images.githubusercontent.com/111412305/186312771-5ba40fe3-7358-4dbe-ae93-a0d50c35dcba.png)

L_BFGS_B: ![L_BFGS_B-100](https://user-images.githubusercontent.com/111412305/186312888-aae8c492-d499-4a59-b320-a698911f10b1.png)

CG: ![CG-100](https://user-images.githubusercontent.com/111412305/186313101-249b91a3-82fe-4c28-9d4a-2db025e69fa8.png)

ADAM: ![ADAM-100](https://user-images.githubusercontent.com/111412305/186313284-5ba29e54-de14-48a4-8ef3-66cab8447d25.png)







