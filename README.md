## **CR3BPNormalForms.jl**

A Julia package for performing Birkhoff Normal Form expansions and coordinate transformations in the Circular Restricted Three-Body Problem (CR3BP). This tool allows for the analysis of dynamics near equilibrium points by decoupling the system into oscillatory and hyperbolic modes. 

This is similar to what is done when the dynamics are linearized about an equilibrium point. The difference is that instead of ignoring the higher order terms, we take them into account. This gives a far more physical system to create trajectories from.

---

### **Directory Structure**
```text
CR3BPNormalForms/
├── src/
├── examples/
├── Project.toml
└── README.md
```

---

### **Libraries Used**
This package relies on the following Julia libraries:
* `LinearAlgebra`: Matrix operations and eigen decompositions.
* `ForwardDiff`: Automatic differentiation for Jacobian computations.
* `TaylorSeries`: Symbolic-like manipulation of high-order Taylor polynomials.
* `JLD2`: Saving and loading pre-computed Normal Form data.
* `DifferentialEquations`: Numerical integration of Lie series transformations.

---

### **Core Functionality**
* **Normal Form Generation**: High-order Taylor series expansion of the CR3BP Hamiltonian near $L_1, L_2,$ and $L_3$.
* **Coordinate Transformations**: Mapping between:
    * **RTB**: CR3BP rotating frame.
    * **NF**: Normal form frame.
    * **AA**: Action-angle representation.

---

### **Coordinate Systems**

#### **Normal Form (NF) Coordinates**
Defined as $X = [q_1, q_2, q_3, p_1, p_2, p_3]$:

| Index | Variable | Behavior | Physical Role |
| :--- | :--- | :--- | :--- |
| **1** | $q_1$ | Oscillatory ($\cos$) | Planar Phase/Position |
| **2** | $q_2$ | Oscillatory ($\cos$) | Vertical Phase/Position |
| **3** | $q_3$ | Exponential Growth | Unstable Amplitude (Departure) |
| **4** | $p_1$ | Oscillatory ($\sin$) | Planar Momentum |
| **5** | $p_2$ | Oscillatory ($\sin$) | Vertical Momentum |
| **6** | $p_3$ | Exponential Decay | Stable Amplitude (Approach) |

#### **Action-Angle (AA) Coordinates**
Defined as $AA = [I_1, I_2, I_3, \phi_1, \phi_2, \phi_3]$:
* **Actions ($I_1, I_2$):** Represent the "size" or energy of the planar and vertical oscillations respectively. $I_k \approx (q_k^2 + p_k^2)/2$.
* **Action ($I_3$):** Represents the saddle energy $I_3 \approx |q_3 \cdot p_3|$.
* **Angles ($\phi_1, \phi_2$):** Represent the phase of the orbit (0 to $2\pi$).
* **Angle ($\phi_3$):** Represents the hyperbolic phase $\phi_3 \approx 0.5 \ln|q_3/p_3|$.

---

### **Public API Reference**

| Function | Arguments | Description |
| :--- | :--- | :--- |
| `create_normal_form` | `mu=0.01215`, `L_idx=1`, `order=11`, `verbose=false` | Generates NF data and saves it to a `.jld2` file. This only needs to be done when you want a new point / $\mu$ value.|
| `init_NF` | `filename` | Loads a `.jld2` file and returns a `NormalFormAPI` object. |
| `AAtoNF` | `api`, `AA` | Transforms action-angle to normal form coordinates. |
| `NFtoAA` | `api`, `NF` | Transforms normal form to action-angle coordinates. |
| `NFtoRTB` | `api`, `NF` | Transforms normal form to physical coordinates. |
| `RTBtoNF` | `api`, `RTB` | Transforms physical to normal form coordinates. |
| `AAtoRTB` | `api`, `AA` | Transforms action-angle to physical coordinates. |
| `RTBtoAA` | `api`, `RTB` | Transforms physical to action-angle coordinates. |

---

### **Quick Start**

#### **1. Installation**
```julia
using Pkg
Pkg.develop(path="/path/to/CR3BPNormalForms")
```

#### **2. Generate and Load Data**
```julia
using CR3BPNormalForms

# Generate 11th order expansion for Earth-Moon L1
mu = 0.0121505856
create_normal_form(mu, 1, 11, true)

# Initialize API
api = init_NF("L1_mu0.0121505856_order11.jld2")
```

#### **3. Examples**
A detailed example can be found in the `examples/` directory.

---

### **References**
* **Jorba, À.** (1999). A Methodology for the Numerical Computation of Normal Forms, Centre Manifolds and First Integrals of Hamiltonian Systems. *Experimental Mathematics*, 8(2), 155-195. (Initial work establishing the methodology for numerical computation of normal forms).

* **Hunsberger, C.** NF_CR3BP_Python GitHub repository. (Primary reference for API design and coordinate transformation logic).

* **Peterson, L. T., and Scheeres, D. J.** (2023). Local Orbital Elements for the Circular Restricted Three-Body Problem. *Journal of Guidance, Control, and Dynamics*, 46(12), 2275-2289. doi:10.2514/1.G007435. (Action-Angle contributions)