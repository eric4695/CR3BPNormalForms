module CR3BPNormalForms

using LinearAlgebra
using ForwardDiff
using TaylorSeries
using JLD2
using DifferentialEquations

export create_normal_form, init_NF
export AAtoNF, NFtoAA
export AAtoRTB, RTBtoAA
export NFtoRTB, RTBtoNF

# Define cr3bp ODE
function cr3bp(du, u, μ, t)
    x, y, z, vx, vy, vz = u

    r13 = ((x + μ)^2 + y^2 + z^2)^1.5
    r23 = ((x - 1 + μ)^2 + y^2 + z^2)^1.5

    # Velocities
    du[1] = vx
    du[2] = vy
    du[3] = vz

    # Accelerations
    du[4] =  2 * vy + x - (1 - μ) * (x + μ) / r13 - μ * (x - 1 + μ) / r23
    du[5] = -2 * vx + y - (1 - μ) * y / r13 - μ * y / r23
    du[6] = -(1 - μ) * z / r13 - μ * z / r23
end
"""
    NormalFormAPI

A struct to hold all necessary data and parameters for normal form computations.

# Fields
- `data::Dict{String, Any}`: Stores the pre-computed system data for a single Lagrange point.
- `mu::Float64`: The mass parameter of the CR3BP system.
- `opts::NamedTuple`: Default options for the numerical solvers (e.g., `reltol`, `abstol`).
"""
struct NormalFormAPI
    data::Dict{String, Any}
    mu::Float64
    opts::NamedTuple
end

"""
    get_canonical_jacobian(u, mu)

Computes the Jacobian of the canonical Hamiltonian system for the CR3BP at state `u`.

For a Hamiltonian system with state `X = [q, p]` and equations of motion `Ẋ = J * ∇H`, 
the Jacobian is `A = J * ∇²H`, where `∇²H` is the Hessian of the Hamiltonian. 
This function computes `A` at the state `u` using automatic differentiation on the vector 
field `f(u) = Ẋ`.
The canonical state is `[x, y, z, p_x, p_y, p_z]`.
"""
function get_canonical_jacobian(u, mu)
    f(state) = begin
        x, y, z, px, py, pz = state
        r1 = sqrt((x+mu)^2 + y^2 + z^2); r2 = sqrt((x-1+mu)^2 + y^2 + z^2)
        dVdx = (1-mu)*(x+mu)/r1^3 + mu*(x-1+mu)/r2^3
        dVdy = (1-mu)*y/r1^3 + mu*y/r2^3
        dVdz = (1-mu)*z/r1^3 + mu*z/r2^3
        [px+y, py-x, pz, py-dVdx, -px-dVdy, -dVdz]
    end
    return ForwardDiff.jacobian(f, u)
end

"""
    compute_symplectic_basis(mu, L_pos)

Computes the complex eigenvectors and eigenvalues of the linearized CR3BP dynamics at a Lagrange point. The resulting basis of eigenvectors is symplectically normalized.

1.  Linearization: The dynamics are linearized around the equilibrium point `L_pos`, yielding the matrix `A`.
2.  Eigendecomposition: The eigenvalues `λ` and eigenvectors `v` of `A` are found. 
    Due to the Hamiltonian structure, they appear in pairs `(λ, -λ)` or `(λ, λ*)`. For 
    L1/2/3, we expect one real pair `(±λ_s)`, one pure imaginary pair `(±iω_p)`, and 
    one pure imaginary pair `(±iω_v)`.
3.  Symplectic Normalization: The eigenvectors `v_k` are scaled such that they form a symplectic basis. The standard normalization condition is:
    `v_k^T * J * v_{k+3} = 1` for `k=1,2,3`.
    This function uses a variant that results in `v_k^T * J * v_{k+3} = 1`, which 
    simplifies some subsequent transformations.
"""
function compute_symplectic_basis(mu, L_pos)
    u0 = [L_pos[1], 0.0, 0.0, 0.0, L_pos[1], 0.0]
    A = get_canonical_jacobian(u0, mu)
    
    idx_p = [1, 2, 4, 5]; idx_v = [3, 6]
    vals_p, vecs_p = eigen(A[idx_p, idx_p])
    vals_v, vecs_v = eigen(A[idx_v, idx_v])
    
    i_c = findfirst(x -> imag(x) > 0.1, vals_p); i_c_neg = findfirst(x -> imag(x) < -0.1, vals_p)
    i_s = findfirst(x -> real(x) > 0.1, vals_p); i_s_neg = findfirst(x -> real(x) < -0.1, vals_p)
    i_v = findfirst(x -> imag(x) > 0.1, vals_v); i_v_neg = findfirst(x -> imag(x) < -0.1, vals_v)
    
    eigvecs = zeros(ComplexF64, 6, 6)
    eigvals = zeros(ComplexF64, 6)
    map_p = [1,2,4,5]; map_v = [3,6]
    
    eigvals[1]=vals_p[i_c];     eigvecs[map_p,1]=vecs_p[:,i_c];
    eigvals[2]=vals_v[i_v];     eigvecs[map_v,2]=vecs_v[:,i_v];
    eigvals[3]=vals_p[i_s];     eigvecs[map_p,3]=vecs_p[:,i_s];
    eigvals[4]=vals_p[i_c_neg]; eigvecs[map_p,4]=vecs_p[:,i_c_neg]
    eigvals[5]=vals_v[i_v_neg]; eigvecs[map_v,5]=vecs_v[:,i_v_neg]
    eigvals[6]=vals_p[i_s_neg]; eigvecs[map_p,6]=vecs_p[:,i_s_neg]
    return eigvecs, eigvals
end

"""
    compute_poisson_bracket(f, g)

Computes the Poisson bracket of two functions `f` and `g` of the canonical phase space variables `(q, p)`.

The Poisson bracket is defined as:
`{f, g} = Σᵢ (∂f/∂qᵢ * ∂g/∂pᵢ - ∂f/∂pᵢ * ∂g/∂qᵢ)`
For a `d`-dimensional system, the sum is from `i=1` to `d`. Here `d=3`. It's computed 
using symbolic differentiation from `TaylorSeries.jl`.
"""
function compute_poisson_bracket(f, g)
    res = zero(f)
    for i in 1:3; res += derivative(f,i)*derivative(g,i+3)-derivative(f,i+3)*derivative(g,i); end
    return res
end

"""
    birkhoff_normal_form(H_input, lambda_vec, order, verbose)

Performs the Birkhoff normalization of the Hamiltonian `H_input` up to the specified `order` using a Lie series method.

The goal is to find a canonical transformation `(q,p) -> (Q,P)` that simplifies the 
Hamiltonian. This is done via a generating function `G`. The new Hamiltonian `K` is 
related to the old `H` by the Lie series: `K = exp(L_G) H = H + {H,G} + 
1/2!{{H,G},G} + ...`, where `L_G f = {f,G}`.

This function solves for `G` order-by-order by solving the homological equation:
`{H₂, Gₖ} + Hₖ = Kₖ`
where `H₂` is the quadratic part of the Hamiltonian, and `Hₖ` and `Kₖ` are the 
order-`k` parts of `H` and `K`. In the eigenvector basis, this becomes an 
algebraic equation for the coefficients of `Gₖ`:
`g_j = -h_j / div_j`
The divisor `div_j = iω ⋅ (m - n)` is a linear combination of the system's fundamental frequencies `ω`.

# References
- Jorba, À. (1999). A Methodology for the Numerical Computation of Normal Forms, Centre Manifolds and First Integrals of Hamiltonian Systems. Experimental Mathematics, 8(2), 155-195.
"""
function birkhoff_normal_form(H_input, lambda_vec, order, verbose)
    H_current = H_input; G_list = []
    
    q_vars = get_variables()
    
    for k in 3:order
        if verbose
            println("Computing Order $k...")
        end
        if k > get_order(H_current)
            dummy_taylor = 0.0 * q_vars[1]^k
            push!(G_list, dummy_taylor[k]) 
            continue
        end
        
        H_k = H_current[k]; G_k = zero(H_k); lookup = TaylorSeries.coeff_table[k+1]
        for (i, c) in enumerate(H_k.coeffs)
            if abs(c)<1e-14; continue; end
            exps = lookup[i]
            div = sum((exps[j]-exps[j+3])*lambda_vec[j] for j in 1:3)
            G_k[i] = -c/div
        end
        push!(G_list, G_k)
        
        term = H_current; max_iters = floor(Int, order/(k-2))+2
        for j in 1:max_iters
            term = compute_poisson_bracket(term, G_k)*(1.0/j)
            if all(norm.(term.coeffs).<1e-14); break; end
            H_current += term
        end
    end
    return H_current, G_list
end

# Evaluate x-acceleration (du[4]) at (x,0,0,0,0,0)
function fx(x, μ)
    u = (x, 0.0, 0.0, 0.0, 0.0, 0.0)
    du = zeros(6)
    cr3bp(du, u, μ, 0.0)
    return du[4]
end

# Finite difference derivative of fx
function dfx(x, μ; h=1e-8)
    return (fx(x+h, μ) - fx(x-h, μ)) / (2h)
end

# Newton iteration for a root
function newton_root(f, fp, μ, x0; tol=1e-14, maxiter=10000)
    x = x0
    for _ in 1:maxiter
        dx = f(x, μ) / fp(x, μ)
        x -= dx
        if abs(dx) < tol
            return x
        end
    end
    return x
end

function all_lagrange_points(μ; tol=1e-12, maxiter=100)
    # initial guesses
    r = (μ/3)^(1/3)
    x0_L1 = (1-μ) - r
    x0_L2 = (1-μ) + r
    x0_L3 = -(1-μ)
    L1 = (newton_root(fx, dfx, μ, x0_L1; tol=tol, maxiter=maxiter), 0)
    L2 = (newton_root(fx, dfx, μ, x0_L2; tol=tol, maxiter=maxiter), 0)
    L3 = (newton_root(fx, dfx, μ, x0_L3; tol=tol, maxiter=maxiter), 0)
    L4 = (1/2-μ, sqrt(3)/2)
    L5 = (1/2-μ, -sqrt(3)/2)
    return (L1, L2, L3, L4, L5)
end


"""
    create_normal_form(mu, L_idx, order=11, verbose=false)

Generates and saves all data required for the normal form analysis for a given `mu` and Lagrange point `L_idx`.

This function orchestrates the entire normal form generation process:
1.  Locates L-point: Calls `all_lagrange_points`.
2.  Finds Linear Basis: Calls `compute_symplectic_basis` to get the matrix `C` that diagonalizes the linear dynamics.
3.  Expands Hamiltonian: The full CR3BP Hamiltonian `H` is expanded as a Taylor 
    series in the linear modal coordinates `Q`, where the physical state `X` is given by 
    the linear transformation `X = C * Q`.
    `H(Q) = 0.5*(p'p) + y*px - x*py - (1-μ)/r₁ - μ/r₂`
4.  Normalizes: Calls `birkhoff_normal_form` on the Taylor-expanded `H(Q)` to 
    get the normalized Hamiltonian `H_nf` and the generating functions `G_list`.
5.  Saves Data: All relevant objects (`H_nf`, `G_list`, `C`, etc.) are saved to a JLD2 file for use by the API.
"""
function create_normal_form(mu=0.012154, L_idx=1, order=11, verbose=false)
    if verbose
        println("=== Creating Normal Form Data ===")
    end
    if !isinteger(order) || order < 1
        error("Order must be a positive integer")
    end
    if !isa(mu, AbstractFloat) || mu <= 0
        error("mu must be a number greater than 0")
    end
    if !isinteger(L_idx) || L_idx < 1 || L_idx > 3
        error("L_idx must be an integer between 1 and 3")
    end

    all_L_points = all_lagrange_points(mu)
    L_pos_tuple = all_L_points[L_idx]
    L_pos = [L_pos_tuple[1], 0.0, 0.0]
    if verbose
        println("Eqillibrium point found at x=$(round(L_pos[1], digits=4))...")
    end
    
    C, vals = compute_symplectic_basis(mu, L_pos)
    if verbose
        println("Linearized about L$(L_idx)...")
        println("Prepaing for symbolic manipulation...")
    end
    TaylorSeries.set_variables("q1 q2 q3 p1 p2 p3", order=order)
    Q = get_variables()
    X_poly = ComplexF64.(C) * Q
    
    x = L_pos[1] + X_poly[1]; 
    y = X_poly[2]; 
    z = X_poly[3]
    px = X_poly[4]; 
    py = L_pos[1] + X_poly[5]; 
    pz = X_poly[6]
    r1 = sqrt((x + mu)^2 + y^2 + z^2);
    r2 = sqrt((x - 1 + mu)^2+ y^2 +z^2)

    # Compute symbolic hamiltonian
    H_poly = 0.5 * (px^2 + py^2 + pz^2) + y*px - x*py - (1 - mu) / r1 - mu / r2
    
    # Run adaptive normal form
    H_final, G_list = birkhoff_normal_form(H_poly, vals[1:3], order, verbose)
    
    T1 = Matrix{Float64}(I, 6, 6)
    
    fname = "L$(L_idx)_mu$(mu)_order$(order).jld2"
    jldsave(fname; mu=mu, L_point=L_pos, H_nf=H_final, G_generators=G_list, 
            C=C, eigenvalues=vals, T1=T1, order=order)
    if verbose
        println("Saved $fname")
    end
end

function check_dims(x)
    if ndims(x) == 1
        return true, 1, reshape(x, 6, 1)
    elseif ndims(x) == 2
        r, c = size(x)
        if r == 6
            return true, c, x
        elseif c == 6
            return false, r, x'
        else
            error("Input must be 6xN or Nx6")
        end
    else
        error("Input must be 1D or 2D array")
    end
end

function init_NF(filename::String)
    jld_data = load(filename)
    order = jld_data["order"]
    TaylorSeries.set_variables("q1 q2 q3 p1 p2 p3", order=order)
    
    H_nf = jld_data["H_nf"]

    sys_data = Dict{String, Any}(
        "H_nf" => H_nf, "G_list" => jld_data["G_generators"],
        "C" => jld_data["C"], "Cinv" => inv(jld_data["C"]),
        "T1" => jld_data["T1"], "T1inv" => inv(jld_data["T1"]),
        "eigenvalues" => jld_data["eigenvalues"], "L_point" => jld_data["L_point"]
    )
    
    return NormalFormAPI(sys_data, jld_data["mu"], (reltol=1e-12, abstol=1e-16))
end

function AAtoNF_math(x_in)
    # Input: AA = [I1, I2, I3, phi1, phi2, phi3]
    I = x_in[1:3]
    Phi = x_in[4:6]
    
    nf_coords = zeros(Float64, 6)

    # MODES 1 & 2: Planar and Vertical
    #  q = sqrt(2I)cos(phi), p = -sqrt(2I)sin(phi)
    for k in 1:2
        if I[k] < 0
            @warn "Negative Action detected in Mode $k: $(I[k]). Forcing absolute value."
        end
        amp = sqrt(2 * abs(I[k]))
        nf_coords[k]   =  amp * cos(real(Phi[k]))      # q
        nf_coords[k+3] = -amp * sin(real(Phi[k]))      # p
    end

    # MODE 3: Saddle
    # q = sqrt(I)e^phi, p = sqrt(I)e^-phi
    saddle_amp = sqrt(abs(I[3]))
    nf_coords[3] = saddle_amp * exp(real(Phi[3]))      # q_saddle
    nf_coords[6] = saddle_amp * exp(-real(Phi[3]))     # p_saddle

    return nf_coords
end

function NFtoAA_math(x)
    # Input: NF = [q1, q2, q3, p1, p2, p3]
    I = zeros(Float64, 3)
    Phi = zeros(Float64, 3)
    
    # MODES 1 & 2: Planar and Vertical
    for k in 1:2
        q = x[k]
        p = x[k+3]
        
        # I = (q^2 + p^2) / 2
        I[k] = 0.5 * (q^2 + p^2)
        
        # phi = -atan(p, q)  (matches q=cos, p=-sin)
        Phi[k] = -atan(p, q)
        
        # Normalize to [0, 2pi]
        if Phi[k] < 0; Phi[k] += 2pi; end
    end

    # MODE 3: Saddle
    q_s = x[3]
    p_s = x[6]
    
    # I = |q * p|
    I[3] = abs(q_s * p_s)
    
    # phi = 0.5 * ln|q/p|
    # Avoid div/0
    if abs(p_s) > 1e-16 && abs(q_s) > 1e-16
        Phi[3] = 0.5 * log(abs(q_s / p_s))
    else
        Phi[3] = 0.0
    end

    return [I; Phi]
end

# ==============================================================================
# MAIN API
# ==============================================================================

"""
    qp_to_z(qp)

Transforms real canonical coordinates `(q, p)` into complex modal coordinates `z`.

For a 2D harmonic oscillator with coordinates `(q, p)`, a complex variable `z` can be defined:
`z = (q + i*p) / sqrt(2)`
such that the quadratic Hamiltonian `H = ω(q² + p²)/2` becomes `H = ω*z*z̄`. The conjugate variable is `z̄ = (q - i*p) / sqrt(2)`.
This function maps a 6-element real vector `[q₁, q₂, q₃, p₁, p₂, p₃]` to a 6-element 
complex vector `[z₁, z₂, z₃, z̄₁, z̄₂, z̄₃]`, where the saddle modes (3,6) are 
treated as real.
"""
function qp_to_z(qp)
    # Maps Real (q,p) to Complex (z, z_bar)
    z = zeros(ComplexF64, 6)
    s2 = 1.0/sqrt(2)
    for i in 1:2
        z[i]   = s2 * (qp[i] + 1im*qp[i+3])
        z[i+3] = s2 * (qp[i] - 1im*qp[i+3])
    end
    # Saddle (3,6) - Keep Real
    z[3] = qp[3]
    z[6] = qp[6]
    return z
end

"""
    z_to_qp(z)

Transforms complex modal coordinates `z` back into real canonical coordinates `(q, p)`. This is the inverse of `qp_to_z`.

This inverts the transformation `z = (q + i*p) / sqrt(2)`.
`q = (z + z̄) / sqrt(2)`
`p = -i * (z - z̄) / sqrt(2)`
The function reconstructs the real 6-element vector `[q₁, q₂, q₃, p₁, p₂, p₃]` from 
the 6-element complex vector `[z₁, z₂, z₃, z̄₁, z̄₂, z̄₃]`.
"""
function z_to_qp(z)
    # Maps Complex (z, z_bar) to Real (q,p)
    qp = zeros(Float64, 6)
    s2 = 1.0/sqrt(2)
    for i in 1:2
        # q = (z + z̄) / sqrt(2)
        qp[i]   = s2 * real(z[i] + z[i+3])
        # p = -i * (z - z̄) / sqrt(2)
        qp[i+3] = s2 * real(-1im * (z[i] - z[i+3]))
    end
    
    # Saddle
    qp[3] = real(z[3])
    qp[6] = real(z[6])
    return qp
end

"""
    ztoRTB_math(z, T1, C, L_point)

Transforms the complex modal coordinates `z` to the physical rotating frame. This is a linear transformation.

This function reverses the initial linearization and diagonalization step:
1.  Modal to Deviation: `X_dev = C * z`, where `C` is the matrix of eigenvectors. 
    This transforms from the modal basis back to canonical CR3BP coordinates, as 
    deviations from the equilibrium point.
2.  Scaling and Shifting: The deviations are scaled (`T1`) and shifted by the 
    equilibrium point's state vector (`L_point`) to get the full canonical state: 
    `X_canon = T1 * X_dev + X_L`.
3.  Canonical to Kinematic: The `V` matrix converts from canonical momenta `(px, py)` 
    to kinematic velocities `(vx, vy)` using `vx = px + y` and `vy = py - x`.
"""
function ztoRTB_math(z, T1, C, L_point)
    # Input is now Complex z (Modal Amplitudes)
    # Apply C to get X (Complex)
    X_complex = C * z
    
    # X_complex should be [x, y, z, px, py, pz] with Im ≈ 0
    # Apply shift to get the full canonical state
    shift = [L_point[1], L_point[2], L_point[3], -L_point[2], L_point[1], 0.0]
    X_canon = T1 * X_complex + shift
    
    # Convert from canonical momenta to kinematic velocities
    x, y, z, px, py, pz = real(X_canon)
    vx = px + y
    vy = py - x
    vz = pz
    
    return [x, y, z, vx, vy, vz]
end

"""
    RTBtoz_math(X, Cinv, T1inv, Vinv, L_point)

Transforms a physical CR3BP state vector `X` into the complex modal coordinates `z`. This is the inverse of `ztoRTB_math`.

This function performs the linearization and diagonalization of the physical state:
1.  Kinematic to Canonical: The inverse `Vinv` converts velocities to canonical momenta.
2.  Shift and Scale: The state is shifted to be a deviation from the L-point and scaled: `X_dev = T1inv * (Vinv*X - X_L)`.
3.  Deviation to Modal: `z = Cinv * X_dev`, where `Cinv` is the inverse of the 
    eigenvector matrix. This projects the canonical deviation vector onto the modal basis.
"""
function RTBtoz_math(X, Cinv, T1inv, L_point)
    # Convert from kinematic velocities to canonical momenta
    x, y, z, vx, vy, vz = X
    px = vx - y
    py = vy + x
    pz = vz
    X_canon = [x, y, z, px, py, pz]

    shift = [L_point[1], L_point[2], L_point[3], -L_point[2], L_point[1], 0.0]
    X_dev = T1inv * (X_canon - shift)
    return Cinv * X_dev
end

function is_taylor_small(poly::TaylorN, tol=1e-15)
    for hom in poly.coeffs; if !is_taylor_small(hom, tol); return false; end; end
    return true
end
function is_taylor_small(poly::HomogeneousPolynomial, tol=1e-15)
    return isempty(poly.coeffs) || all(abs.(poly.coeffs) .< tol)
end

"""
    evaluate_gradient_poly(poly, point)

Evaluates the gradient of a homogeneous polynomial at a specific point.

Given a polynomial `P(x)` where `x` is a vector, the gradient `∇P` is a vector 
whose components are the partial derivatives `∂P/∂xᵢ`. This function computes the 
gradient by analytically differentiating the Taylor series representation of the 
polynomial and then evaluating it at the given `point`.
"""
function evaluate_gradient_poly(poly, point)
    grad = zeros(ComplexF64, 6)
    deg = poly.order; lookup = TaylorSeries.coeff_table[deg+1]
    for (i, c) in enumerate(poly.coeffs)
        if abs(c) < 1e-15; continue; end
        exps = lookup[i]
        for v in 1:6
            k = exps[v]
            if k > 0
                term = c * k
                for j in 1:6
                    p = exps[j]; if j == v; p -= 1; end
                    if p > 0; term *= point[j]^p; end
                end
                grad[v] += term
            end
        end
    end
    return grad
end

"""
    numNFtransform_math(x, G_list, direction)

Applies the canonical transformation defined by the generating functions in `G_list` to a state vector `x`.

A canonical transformation generated by `G` maps a state `X` to `X_new` via the Lie series `X_new = exp(L_G) X`. The time-1 flow of the Hamiltonian vector field of `G` gives this transformation. The EOM for this flow is:
`dX/dτ = {X, G} = J * ∇G(X)`
This function numerically integrates this differential equation from `τ=0` to `τ=1` 
(or `-1` for the inverse) for each generator `G` in `G_list` to compute the full 
transformation.

# References
- Jorba, À. (1999). A Methodology for the Numerical Computation of Normal Forms, Centre Manifolds and First Integrals of Hamiltonian Systems. *Experimental Mathematics*, 8(2), 155-195.
"""
function numNFtransform_math(x, G_list, direction, opts)
    state = complex.(x)
    t_end = Float64(direction)
    for G in G_list
        if is_taylor_small(G); continue; end
        function dyn(u, p, t)
            g_vec = evaluate_gradient_poly(G, u)
            return [g_vec[4:6]; -g_vec[1:3]]
        end
        prob = ODEProblem(dyn, state, (0.0, t_end))
        sol = solve(prob, Vern9(); reltol=opts.reltol, abstol=opts.abstol)
        state = sol[end]
    end
    return state
end

# ==============================================================================
# MAIN API
# ==============================================================================

"""
    AAtoNF(api::NormalFormAPI, AA)

Transforms a state from Action-Angle (AA) coordinates to Normal Form (NF) coordinates.
"""
function AAtoNF(api::NormalFormAPI, AA)
    tr, num, clean_AA = check_dims(AA)
    res = zeros(6, num)
    for i in 1:num; res[:, i] = AAtoNF_math(clean_AA[:, i]); end
    return tr ? res : res'
end

"""
    NFtoAA(api::NormalFormAPI, NF)

Transforms a state from Normal Form (NF) coordinates to Action-Angle (AA) coordinates.
"""
function NFtoAA(api::NormalFormAPI, NF)
    tr, num, clean_NF = check_dims(NF)
    res = zeros(ComplexF64, 6, num)
    for i in 1:num; res[:, i] = NFtoAA_math(clean_NF[:, i]); end
    return tr ? res : res'
end

"""
    NFtoRTB(api::NormalFormAPI, NF)

Transforms a state vector from the normalized frame (NF) to the rotating barycentric (RTB) frame.

This is the full forward transformation, composing several steps:
1.  Nonlinear Transformation (`numNFtransform_math`): The sequence of Lie series 
    transformations is applied to the NF state to map it to the linearized canonical 
    coordinates `(q,p)`.
2.  Real to Complex (`qp_to_z`): The real `(q,p)` coordinates are converted to complex modal coordinates `z`.
3.  Linear Transformation (`ztoRTB_math`): The modal coordinates `z` are transformed back into the physical RTB state.
"""
function NFtoRTB(api::NormalFormAPI, NF)
    tr, num, clean_NF = check_dims(NF)
    data = api.data
    res = zeros(ComplexF64, 6, num)
    for i in 1:num
        nf_vec = clean_NF[:, i]
        # 1. Lie Transform (Forward)
        qp = numNFtransform_math(nf_vec, reverse(data["G_list"]), 1.0, api.opts)
        # 2. Real (qp) -> Complex (z)
        z = qp_to_z(qp)
        # 3. Linear Map
        res[:, i] = ztoRTB_math(z, data["T1"], data["C"], data["L_point"])
    end
    return tr ? real.(res) : real.(res')
end

"""
    RTBtoNF(api::NormalFormAPI, RTB)

Transforms a state vector from the rotating barycentric (CR3BP) frame to the normalized frame (NF).

This is the full inverse transformation, composing several steps:
1.  Linearization (`RTBtoz_math`): The RTB state is converted to complex modal coordinates `z`.
2.  Complex to Real (`z_to_qp`): The modal coordinates `z` are converted to real canonical coordinates `(q,p)`.
3.  Nonlinear Transformation (`numNFtransform_math`): The sequence of Lie series 
    transformations is applied (in reverse) to map from the linearized coordinates 
    `(q,p)` to the final normal form coordinates.
"""
function RTBtoNF(api::NormalFormAPI, RTB)
    tr, num, clean_RTB = check_dims(RTB)
    data = api.data
    res = zeros(ComplexF64, 6, num)
    for i in 1:num
        rtb_vec = clean_RTB[:, i]
        # 1. Linear Map (Inverse)
        z = RTBtoz_math(rtb_vec, data["Cinv"], data["T1inv"], data["L_point"])
        # 2. Complex (z) -> Real (qp)
        qp = z_to_qp(z)
        # 3. Lie Transform (Inverse)
        res[:, i] = numNFtransform_math(qp, data["G_list"], -1.0, api.opts)
    end
    return tr ? real.(res) : real.(res')
end

"""
    AAtoRTB(api::NormalFormAPI, AA)

Composes `AAtoNF` and `NFtoRTB` to transform from Action-Angle to the RTB frame.
"""
function AAtoRTB(api::NormalFormAPI, AA)
    nf = AAtoNF(api, AA)
    return NFtoRTB(api, nf)
end

"""
    RTBtoAA(api::NormalFormAPI, RTB)

Composes `RTBtoNF` and `NFtoAA` to transform from the RTB frame to Action-Angle.
"""
function RTBtoAA(api::NormalFormAPI, RTB)
    nf = RTBtoNF(api, RTB)
    return NFtoAA(api, nf)
end

end