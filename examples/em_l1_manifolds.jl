using Plots # For plotting manifolds
using OrdinaryDiffEq # For propogating trajectories 
using LaTeXStrings # For nice labels 
using CR3BPNormalForms

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

# Parameters
μ = 0.01215058
L_idx = 1

create_normal_form(μ, L_idx, 15, true) # Verbose flag prints info on what the NF code is doing
 
fname = "L1_mu0.01215058_order15.jld2"
api = init_NF(fname)

eps = 1e-3  # Perturbation size
x_L1 = api.data["L_point"][1]

# Perturbations
nf_u_plus  = [0.0, 0.0,  eps, 0.0, 0.0, 0.0] 
nf_u_minus = [0.0, 0.0, -eps, 0.0, 0.0, 0.0]
nf_s_plus  = [0.0, 0.0, 0.0, 0.0, 0.0,  eps]
nf_s_minus = [0.0, 0.0, 0.0, 0.0, 0.0, -eps]

# Convert to Cartesian
X_u_p = NFtoRTB(api, nf_u_plus)
X_u_m = NFtoRTB(api, nf_u_minus)
X_s_p = NFtoRTB(api, nf_s_plus)
X_s_m = NFtoRTB(api, nf_s_minus)

# Propogate
time = 4

sol_u_p = solve(ODEProblem(cr3bp, X_u_p, (0.0,  time), μ), VCABM(), reltol=1e-12)
sol_u_m = solve(ODEProblem(cr3bp, X_u_m, (0.0,  time), μ), VCABM(), reltol=1e-12)
sol_s_p = solve(ODEProblem(cr3bp, X_s_p, (0.0, -time), μ), VCABM(), reltol=1e-12)
sol_s_m = solve(ODEProblem(cr3bp, X_s_m, (0.0, -time), μ), VCABM(), reltol=1e-12)

# Create Plot
plt = plot(
    xlabel = L"x\ [non-dim]",
    ylabel = L"y\ [non-dim]",
    legend = :bottomleft,
    tickfontfamily = "Computer Modern"
)

plot!(plt, sol_u_p, idxs=(1, 2), color=:red, linewidth=1.5, label=L"W^U")
plot!(plt, sol_u_m, idxs=(1, 2), color=:red, linewidth=1.5, label=false)
plot!(plt, sol_s_p, idxs=(1, 2), color=:blue, linewidth=1.5, label=L"W^S")
plot!(plt, sol_s_m, idxs=(1, 2), color=:blue, linewidth=1.5, label=false)

# Plot Bodies and L1
scatter!(plt, [-μ], [0], label=L"Earth", color=:green, markersize=8)
scatter!(plt, [1 - μ],  [0], label=L"Moon",  color=:gray,  markersize=6)

# L1 as a Cross
scatter!(plt, [x_L1], [0], label=L"L_1", shape=:cross, color=:black, markersize=8, stroke_width=2)

display(plt)
savefig("NFmanifolds.png")