# Pipe friction calculation

First, we will import the necessary packages to solve the problems.



```
from fluids import friction, core, fittings
from fluids.piping import nearest_pipe
from pyfluids import Fluid, FluidsList, Input
import numpy as np
from scipy.optimize import root_scalar
```

## 1. Water flow
Water at 30<sup>o</sup>C flows through a 20m length of 50mm steel pipe (smooth wall) of Sch. 40 at a flow rate of 200 liter per min. Calculate the Reynolds number and friction factor. What is the pressure drop and head loss in the pipe?

============== Answer ==============

We will use the following equations to get the answers.

$$
\text{Re} _d = \frac{\rho V D}{\mu}
$$

$$
h_f = \frac{1}{2g} \frac{V^2}{D}
$$

The density (\\(\rho\\)) and dynamic viscosity (\\(\mu\\)) of water can be obtained using `pyfluids` package which is a wrapper for `CoolProp`.


```
# Following is given from the problem:
temp_water_deg_C = 30
length_pipe_m = 20
nominal_diamter_m = 0.05
pipe_schedule = "40"
flow_l_per_min = 200
roughness = friction.material_roughness("clean steel pipe")
grav_acc_m_per_s2 = 9.81
print(f"Roughness of the pipe is {roughness:.6f}")

# First, let's define our water
water = Fluid(FluidsList.Water).with_state(
    Input.pressure(1e5), Input.temperature(temp_water_deg_C)
)

# Calculate the inner diameter of the pipe
nps, d_inner_m, d_outer_m, thickness_m = nearest_pipe(Di=0.05, schedule="40")

print(f"The inner diameter of the pipe is: {d_inner_m:.4f}m")

# Calculate the velocity from the flow and the diameter
flow_m3_per_s = flow_l_per_min / 60 / 1e3
area_pipe_m2 = d_inner_m ** 2 * np.pi / 4
velocity_m_per_s = flow_m3_per_s / area_pipe_m2

# Calculate Reynolds number
re = water.density * velocity_m_per_s * d_inner_m / water.dynamic_viscosity
print(f"Re = {re:.1f}")

# Let's try the API
re_api = core.Reynolds(
    V=velocity_m_per_s, D=d_inner_m, rho=water.density, mu=water.dynamic_viscosity
)
print(f"Re = {re_api:.1f} from fluids API")

# Calulcate friction factor
fric_factor = friction.friction_factor(Re=re, eD=roughness/d_inner_m)
print(f"Friction factor is {fric_factor:.4f}")

# Calculate pressure drop and head loss with Darcy-Weibach equation
head_loss = fric_factor \
    * 1 / (2 * grav_acc_m_per_s2) \
    * velocity_m_per_s**2 / d_inner_m \
    * length_pipe_m
p_drop = water.density * grav_acc_m_per_s2 * head_loss
print(f"Head loss is {head_loss:.2f} m")
print(f"pressure drop is {p_drop:.0f} Pa")
```

    Roughness of the pipe is 0.000150
    The inner diameter of the pipe is: 0.0525m
    Re = 101000.2
    Re = 101000.2 from fluids API
    Friction factor is 0.0271
    Head loss is 1.25 m
    pressure drop is 12230 Pa
    

We can try an API from `fluids` to get the same answer .


```
mass_flow_kg_per_s = flow_m3_per_s * water.density
p_drop = friction.one_phase_dP(
    m=mass_flow_kg_per_s,
    rho=water.density,
    mu=water.dynamic_viscosity,
    D=d_inner_m,
    roughness=roughness,
    L=length_pipe_m
)
head_loss = p_drop / (water.density * grav_acc_m_per_s2)
print(f"Pressure drop from API is: {p_drop:.0f} Pa")
print(f"Head loss from API is: {head_loss:.3f} m")
```

    Pressure drop from API is: 12230 Pa
    Head loss from API is: 1.252 m
    

## 2. Find flow from the pressure drop
We measured the pressure drop of the pipe and we found it to be 25000 Pa. What is the flow in the pipe? 

============== Answer ==============

Finding flow from the pressure drop is not a trivial problem because the function to solve is non-linear implicit equation.

$$
\Delta p = f_d\left(\mathrm{Re}_D\right) \frac{\rho}{2} \frac{V^2}{D_i} \cdot L
$$

$$
\text{Re} _d = \frac{\rho V D}{\mu}
$$

$$
Q = V \cdot A = \frac{\pi D^2}{4} V
$$

First, we will try with an assumption that the friction factor is constant as the Re is quite high.

$$
V = \sqrt{\Delta p \cdot \frac{2D_i}{f_d \rho L} }
$$


```
p_drop_pa = 25000
velocity_m_per_s = np.sqrt(
    p_drop_pa * 2 * d_inner_m \
    / (fric_factor * water.density * length_pipe_m)
)
flow_m3_per_s = velocity_m_per_s * area_pipe_m2
flow_l_per_min = flow_m3_per_s * 60 * 1000
flow_kg_per_s = flow_m3_per_s * water.density
print(f"Water flow is {flow_l_per_min:.1f} l/min or {flow_kg_per_s:.2f} kg/s")
```

    Water flow is 285.9 l/min or 4.74 kg/s
    

We can try to verify it by calculating the pressure drop again.


```
p_drop_estimated = friction.one_phase_dP(
    m=flow_kg_per_s,
    rho=water.density,
    mu=water.dynamic_viscosity,
    D=d_inner_m,
    roughness=roughness,
    L=length_pipe_m
)
print(f"Presure drop calculated back is {p_drop_estimated:.1f} Pa")
```

    Presure drop calculated back is 24649.6 Pa
    

This may be close enough. However, we can get more accurate solution by solving the equation numerically. First, we define the function that is the equation to solve. 

$$
f(Q) = \Delta P - \Delta P(Q) = 0
$$


```
def function_to_solve(x_as_flow_l_per_min):
    flow_kg_per_s = x_as_flow_l_per_min / 60 / 1000 * water.density
    return p_drop_pa - friction.one_phase_dP(
        m=flow_kg_per_s,
        rho=water.density,
        mu=water.dynamic_viscosity,
        D=d_inner_m,
        roughness=roughness,
        L=length_pipe_m
    )
```

Then, we can use the solver in `SciPy` package to solve the eqation.


```
solution = root_scalar(f=function_to_solve, x0=flow_l_per_min, x1=flow_l_per_min * 0.99)
print(f"The accurate answer is {solution.root:.2f} l/min")
```

    The accurate answer is 288.00 l/min
    

## 3. Friction from pipe fittings and valves

Given the following pipe fittings and pipe, calculate the pressure drop of the pipe section.

- fluid: methane at -163 <sup>o</sup>C and 1 bar
- flow: 9000 m<sup>3</sup>/h
- nominal pipe diameter: 450mm, 
- Clean stainless steel pipe
- pipe schedule: 80s
- pipe length: 24m
- 90<sup>o</sup> bend: bend diameter 457mm
- Valve: Gate valve with \\(Cv=24000\\)

=============== Answer ===============

We will calculate the pressure drop for the pipe and calculate K value for each fitting or
valve to calculate the pressure drop with the following formula.

$$
K=\frac{\Delta p}{0.5 \rho V^2}
$$

$$
\sum \Delta p = \Delta p_{\text{pipe}} + {0.5 \rho V^2} \sum K
$$

$$
K=1.6 \times 10^9 \frac{D^4}{\left(K_v / 1.56\right)^2}
$$



```
liquid_methane = Fluid(FluidsList.Methane).with_state(
    Input.pressure(1e5), Input.temperature(-163)
)
print(f"Liquid methane: \n\t"
      f"density: {liquid_methane.density} kg/m3\n\t"
      f"dynamic viscosity: {liquid_methane.dynamic_viscosity:.3e} Pa s")

flow_m3_per_h = 9000
roughness = friction.material_roughness("clean stainless steel pipe")
print(f"Roughness of the pipe is {roughness:.3e}")
nps, d_inner_m, d_outer_m, thickness_m = nearest_pipe(Di=0.45, schedule="80")
length_pipe_m = 24
bend_radius_m = 0.457
c_v_valvue = 24000

# We need Reynods number to calculate K for the bend
area_pipe_section = np.pi * d_inner_m**2 / 4
velocity_m_per_s = flow_m3_per_h / 3600 / area_pipe_section
re = liquid_methane.density * velocity_m_per_s * d_inner_m \
    / liquid_methane.dynamic_viscosity

# get the K value for the bend
k_bend = fittings.bend_rounded(
    Di=d_inner_m,
    angle=90,
    rc=bend_radius_m,
    Re=re,
    roughness=roughness
)
print(f"K for the bend is {k_bend:.3f}")

# Convert Cv to K for the valvue
k_valve = 1.6e9 * d_inner_m**4 / (c_v_valvue / 1.1561)**2
print(f"K for the valve is {k_valve:.3f}")
# We can also use the prebuilt function
k_valve_api = fittings.Cv_to_K(Cv=c_v_valvue, D=d_inner_m)
assert np.isclose(k_valve, k_valve_api), f"The values are different {k_valve} vs {k_valve_api}"

# Calculate the pressure drop for each
flow_kg_per_s = flow_m3_per_h / 3600 * liquid_methane.density
p_drop_pipe = friction.one_phase_dP(
    m=flow_kg_per_s,
    rho=liquid_methane.density,
    mu=liquid_methane.dynamic_viscosity,
    D=d_inner_m,
    roughness=roughness,
    L=length_pipe_m
)
p_drop_bend = k_bend * 0.5 * liquid_methane.density * velocity_m_per_s**2
p_drop_valve = k_valve * 0.5 * liquid_methane.density * velocity_m_per_s**2
p_drop_total = p_drop_pipe + p_drop_bend + p_drop_valve
print("Pressure drop [Pa]:\n\t"
      f"pipe: {p_drop_pipe:.0f}\n\t"
      f"bend: {p_drop_bend:.0f}\n\t"
      f"valve: {p_drop_valve:.0f}\n\t"
      f"total: {p_drop_total:.0f}\n\t")
```

    Liquid methane: 
    	density: 424.5701316256901 kg/m3
    	dynamic viscosity: 1.210e-04 Pa s
    Roughness of the pipe is 1.500e-04
    K for the bend is 0.276
    K for the valve is 0.160
    Pressure drop [Pa]:
    	pipe: 40138
    	bend: 13761
    	valve: 7986
    	total: 61885
    	
    
