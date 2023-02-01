# Displacement Pump Performance

First, we will import the necessary packages to solve the problems.



```
import numpy as np
```

## 1. Volumetric Efficiency
A pump having a volumetric efficiency of 96% delivers 29 liter per minute of oil at 1000 RPM. What is the volumetric displacement of the pump?

============== Answer ==============

We will use the following equations to get the answers.

$$
\eta_{v} = \frac{Q_A}{Q_T}
$$

$$
Q_T = V_D \cdot N
$$

$$
V_D = \frac{Q_A}{\eta_v N}
$$


```python
# Following is given from the problem:
vol_flow_liter_per_min = 29
efficiency_vol = 0.96
speed_rpm = 1000
vol_disp_liter = vol_flow_liter_per_min / efficiency_vol / speed_rpm

print(f"Volumetric displacement: {vol_disp_liter:.4f}L")
```

    Volumetric displacement: 0.0302L


## Displacement Pump Performance
A pump has a displacement volume of 100 cm<sup>3</sup>. It delivers 0.0152 m<sup>3</sup>/s of oil at 1000 RPM and 70 bar. If the prime mover input torque is 124.3Nm. What is the overall efficiency of the pump? What is the theoretical torque required to operate the pump?


```python
# Solution
vol_disp_m3 = 100e-6
vol_flow_m3_per_s = 0.00152
speed_rpm = 1000
pressure_pump_pa = 70e5
torque_shaft_nm = 124.3

# Overall efficiency
speed_rad_per_s = speed_rpm / 60 * 2 * np.pi
efficiency_overall = pressure_pump_pa * vol_flow_m3_per_s / (torque_shaft_nm * speed_rad_per_s)
print(f"The overall efficiency of the pump is {efficiency_overall:.3f}")

# Volumetric / mechanical efficiency
efficiency_vol = vol_flow_m3_per_s / (vol_disp_m3 * speed_rpm / 60)
print(f"The volume efficiency of the pump is {efficiency_vol:.3f}")
efficiency_mech = efficiency_overall / efficiency_vol
print(f"The mechanical efficiency of the pump is {efficiency_mech:.3f}")
torque_ideal_nm = torque_shaft_nm / efficiency_mech
print(f"The ideal torque is {torque_ideal_nm:.1f}Nm")

```

    The overall efficiency of the pump is 0.817
    The volume efficiency of the pump is 0.912
    The mechanical efficiency of the pump is 0.896
    The ideal torque is 138.7Nm

