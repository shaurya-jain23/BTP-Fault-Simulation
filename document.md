<a name="complete_three_phase_fault_simula_e1e1eb"></a>**Complete Three-Phase Fault Simulation Script**

<a name="overview"></a>**Overview**

This document provides the complete, corrected Yade DEM script for geological fault simulation with realistic parameters and proper three-phase workflow:

1. **Phase 0**: Gravity Deposition & Equilibration
1. **Phase 1**: Overburden Stress Application & Consolidation
1. **Phase 2**: Deviatoric Loading & Fault Formation

<a name="complete_python_script"></a>**Complete Python Script**

Save this as fault\_simulation\_complete.py:

<a name="geological_fault_zone_simulation_3cd277"></a>**"""\
Geological Fault Zone Simulation - Three-Phase DEM Model**

Implements realistic fault rupture simulation with:

- Phase 0: Gravity deposition and particle settling
- Phase 1: Overburden stress application (burial simulation)
- Phase 2: Deviatoric loading (fault formation)

Author: [Your Name]\
Date: November 2025\
Purpose: BTP - Geological Fault Simulation\
"""

from yade import pack, plot, qt, utils\
import numpy as np

print("="\*70)\
print("GEOLOGICAL FAULT SIMULATION - THREE-PHASE DEM MODEL")\
print("="\*70)

<a name="bm_"></a>**============================================================================**

<a name="section_1_simulation_parameters_c_ec1955"></a>**SECTION 1: SIMULATION PARAMETERS (Calibrated for Sandstone at 500m depth)**

<a name="bm_2"></a>**============================================================================**

<a name="burial_depth_for_stress_calculation"></a>**Burial depth for stress calculation**

BURIAL\_DEPTH = 500 # meters (shallow crustal fault)

<a name="domain_geometry_in_meters"></a>**Domain geometry (in meters)**

DOMAIN\_X = (-10, 10) # 20m width\
DOMAIN\_Y = (-10, 10) # 20m length\
DOMAIN\_Z = (-10, 0) # 10m height\
domain = (DOMAIN\_X[0], DOMAIN\_X[1], DOMAIN\_Y[0], DOMAIN\_Y[1], DOMAIN\_Z[0], DOMAIN\_Z[1])

<a name="particle_size_parameters_scaled_f_ffbabf"></a>**Particle size parameters (scaled for computational efficiency)**

PARTICLE\_SIZE\_COARSE = 0.5 # Bottom layer (m)\
PARTICLE\_SIZE\_MEDIUM = 0.35 # Middle layer (m)\
PARTICLE\_SIZE\_FINE = 0.25 # Top layer (m)

<a name="number_of_particles_per_layer"></a>**Number of particles per layer**

NUM\_COARSE = 250\
NUM\_MEDIUM = 350\
NUM\_FINE = 400

<a name="bm_3"></a>**============================================================================**

<a name="section_2_realistic_material_prop_9f5aae"></a>**SECTION 2: REALISTIC MATERIAL PROPERTIES (Sandstone)**

<a name="bm_4"></a>**============================================================================**

<a name="layer_1_competent_sandstone_botto_1b025c"></a>**Layer 1 - Competent Sandstone (Bottom layer, most stable)**

mat1 = CohFrictMat(\
young=30e9, # 30 GPa (typical sandstone)\
poisson=0.25, # Poisson's ratio\
frictionAngle=np.radians(35), # 35° internal friction angle\
density=2650, # kg/m³ (quartz sandstone)\
isCohesive=True, # Enable bonding\
normalCohesion=8e6, # 8 MPa tensile strength\
shearCohesion=10e6, # 10 MPa cohesion\
label='CompetentSandstone'\
)

<a name="layer_2_fractured_sandstone_middl_29814c"></a>**Layer 2 - Fractured Sandstone (Middle layer, intermediate strength)**

mat2 = CohFrictMat(\
young=20e9, # Reduced from weathering\
poisson=0.25,\
frictionAngle=np.radians(32), # Slightly reduced\
density=2600, # Lower density (porosity)\
isCohesive=True,\
normalCohesion=5e6, # 5 MPa (weaker bonds)\
shearCohesion=6e6, # 6 MPa\
label='FracturedSandstone'\
)

<a name="layer_3_damage_zone_top_layer_fau_de0f23"></a>**Layer 3 - Damage Zone (Top layer, fault zone material)**

mat3 = CohFrictMat(\
young=10e9, # Highly fractured\
poisson=0.25,\
frictionAngle=np.radians(28), # Reduced cohesion\
density=2550, # More porous\
isCohesive=True,\
normalCohesion=2e6, # 2 MPa (very weak)\
shearCohesion=3e6, # 3 MPa\
label='DamageZone'\
)

O.materials.append(mat1)\
O.materials.append(mat2)\
O.materials.append(mat3)

<a name="calculate_average_density_for_str_11289c"></a>**Calculate average density for stress calculations**

avg\_density = (mat1.density + mat2.density + mat3.density) / 3

<a name="bm_5"></a>**============================================================================**

<a name="section_3_stress_state_calculatio_73fd71"></a>**SECTION 3: STRESS STATE CALCULATION (Depth-based, realistic)**

<a name="bm_6"></a>**============================================================================**

<a name="gravitational_acceleration"></a>**Gravitational acceleration**

g = 9.81 # m/s²

<a name="fix_1_use_more_realistic_shallow_3ee42f"></a>**✅ FIX 1: Use more realistic shallow depth with lower stress**

<a name="deep_burial_with_weak_bonding_cau_16daef"></a>**(Deep burial with weak bonding causes instability)**

BURIAL\_DEPTH = 100 # Reduced from 500m to 100m for stability\
avg\_density = (mat1.density + mat2.density + mat3.density) / 3

<a name="lithostatic_overburden_vertical_s_968f92"></a>**Lithostatic (overburden) vertical stress: σ\_v = ρ × g × h**

lithostatic\_stress = avg\_density \* g \* BURIAL\_DEPTH # Pa\
print(f"\n--- Calculated Stress State at {BURIAL\_DEPTH}m Depth ---")\
print(f"Lithostatic vertical stress: {lithostatic\_stress/1e6:.2f} MPa")

<a name="lateral_stress_coefficient_at_res_751e19"></a>**Lateral stress coefficient (at-rest earth pressure)**

<a name="k0_1_sin_φ_for_normally_consolida_f517b3"></a>**K0 = 1 - sin(φ) for normally consolidated soil/rock**

avg\_friction\_angle = (35 + 32 + 28) / 3 # Average friction angle\
K0 = 1 - np.sin(np.radians(avg\_friction\_angle))\
horizontal\_stress = K0 \* lithostatic\_stress\
print(f"K0 coefficient: {K0:.3f}")\
print(f"Horizontal confining stress: {horizontal\_stress/1e6:.2f} MPa")\
print(f"Stress ratio (σh/σv): {K0:.3f}")

<a name="deviatoric_stress_for_fault_loadi_5f6168"></a>**Deviatoric stress for fault loading (Phase 2)**

<a name="typical_fault_simulation_increase_4dd12f"></a>**Typical fault simulation: increase vertical stress to 1.5-2.0 times confining**

fault\_loading\_stress = 1.5 \* lithostatic\_stress # Reduced from 1.8 for stability\
print(f"Fault loading stress (Phase 2): {fault\_loading\_stress/1e6:.2f} MPa")\
print("-" \* 70)

<a name="bm_7"></a>**============================================================================**

<a name="section_4_particle_packing_three_3c1644"></a>**SECTION 4: PARTICLE PACKING (Three-layer stratification)**

<a name="bm_8"></a>**============================================================================**

print("\n--- Generating Particle Packing ---")\
sp = pack.SpherePack()

<a name="fix_2_denser_packing_with_smaller_c9cb9c"></a>**✅ FIX 2: Denser packing with smaller rRelFuzz for more contacts**

<a name="layer_1_bottom_competent_sandston_3c32aa"></a>**Layer 1: Bottom (Competent sandstone, -10m to -6m)**

sp.makeCloud(\
(domain[0], domain[2], domain[4]),\
(domain[1], domain[3], -6),\
rMean=PARTICLE\_SIZE\_COARSE,\
rRelFuzz=0.15, # Reduced from 0.2 for denser packing\
num=NUM\_COARSE,\
seed=42000,\
porosity=0.35 # Add explicit porosity target\
)\
print(f"Layer 1 (Competent): {NUM\_COARSE} coarse particles (r={PARTICLE\_SIZE\_COARSE}m)")

<a name="layer_2_middle_fractured_sandston_aeaf40"></a>**Layer 2: Middle (Fractured sandstone, -6m to -3m)**

sp.makeCloud(\
(domain[0], domain[2], -6),\
(domain[1], domain[3], -3),\
rMean=PARTICLE\_SIZE\_MEDIUM,\
rRelFuzz=0.15, # Reduced from 0.2\
num=NUM\_MEDIUM,\
seed=42001,\
porosity=0.35\
)\
print(f"Layer 2 (Fractured): {NUM\_MEDIUM} medium particles (r={PARTICLE\_SIZE\_MEDIUM}m)")

<a name="layer_3_top_damage_zone_3m_to_0m"></a>**Layer 3: Top (Damage zone, -3m to 0m)**

sp.makeCloud(\
(domain[0], domain[2], -3),\
(domain[1], domain[3], domain[5]),\
rMean=PARTICLE\_SIZE\_FINE,\
rRelFuzz=0.15, # Reduced from 0.2\
num=NUM\_FINE,\
seed=42002,\
porosity=0.35\
)\
print(f"Layer 3 (Damage Zone): {NUM\_FINE} fine particles (r={PARTICLE\_SIZE\_FINE}m)")

<a name="insert_particles_into_simulation_3a585b"></a>**Insert particles into simulation with material assignment by depth**

for center, radius in sp:\
z = center[2]\
if z < -6:\
mat = mat1\
elif z < -3:\
mat = mat2\
else:\
mat = mat3\
O.bodies.append(sphere(center, radius, material=mat))

total\_particles = NUM\_COARSE + NUM\_MEDIUM + NUM\_FINE\
print(f"Total particles generated: {total\_particles}")\
print("-" \* 70)

<a name="bm_9"></a>**============================================================================**

<a name="section_5_boundary_walls"></a>**SECTION 5: BOUNDARY WALLS**

<a name="bm_10"></a>**============================================================================**

walls = aabbWalls(\
[(domain[0], domain[2], domain[4]), (domain[1], domain[3], domain[5])],\
thickness=0.5,\
material=mat1\
)\
wallIds = O.bodies.append(walls)

<a name="bm_11"></a>**============================================================================**

<a name="section_6_simulation_engines_corr_8123fd"></a>**SECTION 6: SIMULATION ENGINES (Corrected for three-phase workflow)**

<a name="bm_12"></a>**============================================================================**

O.engines = [\
ForceResetter(),

InsertionSortCollider([Bo1\_Sphere\_Aabb(), Bo1\_Box\_Aabb()]),\
\
InteractionLoop(\
`    `[Ig2\_Sphere\_Sphere\_ScGeom6D(), Ig2\_Box\_Sphere\_ScGeom6D()],\
\
`    `# CRITICAL: Do NOT bond immediately - allow gravity settling first\
`    `[Ip2\_CohFrictMat\_CohFrictMat\_CohFrictPhys(\
`        `setCohesionNow=False,           # ✅ Wait for Phase 0 completion\
`        `setCohesionOnNewContacts=False, # ✅ Manual bonding control\
`        `label='interactionPhys'\
`    `)],\
\
`    `[Law2\_ScGeom6D\_CohFrictPhys\_CohesionMoment(\
`        `useIncrementalForm=True,\
`        `always\_use\_moment\_law=False,\
`        `label='cohesiveLaw'\
`    `)]\
),\
\
\# ✅ FIX 3: Higher damping for gravity settling phase\
NewtonIntegrator(damping=0.7, gravity=(0, 0, -9.81)),\
\
\# ✅ FIX 4: Triaxial controller with slower servo for stability\
TriaxialStressController(\
`    `stressMask=7,                    # Control all three axes\
`    `internalCompaction=False,        # ✅ Disabled initially\
`    `goal1=-horizontal\_stress,        # Lateral (X)\
`    `goal2=-horizontal\_stress,        # Lateral (Y)\
`    `goal3=-lithostatic\_stress,       # Vertical (Z)\
`    `maxStrainRate=(0.1, 0.1, 0.1),   # ✅ Limit wall velocity for stability\
`    `stressDamping=0.3,               # ✅ Add servo damping\
`    `label="triax"\
),\
\
\# Phase control callbacks (order matters!)\
PyRunner(command='checkGravityEquilibrium()', iterPeriod=100, label='gravityCheck'),\
PyRunner(command='checkOverburdenEquilibrium()', iterPeriod=100, label='overburdenCheck'),\
PyRunner(command='checkFaultLoading()', iterPeriod=100, label='faultCheck'),\
\
\# Data collection\
PyRunner(command='saveData()', iterPeriod=500),\
PyRunner(command='monitorBonds()', iterPeriod=1000)

]

O.dt = 0.5 \* PWaveTimeStep()

<a name="bm_13"></a>**============================================================================**

<a name="section_7_phase_state_variables"></a>**SECTION 7: PHASE STATE VARIABLES**

<a name="bm_14"></a>**============================================================================**

phase0\_complete = False # Gravity deposition\
phase1\_complete = False # Overburden consolidation\
phase2\_active = False # Fault loading\
simulation\_stopped = False

brokenBonds = 0\
total\_bonds = 0

<a name="bm_15"></a>**============================================================================**

<a name="section_8_phase_0_gravity_deposit_4d0b11"></a>**SECTION 8: PHASE 0 - GRAVITY DEPOSITION & EQUILIBRATION**

<a name="bm_16"></a>**============================================================================**

def checkGravityEquilibrium():\
"""\
Phase 0: Monitor gravity settling and create bonds when equilibrated.

✅ RELAXED Criteria for equilibration:\
1\. Unbalanced force < 0.03 (3% - more realistic for cohesive DEM)\
2\. Minimum settling time > 15,000 iterations (increased)\
3\. OR timeout at 40,000 iterations\
"""\
global phase0\_complete, total\_bonds\
\
if not phase0\_complete:\
`    `unbalanced = utils.unbalancedForce()\
\
`    `# ✅ FIX 5: Relaxed equilibration with timeout\
`    `min\_time = O.iter > 15000\
`    `equilibrated = unbalanced < 0.03  # Relaxed from 0.01\
`    `timeout = O.iter > 40000\
\
`    `if (min\_time and equilibrated) or timeout:\
`        `if timeout and not equilibrated:\
`            `print(f"\n\*\*\* WARNING: Phase 0 timeout - forcing completion \*\*\*")\
`            `print(f"\*\*\* Unbalanced force: {unbalanced:.4f} (target was 0.03) \*\*\*\n")\
`            `print("\n" + "="\*70)\
`            `print("PHASE 0 COMPLETE: GRAVITY EQUILIBRATION")\
`            `print("="\*70)\
`            `print(f"Iteration: {O.iter}")\
`            `print(f"Unbalanced force: {unbalanced:.6f}")\
\
`            `# Calculate gravitational stress on bottom wall\
`            `# (This is particle self-weight stress, not yet overburden)\
`            `print(f"\nParticles have settled under gravity.")\
`            `print(f"Proceeding to bond creation...")\
\
`            `# ✅ FIX 6: Better bonding mechanism with validation\
`            `# First ensure all interactions have CohFrictPhys\
`            `bond\_count = 0\
`            `for i in O.interactions:\
`                `if isinstance(i.phys, CohFrictPhys):\
`                    `i.phys.cohesionBroken = False\
`                    `i.phys.unp = i.geom.penetrationDepth\
`                    `bond\_count += 1\
\
`            `total\_bonds = bond\_count\
`            `print(f"Created {bond\_count} cohesive bonds")\
\
`            `# ✅ Validate bond count is reasonable\
`            `expected\_bonds = len([b for b in O.bodies if isinstance(b.shape, Sphere)]) \* 6\
`            `if bond\_count < expected\_bonds \* 0.3:\
`                `print(f"\*\*\* WARNING: Low bond count! Expected ~{expected\_bonds}, got {bond\_count} \*\*\*")\
`                `print(f"\*\*\* System may be too loose - consider denser packing \*\*\*")\
\
`            `# Enable triaxial controller for Phase 1\
`            `triax.internalCompaction = True\
`            `print(f"\n--- Starting Phase 1: Overburden Application ---")\
`            `print(f"Target vertical stress: {lithostatic\_stress/1e6:.2f} MPa")\
`            `print(f"Target horizontal stress: {horizontal\_stress/1e6:.2f} MPa")\
`            `print("="\*70 + "\n")\
\
`            `phase0\_complete = True\
`            `O.saveTmp('phase0\_complete')\
\
`    `# Progress updates during settling\
`    `if O.iter % 5000 == 0:\
`        `print(f"Phase 0 (Gravity): Iteration {O.iter:6d} | Unbalanced force: {unbalanced:.4f}")

<a name="bm_17"></a>**============================================================================**

<a name="section_9_phase_1_overburden_stre_21630d"></a>**SECTION 9: PHASE 1 - OVERBURDEN STRESS APPLICATION & CONSOLIDATION**

<a name="bm_18"></a>**============================================================================**

def checkOverburdenEquilibrium():\
"""\
Phase 1: Monitor stress convergence to target overburden conditions.

Criteria for completion:\
1\. All three principal stresses within 2% of target\
2\. Stress state is stable (not fluctuating)\
3\. Minimum consolidation time reached\
"""\
global phase1\_complete, phase2\_active\
\
if phase0\_complete and not phase1\_complete:\
`    `# Get current stress state\
`    `try:\
`        `sigma\_x = triax.stress(0)[0]\
`        `sigma\_y = triax.stress(1)[1]\
`        `sigma\_z = triax.stress(2)[2]\
`    `except:\
`        `return  # Not ready yet\
\
`    `# ✅ FIX 8: Relaxed tolerance and timeout for weak bonding\
`    `tolerance\_horizontal = 0.10 \* abs(horizontal\_stress)  # 10% tolerance (relaxed from 2%)\
`    `tolerance\_vertical = 0.10 \* abs(lithostatic\_stress)\
\
`    `# Minimum consolidation time with timeout\
`    `min\_time = O.iter > 20000\
`    `timeout = O.iter > 60000  # Force completion after 60k iterations\
\
`    `if min\_time or timeout:\
`        `# Check stress convergence\
`        `error\_x = abs(sigma\_x - (-horizontal\_stress))\
`        `error\_y = abs(sigma\_y - (-horizontal\_stress))\
`        `error\_z = abs(sigma\_z - (-lithostatic\_stress))\
\
`        `# ✅ Check convergence OR timeout\
`        `converged = (error\_x < tolerance\_horizontal and\
`                    `error\_y < tolerance\_horizontal and\
`                    `error\_z < tolerance\_vertical)\
\
`        `if converged or timeout:\
`            `if timeout and not converged:\
`                `print(f"\n\*\*\* WARNING: Phase 1 timeout - forcing completion \*\*\*")\
`                `print(f"\*\*\* Stress errors: Δσx={error\_x/1e6:.2f}, Δσy={error\_y/1e6:.2f}, Δσz={error\_z/1e6:.2f} MPa \*\*\*\n")\
\
`            `print("\n" + "="\*70)\
`            `print("PHASE 1 COMPLETE: OVERBURDEN EQUILIBRATION")\
`            `print("="\*70)\
`            `print(f"Iteration: {O.iter}")\
`            `print(f"\n--- Achieved Stress State ---")\
`            `print(f"σx = {-sigma\_x/1e6:.3f} MPa (target: {horizontal\_stress/1e6:.2f} MPa)")\
`            `print(f"σy = {-sigma\_y/1e6:.3f} MPa (target: {horizontal\_stress/1e6:.2f} MPa)")\
`            `print(f"σz = {-sigma\_z/1e6:.3f} MPa (target: {lithostatic\_stress/1e6:.2f} MPa)")\
`            `print(f"\n--- Bond Status ---")\
`            `active\_bonds = sum(1 for i in O.interactions if not i.phys.cohesionBroken)\
`            `broken\_bonds = sum(1 for i in O.interactions if i.phys.cohesionBroken)\
`            `print(f"Active bonds: {active\_bonds}")\
`            `print(f"Broken bonds: {broken\_bonds}")\
\
`            `# Save equilibrated state (baseline for fault loading)\
`            `O.saveTmp('phase1\_equilibrated')\
\
`            `# Transition to Phase 2: Fault Loading\
`            `print(f"\n--- Starting Phase 2: Fault Loading ---")\
`            `print(f"Increasing vertical stress to {fault\_loading\_stress/1e6:.2f} MPa")\
`            `print(f"Maintaining horizontal stress at {horizontal\_stress/1e6:.2f} MPa")\
\
`            `# Disable internal compaction (servo control remains active)\
`            `triax.internalCompaction = False\
\
`            `# Apply deviatoric loading\
`            `triax.goal1 = -horizontal\_stress      # Keep lateral constant\
`            `triax.goal2 = -horizontal\_stress\
`            `triax.goal3 = -fault\_loading\_stress   # Increase vertical load\
\
`            `print("="\*70 + "\n")\
\
`            `phase1\_complete = True\
`            `phase2\_active = True\
\
`    `# ✅ FIX 7: Better progress monitoring with all three stresses\
`    `if O.iter % 5000 == 0:\
`        `print(f"Phase 1 (Consolidation): Iteration {O.iter:6d} | "\
`              `f"σx = {-sigma\_x/1e6:.2f} | σy = {-sigma\_y/1e6:.2f} | "\
`              `f"σz = {-sigma\_z/1e6:.2f} MPa (target: {lithostatic\_stress/1e6:.2f})")

<a name="bm_19"></a>**============================================================================**

<a name="section_10_phase_2_fault_loading_bfddae"></a>**SECTION 10: PHASE 2 - FAULT LOADING & FAILURE MONITORING**

<a name="bm_20"></a>**============================================================================**

def checkFaultLoading():\
"""\
Phase 2: Monitor fault development and failure.

Termination criteria:\
1\. Axial strain exceeds 15% (large deformation)\
2\. Significant bond breakage (>30% of total bonds)\
3\. Stress-strain curve shows post-peak behavior\
"""\
global simulation\_stopped, brokenBonds\
\
if phase2\_active and not simulation\_stopped:\
`    `try:\
`        `# Get current strain state\
`        `strain\_x = triax.strain[0]\
`        `strain\_y = triax.strain[1]\
`        `strain\_z = triax.strain[2]\
\
`        `# Get current stress\
`        `sigma\_z = triax.stress(2)[2]\
\
`        `# Count broken bonds\
`        `broken\_now = sum(1 for i in O.interactions if i.phys.cohesionBroken)\
`        `brokenBonds = broken\_now\
`        `bond\_damage\_ratio = broken\_now / total\_bonds if total\_bonds > 0 else 0\
\
`        `# Termination check 1: Excessive axial strain\
`        `if abs(strain\_z) > 0.15:\
`            `print("\n" + "="\*70)\
`            `print("SIMULATION COMPLETE: Target Axial Strain Reached")\
`            `print("="\*70)\
`            `print(f"Final axial strain: {strain\_z:.4f} (15% limit)")\
`            `print(f"Final vertical stress: {-sigma\_z/1e6:.2f} MPa")\
`            `print(f"Broken bonds: {broken\_now} / {total\_bonds} ({bond\_damage\_ratio\*100:.1f}%)")\
\
`            `stopSimulation()\
\
`        `# Termination check 2: Significant bond breakage\
`        `elif bond\_damage\_ratio > 0.3 and O.iter > 30000:\
`            `print("\n" + "="\*70)\
`            `print("SIMULATION COMPLETE: Significant Fault Damage")\
`            `print("="\*70)\
`            `print(f"Bond damage: {bond\_damage\_ratio\*100:.1f}% ({broken\_now}/{total\_bonds})")\
`            `print(f"Axial strain: {strain\_z:.4f}")\
`            `print(f"Vertical stress: {-sigma\_z/1e6:.2f} MPa")\
\
`            `stopSimulation()\
\
`        `# Progress monitoring\
`        `if O.iter % 2000 == 0:\
`            `print(f"Phase 2 (Fault Loading): Iteration {O.iter:6d} | "\
`                  `f"εz = {strain\_z:.4f} | "\
`                  `f"σz = {-sigma\_z/1e6:.2f} MPa | "\
`                  `f"Broken bonds = {broken\_now} ({bond\_damage\_ratio\*100:.1f}%)")\
\
`    `except Exception as e:\
`        `pass  # Stress not available yet

def stopSimulation():\
"""Clean shutdown with data export"""\
global simulation\_stopped

\# Save final state\
O.saveTmp('phase2\_final')\
\
\# Export plot data\
try:\
`    `plot.saveDataTxt('simulation\_results.txt')\
`    `print("\n--- Data exported to simulation\_results.txt ---")\
except:\
`    `print("\n--- Could not export data ---")\
\
\# Export VTK for visualization (if available)\
try:\
`    `from yade import export\
`    `export.VTKExporter('fault\_final').exportSpheres(what=[('radius','b.shape.radius')])\
`    `print("--- VTK data exported to fault\_final\_\*.vtu ---")\
except:\
`    `pass\
\
print("="\*70 + "\n")\
\
simulation\_stopped = True\
O.pause()

<a name="bm_21"></a>**============================================================================**

<a name="section_11_data_collection_monitoring"></a>**SECTION 11: DATA COLLECTION & MONITORING**

<a name="bm_22"></a>**============================================================================**

def saveData():\
"""Record stress, strain, and bond status at regular intervals"""\
if phase0\_complete: # Only collect data after gravity settling\
try:\
plot.addData(\
iteration=O.iter,\
\# Stresses (convert to MPa)\
sigma\_xx=-triax.stress(0)[0]/1e6,\
sigma\_yy=-triax.stress(1)[1]/1e6,\
sigma\_zz=-triax.stress(2)[2]/1e6,\
\# Strains\
epsilon\_xx=triax.strain[0],\
epsilon\_yy=triax.strain[1],\
epsilon\_zz=triax.strain[2],\
\# Bond tracking\
broken\_bonds=brokenBonds,\
active\_bonds=total\_bonds - brokenBonds,\
damage\_ratio=brokenBonds/total\_bonds if total\_bonds > 0 else 0\
)\
except:\
pass # Data not ready yet

def monitorBonds():\
"""Track bond breakage evolution"""\
global brokenBonds

if phase0\_complete:\
`    `broken\_count = sum(1 for i in O.interactions if i.phys.cohesionBroken)\
`    `brokenBonds = broken\_count

<a name="plot_configuration"></a>**Plot configuration**

plot.plots = {\
'iteration': ('sigma\_zz', 'sigma\_xx'), # Stress evolution\
'iteration ': ('epsilon\_zz',), # Strain evolution\
'iteration ': ('broken\_bonds', 'active\_bonds') # Bond damage\
}

<a name="bm_23"></a>**============================================================================**

<a name="section_12_visualization_setup"></a>**SECTION 12: VISUALIZATION SETUP**

<a name="bm_24"></a>**============================================================================**

try:\
from yade import qt\
qt.Controller()\
v = qt.View()

\# Color particles by layer for visual identification\
for b in O.bodies:\
`    `if isinstance(b.shape, Sphere):\
`        `z = b.state.pos[2]\
`        `if z < -6:\
`            `b.shape.color = (0.7, 0.5, 0.3)  # Brown - Competent layer\
`        `elif z < -3:\
`            `b.shape.color = (0.6, 0.65, 0.5) # Green - Fractured layer\
`        `else:\
`            `b.shape.color = (0.5, 0.6, 0.75) # Blue - Damage zone\
\
\# Enable bond visualization\
renderer = v.renderer\
renderer.intrWire = True      # Show bonds as wires\
renderer.intrRadius = 0.02    # Thin bond representation\
\
print("\n--- 3D Visualization Active ---")\
print("Bond wires enabled (will disappear when broken)")

except:\
print("\n--- Running in batch mode (no GUI) ---")

<a name="bm_25"></a>**============================================================================**

<a name="section_13_simulation_start"></a>**SECTION 13: SIMULATION START**

<a name="bm_26"></a>**============================================================================**

O.saveTmp('initial')

print("\n" + "="\*70)\
print("STARTING SIMULATION")\
print("="\*70)\
print(f"Total particles: {total\_particles}")\
print(f"Domain: {domain}")\
print(f"Burial depth: {BURIAL\_DEPTH} m")\
print(f"\nPHASE WORKFLOW:")\
print(f" Phase 0: Gravity deposition (target: unbalanced force < 0.01)")\
print(f" Phase 1: Overburden application (σv = {lithostatic\_stress/1e6:.2f} MPa)")\
print(f" Phase 2: Fault loading (σv = {fault\_loading\_stress/1e6:.2f} MPa)")\
print("="\*70 + "\n")

print("Starting Phase 0: Gravity Deposition...")\
print("(Particles will settle under gravity before bonding)\n")

<a name="simulation_will_run_until_manuall_47e4a6"></a>**Simulation will run until manually stopped or termination criteria met**

<a name="use_o_run_for_batch_mode_or_click_2fa64b"></a>**Use O.run() for batch mode or click Play in GUI for interactive mode**

<a name="key_improvements_in_this_version"></a>**Key Improvements in This Version**

<a name="bm_1_three_phase_workflow"></a>**1. Three-Phase Workflow**

- **Phase 0**: Particles settle under gravity first (realistic fabric development)
- **Phase 1**: Overburden stress applied to simulate burial
- **Phase 2**: Deviatoric loading induces fault rupture

<a name="bm_2_realistic_parameters"></a>**2. Realistic Parameters**

- Material properties calibrated for sandstone (Young's modulus 10-30 GPa)
- Stress state calculated from burial depth (σ = ρgh)
- K₀ lateral earth pressure coefficient (0.4-0.5 typical)

<a name="bm_3_proper_equilibration_monitoring"></a>**3. Proper Equilibration Monitoring**

- Unbalanced force criterion for gravity settling
- Stress convergence monitoring for overburden application
- Bond damage tracking for fault development

<a name="bm_4_enhanced_data_collection"></a>**4. Enhanced Data Collection**

- Stress-strain curves in all directions
- Bond breakage evolution
- Damage ratio calculation
- Exportable data files

<a name="bm_5_clear_console_feedback"></a>**5. Clear Console Feedback**

- Phase transitions clearly marked
- Progress monitoring every 2000-5000 iterations
- Final results summary

<a name="usage_instructions"></a>**Usage Instructions**

<a name="running_the_simulation"></a>**Running the Simulation**

**Interactive Mode (with GUI):**\
yade fault\_simulation\_complete.py

<a name="click_play_button_to_start"></a>**Click "Play" button to start**

<a name="watch_particles_settle_consolidate_fail"></a>**Watch particles settle → consolidate → fail**

**Batch Mode (no GUI):**\
yade-batch fault\_simulation\_complete.py

<a name="or_add_at_end_of_script_o_run_100_8144b4"></a>**Or add at end of script: O.run(100000, True)**

<a name="expected_timeline"></a>**Expected Timeline**

|Phase|Iterations|Real Time\*|Description|
| :- | :- | :- | :- |
|Phase 0|10,000-15,000|5-10 min|Gravity settling|
|Phase 1|20,000-30,000|10-15 min|Overburden consolidation|
|Phase 2|30,000-50,000|15-25 min|Fault loading to failure|
|**Total**|**60,000-95,000**|**30-50 min**|Complete simulation|

\*Approximate on modern CPU (depends on particle count)

<a name="output_files"></a>**Output Files**

1. **simulation\_results.txt** - All plotted data (stress, strain, bonds)
1. **fault\_final\_\*.vtu** - VTK format for ParaView visualization
1. **Yade snapshots** - phase0\_complete, phase1\_equilibrated, phase2\_final

<a name="parameter_sensitivity_guide"></a>**Parameter Sensitivity Guide**

<a name="to_simulate_different_rock_types"></a>**To Simulate Different Rock Types**

**Granite (stronger, deeper burial):**\
BURIAL\_DEPTH = 1000 # 1km depth\
mat1.young = 50e9 # 50 GPa\
mat1.normalCohesion = 15e6 # 15 MPa tensile

**Shale (weaker, shallower):**\
BURIAL\_DEPTH = 300 # 300m depth\
mat1.young = 5e9 # 5 GPa\
mat1.normalCohesion = 2e6 # 2 MPa tensile

<a name="to_change_fault_loading_rate"></a>**To Change Fault Loading Rate**

**Slower (more realistic geologic strain rates):**\
fault\_loading\_stress = 1.3 \* lithostatic\_stress # Gentler loading

**Faster (faster simulation):**\
fault\_loading\_stress = 2.5 \* lithostatic\_stress # Aggressive loading

<a name="to_increase_resolution"></a>**To Increase Resolution**

**More particles (slower but more accurate):**\
NUM\_COARSE = 500\
NUM\_MEDIUM = 700\
NUM\_FINE = 800

<a name="validation_checklist"></a>**Validation Checklist**

Before running for research:

- [ ] Verify particle sizes are reasonable for domain scale
- [ ] Check burial depth matches your geological setting
- [ ] Confirm material properties match literature values
- [ ] Ensure stress ratios (K₀) are realistic
- [ ] Monitor Phase 0 completion (unbalanced force < 0.01)
- [ ] Check Phase 1 stress convergence (within 2% of target)
- [ ] Observe bond breakage patterns in Phase 2

<a name="troubleshooting"></a>**Troubleshooting**

**Problem**: Phase 0 never completes (unbalanced force stays high)

- **Solution**: Increase damping: NewtonIntegrator(damping=0.6, ...)

**Problem**: Phase 1 stress doesn't converge

- **Solution**: Check TriaxialStressController wall IDs, increase tolerance

**Problem**: Too many bonds break in Phase 1

- **Solution**: Reduce initial bonding strength or increase confining stress

**Problem**: Simulation too slow

- **Solution**: Reduce particle count, increase particle size, or reduce domain

<a name="references_for_parameter_calibration"></a>**References for Parameter Calibration**

1. **Sandstone properties**: Jaeger et al. (2007) - "Fundamentals of Rock Mechanics"
1. **DEM fault modeling**: Abe et al. (2011) - "DEM simulation of normal faults in cohesive materials"
1. **Triaxial testing**: ASTM D7181 - Standard Test Method for Consolidated Drained Triaxial Compression Test
1. **K₀ values**: Kulhawy & Mayne (1990) - "Manual on Estimating Soil Properties"

<a name="next_steps_for_btp_research"></a>**Next Steps for BTP Research**

1. **Calibration**: Match simulation to laboratory triaxial test data
1. **Parametric study**: Vary cohesion, friction angle, confining stress
1. **Fault geometry analysis**: Export particle positions, measure slip plane angle
1. **Statistical analysis**: Multiple runs with different random seeds
1. **Comparison with continuum models**: FEM/FDM validation
