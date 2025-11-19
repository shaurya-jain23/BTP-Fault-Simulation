
from yade import pack, plot, qt, utils
import numpy as np

# ============================================================================
# SECTION 1: SIMULATION PARAMETERS (10-Layer Stratigraphy)
# ============================================================================

# Burial depth for stress calculation
BURIAL_DEPTH = 150 # meters (shallow crustal fault)

# Domain geometry (in meters)
DOMAIN_X = (-10, 10) # 20m width
DOMAIN_Y = (-10, 10) # 20m length
DOMAIN_Z = (0, 10) # 10m height (10 layers × 1m each) - POSITIVE Z for proper settling
domain = (DOMAIN_X[0], DOMAIN_X[1], DOMAIN_Y[0], DOMAIN_Y[1], DOMAIN_Z[0], DOMAIN_Z[1])

# Particle size - uniform smaller particles for better packing
PARTICLE_RADIUS = 0.25 # meters (uniform size for all layers)

# Number of particles per 1m layer (adjust to fill domain properly)
PARTICLES_PER_LAYER = 400 # Total: 4000 particles

# Layer boundaries (10 layers, each 1m thick) - POSITIVE Z upward
LAYER_BOUNDARIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ============================================================================
# SECTION 2: 10-LAYER MATERIAL PROPERTIES (Alternating Sandstone/Shale)
# ============================================================================
# Realistic stratigraphy: Layers 0,2,4,6,8 = Sandstone | Layers 1,3,5,7,9 = Shale

materials = []
original_youngs = []  # Store original Young's modulus for restoration after settling

# Stiffness reduction factor for Phase 0 settling (restored after bonding)
SETTLING_STIFFNESS_FACTOR = 0.001  # Use 0.1% of target stiffness during settling

for i in range(10):
    if i % 2 == 0:
        # SANDSTONE layers (even indices: 0, 2, 4, 6, 8)
        # Depth increases with layer number: adjust properties accordingly
        depth_factor = 1.0 + (i * 0.05)  # Slight increase with depth
        
        target_young = 19.9e9 * depth_factor  # Store target value
        original_youngs.append(target_young)
        
        mat = CohFrictMat(
            young=target_young * SETTLING_STIFFNESS_FACTOR,  # Reduced for settling
            poisson=0.28,                      # Poisson's ratio
            frictionAngle=np.radians(35),      # 35° internal friction angle
            density=2400,                      # kg/m³
            isCohesive=True,                   # Enable bonding
            normalCohesion=8.04e6 * depth_factor,  # 8.04 MPa tensile strength
            shearCohesion=11.49e6 * depth_factor,  # 11.49 MPa cohesion
            label=f'Sandstone_Layer{i}'
        )
    else:
        # SHALE layers (odd indices: 1, 3, 5, 7, 9)
        depth_factor = 1.0 + (i * 0.05)
        
        target_young = 20e9 * depth_factor  # Store target value
        original_youngs.append(target_young)
        
        mat = CohFrictMat(
            young=target_young * SETTLING_STIFFNESS_FACTOR,  # Reduced for settling
            poisson=0.35,                      # 0.35 (clay-rich)
            frictionAngle=np.radians(28),      # 28° (lower than sandstone)
            density=2600,                      # kg/m³ (denser, clay-rich)
            isCohesive=True,
            normalCohesion=6.3e6 * depth_factor,   # 6.3 MPa tensile strength
            shearCohesion=9e6 * depth_factor,      # 9 MPa cohesion
            label=f'Shale_Layer{i}'
        )
    
    materials.append(mat)
    O.materials.append(mat)

# Calculate average density for stress calculations
avg_density = sum(mat.density for mat in materials) / len(materials)

print("\n--- Material Properties Summary ---")
print(f"Total layers: {len(materials)}")
print(f"Sandstone layers: 0, 2, 4, 6, 8 (Target Young's: 19.9-24.9 GPa)")
print(f"Shale layers: 1, 3, 5, 7, 9 (Target Young's: 20.0-25.0 GPa)")
print(f"Average density: {avg_density:.0f} kg/m³")
print(f"\n⚠️  Phase 0 uses reduced stiffness (factor: {SETTLING_STIFFNESS_FACTOR})")
print(f"   Stiffness will be restored to target values after bonding")
print("-" * 70)

# ============================================================================
# SECTION 3: STRESS STATE CALCULATION (Depth-based, realistic)
# ============================================================================

# Gravitational acceleration
g = 9.81 # m/s²

# Lithostatic (overburden) vertical stress: σ_v = ρ × g × h
lithostatic_stress = avg_density * g * BURIAL_DEPTH # Pa
print(f"\n--- Calculated Stress State at {BURIAL_DEPTH}m Depth ---")
print(f"Lithostatic vertical stress: {lithostatic_stress/1e6:.2f} MPa")

# Lateral stress coefficient (at-rest earth pressure)
# K0 = 1 - sin(φ) for normally consolidated soil/rock
avg_friction_angle = (35 + 32 + 28) / 3 # Average friction angle
K0 = 1 - np.sin(np.radians(avg_friction_angle))
horizontal_stress = K0 * lithostatic_stress
print(f"K0 coefficient: {K0:.3f}")
print(f"Horizontal confining stress: {horizontal_stress/1e6:.2f} MPa")
print(f"Stress ratio (σh/σv): {K0:.3f}")

# Deviatoric stress for fault loading (Phase 2)
# Typical fault simulation: increase vertical stress to 1.5-2.0 times confining
fault_loading_stress = 1.8 * lithostatic_stress
print(f"Fault loading stress (Phase 2): {fault_loading_stress/1e6:.2f} MPa")
print("-" * 70)

# ============================================================================
# SECTION 4: PARTICLE PACKING (10-layer stratification)
# ============================================================================

print("\n--- Generating 10-Layer Particle Packing ---")
sp = pack.SpherePack()

# Generate particles for each 1m-thick layer
for layer_idx in range(10):
    z_bottom = LAYER_BOUNDARIES[layer_idx]
    z_top = LAYER_BOUNDARIES[layer_idx + 1]
    
    layer_type = "Sandstone" if layer_idx % 2 == 0 else "Shale"
    
    sp.makeCloud(
        (domain[0], domain[2], z_bottom),
        (domain[1], domain[3], z_top),
        rMean=PARTICLE_RADIUS,
        rRelFuzz=0.15,  # Tighter packing for better layer stability
        num=PARTICLES_PER_LAYER,
        seed=42000 + layer_idx
    )
    
    print(f"Layer {layer_idx} ({layer_type}): {PARTICLES_PER_LAYER} particles | "
          f"Z-range: [{z_bottom:.1f}, {z_top:.1f}] m | r={PARTICLE_RADIUS}m")

# Insert particles into simulation with material assignment by layer
particle_count_by_layer = [0] * 10

for center, radius in sp:
    z = center[2]
    
    # Determine which layer this particle belongs to
    layer_idx = None
    for i in range(10):
        if LAYER_BOUNDARIES[i] <= z < LAYER_BOUNDARIES[i + 1]:
            layer_idx = i
            break
    
    # Fallback for particles exactly at top boundary
    if layer_idx is None:
        layer_idx = 9
    
    # Assign appropriate material
    mat = materials[layer_idx]
    O.bodies.append(sphere(center, radius, material=mat))
    particle_count_by_layer[layer_idx] += 1

total_particles = sum(particle_count_by_layer)
print(f"\n--- Particle Distribution Summary ---")
for i, count in enumerate(particle_count_by_layer):
    layer_type = "Sandstone" if i % 2 == 0 else "Shale"
    print(f"Layer {i} ({layer_type}): {count} particles")
print(f"Total particles generated: {total_particles}")
print("-" * 70)

# ============================================================================
# SECTION 5: BOUNDARY WALLS
# ============================================================================
# Note: Walls are automatically created by TriaxialStressController
# Do NOT use aabbWalls - it conflicts with triax movable walls

# ============================================================================
# SECTION 6: SIMULATION ENGINES (Corrected for three-phase workflow)
# ============================================================================

O.engines = [
    ForceResetter(),

    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()]),

    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Box_Sphere_ScGeom6D()],

        # CRITICAL: Do NOT bond immediately - allow gravity settling first
        [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(
            setCohesionNow=False,           # ✅ Wait for Phase 0 completion
            setCohesionOnNewContacts=False, # ✅ Manual bonding control
            label='interactionPhys'
        )],

        [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(
            useIncrementalForm=True,
            always_use_moment_law=False,
            label='cohesiveLaw'
        )]
    ),

    # Very high damping for faster equilibration during settling
    NewtonIntegrator(damping=0.9, gravity=(0, 0, -9.81)),

    # Triaxial controller with light confining stress during Phase 0 settling
    TriaxialStressController(
        stressMask=7,                    # Control all three axes
        internalCompaction=True,         # ✅ Enable during Phase 0 for lateral confinement
        goal1=-0.05e6,                   # Light lateral confining (0.05 MPa) during settling
        goal2=-0.05e6,                   # Light lateral confining (0.05 MPa) during settling
        goal3=-lithostatic_stress,       # Vertical (Z)
        thickness=0.5,                   # Wall thickness
        wall_frictionAngle=np.radians(30),  # Wall friction angle
        max_vel=0.1,                     # Maximum wall velocity (m/s)
        label="triax"
    ),

    # Phase control callbacks (order matters!)
    PyRunner(command='checkGravityEquilibrium()', iterPeriod=100, label='gravityCheck'),
    PyRunner(command='checkFaultLoading()', iterPeriod=100, label='faultCheck'),

    # Data collection
    PyRunner(command='saveData()', iterPeriod=500),
    PyRunner(command='monitorBonds()', iterPeriod=1000)
]

O.dt = 0.2 * PWaveTimeStep()  # Reduced timestep for stability

# ============================================================================
# SECTION 7: PHASE STATE VARIABLES
# ============================================================================

phase0_complete = False # Gravity deposition
phase2_active = False # Fault loading
simulation_stopped = False

brokenBonds = 0
total_bonds = 0

# ============================================================================
# SECTION 8: PHASE 0 - GRAVITY DEPOSITION & EQUILIBRATION
# ============================================================================

def checkGravityEquilibrium():
    """
    Phase 0: Monitor gravity settling and create bonds when equilibrated.

    Criteria for equilibration:
    1. Unbalanced force < 0.01 (1% of system)
    2. Minimum settling time > 20,000 iterations
    3. Particles have reached mechanical equilibrium
    """
    global phase0_complete, total_bonds

    if not phase0_complete:
        unbalanced = utils.unbalancedForce()

        # Use relaxed equilibration criterion as requested:
        # - target unbalanced force < 0.01 (with reduced stiffness, should settle faster)
        # - kinetic energy < 50 (particles nearly at rest)
        # - minimum iterations before checking: 10000
        # - forced timeout: 40000 iterations
        min_iters = 10000
        timeout_iters = 40000
        target_unbalanced = 0.01  # Stricter with softer materials
        target_ke = 50.0  # Kinetic energy threshold

        # Monitor kinetic energy for additional insight
        ke = utils.kineticEnergy()

        # Allow bonding if BOTH criteria met (unbalanced AND kinetic energy) or timeout
        if O.iter > min_iters:
            if (unbalanced < target_unbalanced and ke < target_ke) or O.iter >= timeout_iters:
                print("\n" + "="*70)
                print("PHASE 0 COMPLETE: GRAVITY EQUILIBRATION")
                print("="*70)
                print(f"Iteration: {O.iter}")
                print(f"Unbalanced force: {unbalanced:.6f}")
                print(f"Kinetic energy: {ke:.6e}")
                if O.iter >= timeout_iters:
                    print("(Forced completion due to timeout)")

                # Calculate gravitational stress on bottom wall
                # (This is particle self-weight stress, not yet overburden)
                print(f"\nParticles have settled under gravity.")
                print(f"Proceeding to bond creation...")

                # NOW create cohesive bonds between settled particles
                bond_count = 0
                for i in O.interactions:
                    if isinstance(i.phys, CohFrictPhys):
                        i.phys.cohesionBroken = False
                        i.phys.unp = i.geom.penetrationDepth
                        bond_count += 1

                total_bonds = bond_count
                print(f"Created {bond_count} cohesive bonds")
                
                # CRITICAL: Restore original Young's modulus values
                print(f"\nRestoring material stiffness to target values...")
                for idx, mat in enumerate(materials):
                    mat.young = original_youngs[idx]
                print(f"✓ Material stiffness restored (Young's: 19.9-25.0 GPa)")
                
                # Recompute timestep with restored stiffness
                O.dt = 0.5 * PWaveTimeStep()
                print(f"✓ Timestep recomputed: {O.dt:.6e} s\n")
                
                # Reduce damping for dynamic fault loading
                for eng in O.engines:
                    if isinstance(eng, NewtonIntegrator):
                        eng.damping = 0.4
                        print(f"✓ Reduced damping to {eng.damping} for Phase 2 dynamics\n")

                # Directly start Phase 2 (fault loading) per user request
                triax.internalCompaction = False
                # Set lateral goals to confining stress and axial to 1.5× lithostatic stress
                triax.goal1 = -horizontal_stress  # Restore confining stress
                triax.goal2 = -horizontal_stress  # Restore confining stress
                triax.goal3 = -1.5 * lithostatic_stress
                print(f"\n--- Starting Phase 2: Fault Loading (direct) ---")
                print(f"Axial target: {1.5 * lithostatic_stress/1e6:.2f} MPa (1.5× lithostatic)")
                print(f"Lateral targets: {horizontal_stress/1e6:.2f} MPa (confining stress)")
                print("="*70 + "\n")

                phase0_complete = True
                phase2_active = True
                O.saveTmp('phase0_complete')

        # Progress updates during settling
        if O.iter % 5000 == 0:
            print(f"Phase 0 (Gravity): Iteration {O.iter:6d} | Unbalanced: {unbalanced:.4f} (target: {target_unbalanced:.4f}) | KE: {ke:.2e} (target: {target_ke:.0f})")

# Phase 1 removed: we proceed directly from gravity settling (Phase 0) to
# fault loading (Phase 2). The consolidation routine was intentionally
# removed to simplify the workflow and avoid the overburden equilibration step.

# ============================================================================
# SECTION 10: PHASE 2 - FAULT LOADING & FAILURE MONITORING
# ============================================================================

def checkFaultLoading():
    """
    Phase 2: Monitor fault development and failure.

    Termination criteria:
    1. Axial strain exceeds 15% (large deformation)
    2. Significant bond breakage (>30% of total bonds)
    3. Stress-strain curve shows post-peak behavior
    """
    global simulation_stopped, brokenBonds

    if phase2_active and not simulation_stopped:
        try:
            # Get current strain state
            strain_x = triax.strain[0]
            strain_y = triax.strain[1]
            strain_z = triax.strain[2]

            # Get current stress
            sigma_z = triax.stress(2)[2]

            # Count broken bonds
            broken_now = sum(1 for i in O.interactions if i.phys.cohesionBroken)
            brokenBonds = broken_now
            bond_damage_ratio = broken_now / total_bonds if total_bonds > 0 else 0

            # Termination check 1: Excessive axial strain
            if abs(strain_z) > 0.15:
                print("\n" + "="*70)
                print("SIMULATION COMPLETE: Target Axial Strain Reached")
                print("="*70)
                print(f"Final axial strain: {strain_z:.4f} (15% limit)")
                print(f"Final vertical stress: {-sigma_z/1e6:.2f} MPa")
                print(f"Broken bonds: {broken_now} / {total_bonds} ({bond_damage_ratio*100:.1f}%)")

                stopSimulation()

            # Termination check 2: Significant bond breakage
            elif bond_damage_ratio > 0.3 and O.iter > 30000:
                print("\n" + "="*70)
                print("SIMULATION COMPLETE: Significant Fault Damage")
                print("="*70)
                print(f"Bond damage: {bond_damage_ratio*100:.1f}% ({broken_now}/{total_bonds})")
                print(f"Axial strain: {strain_z:.4f}")
                print(f"Vertical stress: {-sigma_z/1e6:.2f} MPa")

                stopSimulation()

            # Progress monitoring
            if O.iter % 2000 == 0:
                print(f"Phase 2 (Fault Loading): Iteration {O.iter:6d} | "
                      f"εz = {strain_z:.4f} | "
                      f"σz = {-sigma_z/1e6:.2f} MPa | "
                      f"Broken bonds = {broken_now} ({bond_damage_ratio*100:.1f}%)")

        except Exception as e:
            pass  # Stress not available yet

def stopSimulation():
    """Clean shutdown with data export"""
    global simulation_stopped

    # Save final state
    O.saveTmp('phase2_final')

    # Export plot data
    try:
        plot.saveDataTxt('simulation_results.txt')
        print("\n--- Data exported to simulation_results.txt ---")
    except:
        print("\n--- Could not export data ---")

    # Export VTK for visualization (if available)
    try:
        from yade import export
        export.VTKExporter('fault_final').exportSpheres(what=[('radius','b.shape.radius')])
        print("--- VTK data exported to fault_final_*.vtu ---")
    except:
        pass

    print("="*70 + "\n")

    simulation_stopped = True
    O.pause()

# ============================================================================
# SECTION 11: DATA COLLECTION & MONITORING
# ============================================================================

def saveData():
    """Record stress, strain, and bond status at regular intervals"""
    if phase0_complete: # Only collect data after gravity settling
        try:
            plot.addData(
                iteration=O.iter,
                # Stresses (convert to MPa)
                sigma_xx=-triax.stress(0)[0]/1e6,
                sigma_yy=-triax.stress(1)[1]/1e6,
                sigma_zz=-triax.stress(2)[2]/1e6,
                # Strains
                epsilon_xx=triax.strain[0],
                epsilon_yy=triax.strain[1],
                epsilon_zz=triax.strain[2],
                # Bond tracking
                broken_bonds=brokenBonds,
                active_bonds=total_bonds - brokenBonds,
                damage_ratio=brokenBonds/total_bonds if total_bonds > 0 else 0
            )
        except:
            pass # Data not ready yet

def monitorBonds():
    """Track bond breakage evolution"""
    global brokenBonds

    if phase0_complete:
        broken_count = sum(1 for i in O.interactions if i.phys.cohesionBroken)
        brokenBonds = broken_count

# Plot configuration
plot.plots = {
    'iteration': ('sigma_zz', 'sigma_xx'), # Stress evolution
    'iteration ': ('epsilon_zz',), # Strain evolution
    'iteration  ': ('broken_bonds', 'active_bonds') # Bond damage
}

# ============================================================================
# SECTION 12: VISUALIZATION SETUP
# ============================================================================

try:
    from yade import qt
    qt.Controller()
    v = qt.View()

    # Color particles by layer for visual identification
    # Alternating colors for sandstone (warm) and shale (cool)
    for b in O.bodies:
        if isinstance(b.shape, Sphere):
            z = b.state.pos[2]
            
            # Determine layer based on z-position
            layer_idx = None
            for i in range(10):
                if LAYER_BOUNDARIES[i] <= z < LAYER_BOUNDARIES[i + 1]:
                    layer_idx = i
                    break
            if layer_idx is None:
                layer_idx = 9
            
            # Color by layer type
            if layer_idx % 2 == 0:
                # Sandstone layers - warm colors (brown/orange gradient)
                intensity = 0.6 + (layer_idx / 10) * 0.3
                b.shape.color = (0.8 * intensity, 0.5 * intensity, 0.3 * intensity)
            else:
                # Shale layers - cool colors (blue/gray gradient)
                intensity = 0.5 + (layer_idx / 10) * 0.3
                b.shape.color = (0.4 * intensity, 0.5 * intensity, 0.7 * intensity)

    # Enable bond visualization
    renderer = v.renderer
    renderer.intrWire = True      # Show bonds as wires
    renderer.intrRadius = 0.02    # Thin bond representation

    print("\n--- 3D Visualization Active ---")
    print("Bond wires enabled (will disappear when broken)")
    print("Layer colors: Warm (sandstone) | Cool (shale)")

except:
    print("\n--- Running in batch mode (no GUI) ---")

# ============================================================================
# SECTION 13: SIMULATION START
# ============================================================================

O.saveTmp('initial')

print("\n" + "="*70)
print("STARTING SIMULATION - 10-LAYER STRATIGRAPHIC MODEL")
print("="*70)
print(f"Total particles: {total_particles} (400 per layer × 10 layers)")
print(f"Domain: {domain}")
print(f"Burial depth: {BURIAL_DEPTH} m")
print(f"Layer structure: 5 Sandstone + 5 Shale (alternating)")
print(f"\nPHASE WORKFLOW:")
print(f" Phase 0: Gravity deposition (target: unbalanced force < 0.01)")
print(f" Phase 1: Overburden application (σv = {lithostatic_stress/1e6:.2f} MPa)")
print(f" Phase 2: Fault loading (σv = {fault_loading_stress/1e6:.2f} MPa)")
print("="*70 + "\n")

print("Starting Phase 0: Gravity Deposition...")
print("(Particles will settle under gravity before bonding)\n")

# Simulation will run until manually stopped or termination criteria met
# Use O.run() for batch mode or click Play in GUI for interactive mode

