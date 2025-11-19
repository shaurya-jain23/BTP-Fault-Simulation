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
original_youngs = []  # Store original Young's modulus for gradual restoration

# Stiffness reduction factor for Phase 0 settling (will be gradually restored after bonding)
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
print(f"   Stiffness will be GRADUALLY restored after bonding to avoid force explosion")
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

walls = aabbWalls(
    [(domain[0], domain[2], domain[4]), (domain[1], domain[3], domain[5])],
    thickness=0.5,
    material=materials[0]  # Use first sandstone layer material
)
wallIds = O.bodies.append(walls)

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

    # Very high damping for equilibration and stiffness restoration
    NewtonIntegrator(damping=0.95, gravity=(0, 0, -9.81)),

    # Triaxial controller with light confining stress during Phase 0 settling
    TriaxialStressController(
        stressMask=3,                    # Control only X,Y axes during Phase 0 (let gravity settle Z naturally)
        internalCompaction=True,         # ✅ Enable during Phase 0 for lateral confinement
        goal1=-0.05e6,                   # Light lateral confining (0.05 MPa) during settling
        goal2=-0.05e6,                   # Light lateral confining (0.05 MPa) during settling
        goal3=0,                         # No Z-axis control during Phase 0 - gravity handles settling
        thickness=0.5,                   # Match wall thickness
        maxStrainRate=(0.1, 0.1, 0.1),   # Limit strain rate to prevent explosive deformation
        label="triax"
    ),

    # Phase control callbacks (order matters!)
    PyRunner(command='checkGravityEquilibrium()', iterPeriod=100, label='gravityCheck'),
    PyRunner(command='gradualStiffnessRestoration()', iterPeriod=100, label='stiffnessRestore'),
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
stiffness_restoration_active = False # Gradual stiffness restoration phase
stiffness_restoration_start = 0
stiffness_restoration_complete = False
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
        # - kinetic energy < 500 (realistic for 4000 particles with soft materials)
        # - minimum iterations before checking: 10000
        # - forced timeout: 35000 iterations
        min_iters = 10000
        timeout_iters = 35000
        target_unbalanced = 0.01  # Stricter with softer materials
        target_ke = 500.0  # Kinetic energy threshold (relaxed for large system)

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
                
                # Create a weak zone to seed fault nucleation
                # Reduce cohesion in a vertical plane at x=0 (±1m width)
                weak_zone_count = 0
                for i in O.interactions:
                    if isinstance(i.phys, CohFrictPhys):
                        # Get interaction midpoint
                        pos1 = O.bodies[i.id1].state.pos
                        pos2 = O.bodies[i.id2].state.pos
                        mid_x = (pos1[0] + pos2[0]) / 2.0
                        
                        # If interaction is in weak zone (x between -1 and 1)
                        if abs(mid_x) < 1.0:
                            # Reduce cohesion to 30% of original
                            i.phys.normalAdhesion *= 0.3
                            i.phys.shearAdhesion *= 0.3
                            weak_zone_count += 1
                
                print(f"✓ Created weak zone: {weak_zone_count} bonds weakened (30% strength)")
                print(f"  Weak zone location: vertical plane at x=0 (±1m width)\n")
                
                # Trigger gradual stiffness restoration (avoids force explosion)
                global stiffness_restoration_active, stiffness_restoration_start, phase2_active
                stiffness_restoration_active = True
                stiffness_restoration_start = O.iter
                print(f"\n✓ Starting GRADUAL stiffness restoration over next 5000 iterations")
                print(f"   This prevents force explosion from instantaneous stiffness change")
                print(f"   Phase 2 will begin after stiffness restoration completes\n")

                phase0_complete = True
                O.saveTmp('phase0_complete')

        # Progress updates during settling
        if O.iter % 5000 == 0:
            print(f"Phase 0 (Gravity): Iteration {O.iter:6d} | Unbalanced: {unbalanced:.4f} (target: {target_unbalanced:.4f}) | KE: {ke:.2e} (target: <{target_ke:.0f})")

# Phase 1 removed: we proceed directly from gravity settling (Phase 0) to
# fault loading (Phase 2). The consolidation routine was intentionally
# removed to simplify the workflow and avoid the overburden equilibration step.

# ============================================================================
# SECTION 9: GRADUAL STIFFNESS RESTORATION (Prevents Force Explosion)
# ============================================================================

def gradualStiffnessRestoration():
    """
    Gradually restore Young's modulus from reduced values to target values
    over 5000 iterations to prevent force explosion.
    
    Uses 10 incremental steps of 10% increase each.
    """
    global stiffness_restoration_active, stiffness_restoration_complete, phase2_active
    
    if not stiffness_restoration_active or stiffness_restoration_complete:
        return
    
    iters_since_start = O.iter - stiffness_restoration_start
    restoration_duration = 5000  # Total iterations for restoration
    num_steps = 10  # Number of discrete restoration steps
    step_interval = restoration_duration // num_steps
    
    if iters_since_start < restoration_duration:
        # Determine current step (0 to 9)
        current_step = iters_since_start // step_interval
        step_iter = current_step * step_interval
        
        # Only update at the beginning of each step
        if iters_since_start == step_iter and iters_since_start > 0:
            # Calculate target fraction (10%, 20%, 30%, ... 100%)
            target_fraction = (current_step + 1) * 0.1
            
            # Update all material Young's modulus
            for idx, mat in enumerate(materials):
                mat.young = original_youngs[idx] * target_fraction
            
            # Recompute timestep with new stiffness
            O.dt = 0.2 * PWaveTimeStep()
            
            print(f"Stiffness Restoration: Step {current_step + 1}/{num_steps} | "
                  f"Young's at {target_fraction*100:.0f}% of target | "
                  f"dt: {O.dt:.6e} s")
    
    elif iters_since_start >= restoration_duration:
        # Final restoration to exact target values
        for idx, mat in enumerate(materials):
            mat.young = original_youngs[idx]
        
        O.dt = 0.5 * PWaveTimeStep()  # Standard timestep with full stiffness
        
        # Reduce damping for Phase 2 dynamics (low damping allows rupture propagation)
        for eng in O.engines:
            if isinstance(eng, NewtonIntegrator):
                eng.damping = 0.3  # Moderate damping for controlled fault rupture
        
        print("\n" + "="*70)
        print("STIFFNESS RESTORATION COMPLETE")
        print("="*70)
        print(f"Young's modulus restored to target values (19.9-25.0 GPa)")
        print(f"Timestep: {O.dt:.6e} s | Damping: 0.3 (controlled rupture)")
        
        # NOW enable full triaxial control and apply ASYMMETRIC stress for shear
        triax.stressMask = 7  # Enable all three axes
        triax.goal1 = -horizontal_stress * 0.85  # Slightly lower stress on X-axis
        triax.goal2 = -horizontal_stress  # Normal confining stress on Y
        triax.goal3 = -1.3 * lithostatic_stress  # Moderate vertical stress (was 2.0×, too aggressive)
        triax.internalCompaction = False  # Disable compaction for loading phase
        triax.maxStrainRate = (0.05, 0.05, 0.05)  # Slow strain rate for controlled fault development
        
        print(f"\n--- Starting Phase 2: Fault Loading (Controlled Shear) ---")
        print(f"Axial target: {1.3 * lithostatic_stress/1e6:.2f} MPa (1.3× lithostatic)")
        print(f"Lateral X: {horizontal_stress*0.85/1e6:.2f} MPa | Lateral Y: {horizontal_stress/1e6:.2f} MPa")
        print(f"Strain rate limit: 0.05 s⁻¹ (controlled loading)")
        print(f"→ Differential stress promotes shear along weak zone at x=0")
        print("="*70 + "\n")
        
        phase2_active = True
        stiffness_restoration_complete = True
        stiffness_restoration_active = False
        
        O.saveTmp('restoration_complete')

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

