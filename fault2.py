from yade import pack, plot, qt, utils
import numpy as np

# ============================================================================
# THRUST FAULT SIMULATION - 10-LAYER STRATIGRAPHY
# ============================================================================
# Simulates thrust/reverse fault development in layered rock under compression
#
# SIMULATION APPROACH:
# - Phase 0: Gravity settling of particles (reduced stiffness for stability)
# - Phase 1: Bond formation + weak zone creation + gradual stiffness restoration
# - Phase 2: Horizontal tectonic compression (thrust mechanics)
#
# KEY FEATURES:
# 1. Realistic material properties (2 GPa Young's, depth-consistent stresses)
# 2. Fixed bottom boundary (rigid basement) for thrust mechanics
# 3. Horizontal compression loading (not vertical) - proper thrust simulation
# 4. Fault nucleation zone at x=0 (reduced cohesion + friction)
# 5. Crack healing enabled (setCohesionOnNewContacts=True after Phase 0)
# 6. Consistent timestep (0.2×PWaveTimeStep) for numerical stability
# 7. Moderate damping (0.3) for quasi-static deformation
# ============================================================================

# ============================================================================
# SECTION 1: SIMULATION PARAMETERS (10-Layer Stratigraphy)
# ============================================================================

# Burial depth for stress calculation - MATCHES MODEL HEIGHT
BURIAL_DEPTH = 10  # meters (model represents 10m of rock column)

# Domain geometry (in meters)
DOMAIN_X = (-10, 10) # 20m width
DOMAIN_Y = (-10, 10) # 20m length
DOMAIN_Z = (0, 10) # 10m height (10 layers × 1m each)
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
        
        # Reduced by 10× for visible deformation (2 GPa instead of 20 GPa)
        target_young = 2.0e9 * depth_factor  # Store target value
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
        
        # Reduced by 10× for visible deformation (2 GPa instead of 20 GPa)
        target_young = 2.0e9 * depth_factor  # Store target value
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
print(f"Sandstone layers: 0, 2, 4, 6, 8 (Target Young's: 2.0-2.5 GPa)")
print(f"Shale layers: 1, 3, 5, 7, 9 (Target Young's: 2.0-2.5 GPa)")
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

# Calculate timestep AFTER particles with soft materials are added
# With reduced stiffness (0.001×), timestep needs to be computed from actual material properties
O.dt = 0.1 * PWaveTimeStep()  # Use 0.1× (not 0.2×) for extra stability with soft materials
print(f"\nInitial timestep (soft materials): {O.dt:.6e} s")
print("(Will be recomputed after stiffness restoration)")
print("-" * 70)

# ============================================================================
# SECTION 5: BOUNDARY WALLS (Bottom Fixed for Thrust Mechanics)
# ============================================================================

walls = aabbWalls(
    [(domain[0], domain[2], domain[4]), (domain[1], domain[3], domain[5])],
    thickness=0.5,
    material=materials[0]  # Use first sandstone layer material
)
wallIds = O.bodies.append(walls)

# Fix bottom wall to simulate rigid basement (thrust mechanics)
# Bottom wall is at z=0 (index depends on wall creation order)
for wallId in wallIds:
    wall = O.bodies[wallId]
    if wall.state.pos[2] <= domain[4] + 0.6:  # Bottom wall at z=0
        wall.state.blockedDOFs = 'xyzXYZ'  # Completely fixed
        print(f"Fixed bottom wall (ID: {wallId}) - simulates rigid basement")

# ============================================================================
# SECTION 6: SIMULATION ENGINES (Corrected for three-phase workflow)
# ============================================================================

O.engines = [
    ForceResetter(),

    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()]),

    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Box_Sphere_ScGeom6D()],

        # Bonding control: Initially disabled during Phase 0 settling
        # Will be enabled in Phase 1 to allow new contacts to bond (crack healing)
        [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(
            setCohesionNow=False,           # ✅ Wait for Phase 0 completion
            setCohesionOnNewContacts=False, # ✅ Will be enabled in Phase 1
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
        internalCompaction=False,        # ✅ DISABLED during Phase 0 - let gravity work naturally
        goal1=-0.01e6,                   # Very light lateral confining (0.01 MPa = 10 kPa) during settling
        goal2=-0.01e6,                   # Very light lateral confining (0.01 MPa = 10 kPa) during settling
        goal3=0,                         # No Z-axis control during Phase 0 - gravity handles settling
        thickness=0.5,                   # Match wall thickness
        maxStrainRate=(0.01, 0.01, 0.0), # Very slow wall movement to prevent explosion
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

# Timestep will be calculated after materials are added to simulation
# (See below after particle generation)

# ============================================================================
# SECTION 7: PHASE STATE VARIABLES
# ============================================================================
# Phase 0: Particle deposition by gravity settling
# Phase 1: Bond formation + weak zone creation + stiffness restoration
# Phase 2: Apply tectonic loading (compression for thrust/reverse faulting)

phase0_complete = False  # Gravity deposition complete
phase1_active = False    # Bond formation and stiffness restoration
phase1_start = 0         # Iteration when Phase 1 starts
phase1_complete = False  # Stiffness restoration complete
phase2_active = False    # Tectonic loading active
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

        # Use relaxed equilibration criterion:
        # - target unbalanced force < 0.01 (with reduced stiffness, should settle faster)
        # - kinetic energy < 500 (realistic for 4000 particles with soft materials)
        # - minimum iterations before checking: 8000 (reduced from 10000)
        # - forced timeout: 30000 iterations (reduced from 35000)
        min_iters = 8000
        timeout_iters = 30000
        target_unbalanced = 0.01  # Stricter with softer materials
        target_ke = 500.0  # Kinetic energy threshold (relaxed for large system)

        # Monitor kinetic energy for additional insight
        ke = utils.kineticEnergy()

        # Allow bonding if BOTH criteria met (unbalanced AND kinetic energy) or timeout
        if O.iter > min_iters:
            if (unbalanced < target_unbalanced and ke < target_ke) or O.iter >= timeout_iters:
                print("\n" + "="*70)
                print("PHASE 0 COMPLETE: Gravity Settling")
                print("="*70)
                print(f"Iteration: {O.iter}")
                print(f"Unbalanced force: {unbalanced:.6f}")
                print(f"Kinetic energy: {ke:.6e}")
                if O.iter >= timeout_iters:
                    print("(Forced completion due to timeout)")

                print(f"\nParticles have settled under gravity.")
                print(f"\n--- Starting PHASE 1: Bond Creation + Weak Zone ---")

                # NOW create cohesive bonds between settled particles
                bond_count = 0
                for i in O.interactions:
                    if isinstance(i.phys, CohFrictPhys):
                        i.phys.cohesionBroken = False
                        i.phys.unp = i.geom.penetrationDepth
                        bond_count += 1

                total_bonds = bond_count
                print(f"✓ Created {bond_count} cohesive bonds")
                
                # Enable bonding for new contacts (allows crack healing during loading)
                interactionPhys.setCohesionOnNewContacts = True
                print(f"✓ Enabled bonding for new contacts (crack healing allowed)")
                
                # Create a FAULT NUCLEATION ZONE to seed rupture
                # This is a vertical weak plane at x=0 (±1m width)
                # Reduces cohesion AND friction to simulate pre-existing fault damage
                weak_zone_count = 0
                for i in O.interactions:
                    if isinstance(i.phys, CohFrictPhys):
                        # Get interaction midpoint
                        pos1 = O.bodies[i.id1].state.pos
                        pos2 = O.bodies[i.id2].state.pos
                        mid_x = (pos1[0] + pos2[0]) / 2.0
                        
                        # If interaction is in fault nucleation zone (x between -1 and 1)
                        if abs(mid_x) < 1.0:
                            # Reduce cohesion to 30% (weaker bonds)
                            i.phys.normalAdhesion *= 0.3
                            i.phys.shearAdhesion *= 0.3
                            # Reduce friction angle to simulate fault gouge/damage
                            i.phys.tangensOfFrictionAngle *= 0.7  # ~30% friction reduction
                            weak_zone_count += 1
                
                print(f"✓ Created FAULT NUCLEATION ZONE: {weak_zone_count} bonds modified")
                print(f"  Location: Vertical plane at x=0 (±1m width)")
                print(f"  Properties: 30% cohesion, 30% friction reduction")
                print(f"  Purpose: Seed localized shear rupture\n")
                
                # Begin gradual stiffness restoration (avoids force explosion)
                global phase1_active, phase1_start, phase2_active
                phase1_active = True
                phase1_start = O.iter
                print(f"\n✓ Starting gradual stiffness restoration (5000 iterations)")
                print(f"  This prevents force explosion from instantaneous stiffness change")

                phase0_complete = True
                O.saveTmp('phase0_complete')

        # Progress updates during settling
        if O.iter % 5000 == 0:
            print(f"Phase 0 (Settling): Iteration {O.iter:6d} | Unbalanced: {unbalanced:.4f} | KE: {ke:.2e} (targets: {target_unbalanced:.4f}, <{target_ke:.0f})")

# ============================================================================
# SECTION 9: PHASE 1 - BOND FORMATION + STIFFNESS RESTORATION
# ============================================================================
# This phase gradually restores material stiffness after bonding to prevent
# force explosion. It's an essential transition between settling and loading.

def gradualStiffnessRestoration():
    """
    Phase 1 (Part 2): Gradually restore Young's modulus from reduced values 
    to target values over 5000 iterations to prevent force explosion.
    
    Uses 10 incremental steps of 10% increase each.
    """
    global phase1_active, phase1_complete, phase2_active
    
    if not phase1_active or phase1_complete:
        return
    
    iters_since_start = O.iter - phase1_start
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
            O.dt = 0.1 * PWaveTimeStep()
            
            print(f"Phase 1: Stiffness Restoration Step {current_step + 1}/{num_steps} | "
                  f"Young's at {target_fraction*100:.0f}% | dt: {O.dt:.6e} s")
    
    elif iters_since_start >= restoration_duration:
        # Final restoration to exact target values
        for idx, mat in enumerate(materials):
            mat.young = original_youngs[idx]
        
        # Use conservative timestep with full stiffness for stability
        O.dt = 0.15 * PWaveTimeStep()
        
        # Use moderate damping for quasi-static loading (Change 3)
        for eng in O.engines:
            if isinstance(eng, NewtonIntegrator):
                eng.damping = 0.3  # Moderate damping for quasi-static compression
        
        print("\n" + "="*70)
        print("PHASE 1 COMPLETE: Bonding + Stiffness Restoration")
        print("="*70)
        print(f"Young's modulus restored to target values (2.0-2.5 GPa)")
        print(f"Timestep: {O.dt:.6e} s (0.15×PWave, conservative) | Damping: 0.3")
        
        # NOW apply HORIZONTAL COMPRESSION for thrust fault mechanics (Change 2)
        # - Bottom boundary fixed (rigid basement)
        # - Horizontal compression from lateral walls moving inward
        # - Gravity maintained for overburden effect
        # - NO vertical stress control (let gravity handle it)
        
        triax.stressMask = 3  # Control only X,Y axes (horizontal compression)
        triax.goal1 = -horizontal_stress * 1.5  # Compress from X direction
        triax.goal2 = -horizontal_stress * 1.5  # Compress from Y direction  
        triax.goal3 = 0  # NO vertical control - gravity provides overburden
        triax.internalCompaction = False  # Disable compaction
        triax.maxStrainRate = (0.02, 0.02, 0.0)  # Slow horizontal compression, no vertical
        
        print(f"\n--- Starting PHASE 2: Tectonic Compression (Thrust Mechanics) ---")
        print(f"Loading mode: HORIZONTAL COMPRESSION (thrust fault simulation)")
        print(f"  - Bottom boundary: FIXED (rigid basement)")
        print(f"  - Horizontal stress: {horizontal_stress*1.5/1e6:.2f} MPa (1.5× confining)")
        print(f"  - Vertical loading: Gravity only (overburden effect)")
        print(f"  - Compression rate: 0.02 s⁻¹ (slow, quasi-static)")
        print(f"  - Fault nucleation zone: Vertical plane at x=0")
        print(f"→ Horizontal compression will drive thrust rupture along weak zone")
        print("="*70 + "\n")
        
        phase2_active = True
        phase1_complete = True
        phase1_active = False
        
        O.saveTmp('phase1_complete')

# ============================================================================
# SECTION 10: PHASE 2 - TECTONIC LOADING & FAULT RUPTURE MONITORING
# ============================================================================

def checkFaultLoading():
    """
    Phase 2: Monitor tectonic loading and fault rupture development.

    Termination criteria:
    1. Axial strain exceeds 15% (large deformation)
    2. Significant bond breakage (>30% of total bonds indicating fault formation)
    3. Stress drop indicating rupture completion
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

