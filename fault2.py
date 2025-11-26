from yade import pack, plot, utils
from yade import export
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

# Domain geometry (in meters) - QUASI-3D SLICE
DOMAIN_X = (-10, 10)  # 20m width (fault strike direction)
DOMAIN_Y = (-1, 1)    # 2m thickness (thin slice for computational efficiency)
DOMAIN_Z = (0, 10)    # 10m height (10 layers × 1m each)
domain = (DOMAIN_X[0], DOMAIN_X[1], DOMAIN_Y[0], DOMAIN_Y[1], DOMAIN_Z[0], DOMAIN_Z[1])

print("\n" + "="*70)
print("QUASI-3D SLICE GEOMETRY (Computational Efficiency)")
print("="*70)
print(f"Domain: {DOMAIN_X[1]-DOMAIN_X[0]:.0f}m (width) × "
      f"{DOMAIN_Y[1]-DOMAIN_Y[0]:.0f}m (thickness) × "
      f"{DOMAIN_Z[1]-DOMAIN_Z[0]:.0f}m (height)")
print(f"Volume: {(DOMAIN_X[1]-DOMAIN_X[0])*(DOMAIN_Y[1]-DOMAIN_Y[0])*(DOMAIN_Z[1]-DOMAIN_Z[0]):.0f} m³")
print(f"Expected packing fraction: ~60-65% (realistic rock mass)")
print("="*70 + "\n")

# Particle size - uniform smaller particles for better packing
PARTICLE_RADIUS = 0.07 # meters (uniform size for all layers) - reduced for better resolution

# Number of particles per 1m layer (scaled for smaller particle size)
# With r=0.10m (vs original 0.25m), we need ~15× more particles for proper packing
PARTICLES_PER_LAYER = 6000 # Total: 60,000 particles (increased from 400 for better resolution)

# Layer boundaries (10 layers, each 1m thick) - POSITIVE Z upward
LAYER_BOUNDARIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ============================================================================
# SECTION 2: 10-LAYER MATERIAL PROPERTIES (Alternating Sandstone/Shale)
# ============================================================================
# Realistic stratigraphy: Layers 0,2,4,6,8 = Sandstone | Layers 1,3,5,7,9 = Shale

materials = []
original_youngs = []  # Store original Young's modulus for gradual restoration

# Stiffness reduction factor for Phase 0 settling (will be gradually restored after bonding)
SETTLING_STIFFNESS_FACTOR = 1.0  # Use full stiffness from the start (no restoration needed)

for i in range(10):
    if i % 2 == 0:
        # SANDSTONE layers (even indices: 0, 2, 4, 6, 8)
        # Depth increases with layer number: adjust properties accordingly
        depth_factor = 1.0 + (i * 0.05)  # Slight increase with depth
        
        # Use reduced target stiffness for DEM stability (100 MPa instead of 2 GPa)
        target_young = 1.0e8 * depth_factor  # Store target value (scaled down)
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
        
        # Use reduced target stiffness for DEM stability (100 MPa instead of 2 GPa)
        target_young = 1.0e8 * depth_factor  # Store target value (scaled down)
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
print(f"Sandstone layers: 0, 2, 4, 6, 8 (Target Young's: 100-125 MPa)")
print(f"Shale layers: 1, 3, 5, 7, 9 (Target Young's: 100-125 MPa)")
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
# SECTION 5: BOUNDARY WALLS (Leak-Proof Extended Design)
# ============================================================================

# Define materials
wall_mat = materials[0] 
# Frictionless for Front/Back (Simulates infinite Plane Strain)
frictionless_mat = CohFrictMat(
    young=1e8, poisson=0.3, frictionAngle=0, density=0, label='frictionless'
)
O.materials.append(frictionless_mat)

# Wall thickness to prevent tunneling
THICK = 0.2 

# ----------------------------------------------------------------------------
# A. THE BOTTOM (Split into Footwall and Hanging Wall)
# ----------------------------------------------------------------------------
# Left Bottom (Fixed Footwall) - Range: x = -10 to 0
footwall = utils.box(
    center=(-5 - THICK/2, 0, -THICK/2),  
    extents=(5 + THICK/2, 1 + THICK, THICK/2), 
    fixed=True, 
    material=wall_mat, 
    wire=False
)
footwall_id = O.bodies.append(footwall)

# Right Bottom (Movable Hanging Wall) - Range: x = 0 to 15 (EXTRA LONG!)
# We make it 15m long so when it moves left by 3m, it still covers the x=10 boundary.
hanging_wall = utils.box(
    center=(7.5 + THICK/2, 0, -THICK/2), # Center shifted to cover 0 to 15
    extents=(7.5 + THICK/2, 1 + THICK, THICK/2), # Half-width is 7.5m
    fixed=False, # Must be False to allow movement
    material=wall_mat, 
    wire=False
)

# We append explicitly to get the ID
hanging_wall.state.blockedDOFs = 'xyzXYZ'
hanging_wall_id = O.bodies.append(hanging_wall) # <--- ID SAVED HERE

# ----------------------------------------------------------------------------
# B. THE SIDES (Fixed)
# ----------------------------------------------------------------------------
# Left Wall
left_wall = utils.box(
    center=(-10 - THICK/2, 0, 5), 
    extents=(THICK/2, 1 + THICK, 5 + THICK), 
    fixed=True, 
    material=wall_mat
)
O.bodies.append(left_wall)

# Right Wall
right_wall = utils.box(
    center=(10 + THICK/2, 0, 5), 
    extents=(THICK/2, 1 + THICK, 5 + THICK), 
    fixed=True, 
    material=wall_mat
)
O.bodies.append(right_wall)

# ----------------------------------------------------------------------------
# C. FRONT & BACK (The "Glass" - Frictionless)
# ----------------------------------------------------------------------------
# Front Wall (y = -1)
front_wall = utils.box(
    center=(0, -1 - THICK/2, 5), 
    extents=(10 + THICK, THICK/2, 5), 
    fixed=True, 
    material=frictionless_mat 
)
O.bodies.append(front_wall)

# Back Wall (y = +1)
back_wall = utils.box(
    center=(0, 1 + THICK/2, 5), 
    extents=(10 + THICK, THICK/2, 5), 
    fixed=True, 
    material=frictionless_mat 
)
O.bodies.append(back_wall)

# ----------------------------------------------------------------------------
# D. TOP (Cap)
# ----------------------------------------------------------------------------
top_wall = utils.box(
    center=(0, 0, 10 + THICK/2), 
    extents=(10 + THICK, 1 + THICK, THICK/2), 
    fixed=True, 
    material=wall_mat
)
O.bodies.append(top_wall)

print("Boundary walls created: Sealed Split-Box (Extended Hanging Wall)")
# ============================================================================
# SECTION 6: SIMULATION ENGINES (Corrected for three-phase workflow)
# ============================================================================
vtk_recorder = VTKRecorder(
    fileName='simulation_snapshots/3d_data-',
    recorders=['spheres', 'boxes', 'velocity', 'stress', 'ids', 'colors'],
    iterPeriod=2000, 
    label='vtk_recorder'
)

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

    # Damping handles the energy dissipation
    NewtonIntegrator(damping=0.7, gravity=(0, 0, -9.81)),  # Increased damping for Phase 0

    # REMOVED TRIAXIAL CONTROLLER COMPLETELY - Gravity + Rigid Walls create natural stress state

    # Phase control callbacks (order matters!)
    PyRunner(command='checkGravityEquilibrium()', iterPeriod=1000, label='gravityCheck'),
    # PyRunner(command='gradualStiffnessRestoration()', iterPeriod=1000, label='stiffnessRestore'),  # REMOVED: Using full stiffness from start
    PyRunner(command='checkFaultLoading()', iterPeriod=100, label='faultCheck'),
    # Drive hanging wall every iteration when Phase 2 is active
    PyRunner(command='driveHangingWall()', iterPeriod=1, label='driveHW'),

    # Data collection
    PyRunner(command='saveData()', iterPeriod=2000),
    PyRunner(command='monitorBonds()', iterPeriod=1000),
    vtk_recorder
]



# Then add a PyRunner to call it periodically
# (Add this to your O.engines list, after other PyRunners):

# O.engines += [
#     VTKRecorder(
#         fileName='simulation_snapshots/3d_data-',
#         iterPeriod=2000,
#         recorders=['spheres', 'intr', 'stress', 'velocity'],
#         label='vtk_recorder'
#     )
# ]
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
# Hanging wall id for kinematic driver (set when walls are created)
# hanging_wall_id = None
# Hanging wall velocity vector (set at Phase 2 start)
hanging_wall_vel = (0.0, 0.0, 0.0)

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
        # - kinetic energy < 1 (realistic for 6000 particles with soft materials)
        # - minimum iterations before checking: 20000 (reduced from 10000)
        # - forced timeout: 90000 iterations (reduced from 35000)
        min_iters = 20000
        timeout_iters = 90000
        target_unbalanced = 0.01  # Stricter with softer materials
        target_ke = 1.0  # Kinetic energy threshold (relaxed for large system)

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
                            # Reduce cohesion to 10% (much weaker bonds for easier rupture)
                            i.phys.normalAdhesion *= 0.1
                            i.phys.shearAdhesion *= 0.1
                            # Reduce friction angle to simulate fault gouge/damage
                            i.phys.tangensOfFrictionAngle *= 0.5  # 50% friction reduction
                            weak_zone_count += 1
                
                print(f"✓ Created FAULT NUCLEATION ZONE: {weak_zone_count} bonds modified")
                print(f"  Location: Vertical plane at x=0 (±1m width)")
                print(f"  Properties: 10% cohesion, 50% friction reduction")
                print(f"  Purpose: Seed localized shear rupture\n")
                
                # --- SKIP STIFFNESS RESTORATION (Factor=1.0 used) ---
                print("Skipping stiffness restoration (Factor=1.0 used).")
                
                # GO DIRECTLY TO PHASE 2
                global phase2_active, hanging_wall_vel
                phase2_active = True
                phase1_complete = True  # Mark Phase 1 as complete even though we skipped restoration
                
                # Configure hanging wall velocity (UPDATED SPEED: 0.05 m/s)
                dip_angle = np.radians(60)  # 60° thrust fault
                vel_mag = 0.05  # m/s - Quasi-static driving velocity (500x faster than before)
                vel_x = vel_mag * np.cos(dip_angle)
                vel_z = vel_mag * np.sin(dip_angle)
                hanging_wall_vel = (-vel_x, 0.0, vel_z)
                
                print(f"\n--- Starting PHASE 2: Kinematic Hanging Wall Drive (Thrust Mechanics) ---")
                print(f"Loading mode: Kinematic wall movement (no servo control)")
                print(f"  - Bottom footwall: FIXED (rigid basement)")
                print(f"  - Hanging wall velocity: {vel_mag*1e3:.1f} mm/s at {60}° dip")
                print(f"  - Material cohesion: 6-11 MPa (weak zone: 0.6-1.1 MPa)")
                print(f"  - Natural stress state from gravity + rigid walls")
                print(f"  - Fault nucleation zone: Vertical plane at x=0")
                print(f"→ Hanging wall will drive thrust rupture along weak zone")
                print("="*70 + "\n")

                phase0_complete = True
                O.saveTmp('phase0_complete')

        # Progress updates during settling
        if O.iter % 1000 == 0:
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
        print(f"Young's modulus restored to target values (100-125 MPa)")
        print(f"Timestep: {O.dt:.6e} s (0.15×PWave, conservative) | Damping: 0.3")
        
        # Configure hanging wall kinematics: dip and velocity magnitude
        global hanging_wall_vel
        dip_angle = np.radians(60)  # 60° thrust fault
        vel_mag = 1e-4  # m/s - slow, quasi-static driving velocity
        vel_x = vel_mag * np.cos(dip_angle)
        vel_z = vel_mag * np.sin(dip_angle)
        # Move hanging wall leftwards (-X) and upwards (+Z)
        hanging_wall_vel = (-vel_x, 0.0, vel_z)
        
        print(f"\n--- Starting PHASE 2: Kinematic Hanging Wall Drive (Thrust Mechanics) ---")
        print(f"Loading mode: Kinematic wall movement (no servo control)")
        print(f"  - Bottom footwall: FIXED (rigid basement)")
        print(f"  - Hanging wall velocity: {vel_mag*1e6:.1f} μm/s at {60}° dip")
        print(f"  - Material cohesion: 6-11 MPa (weak zone: 0.6-1.1 MPa)")
        print(f"  - Natural stress state from gravity + rigid walls")
        print(f"  - Fault nucleation zone: Vertical plane at x=0")
        print(f"→ Hanging wall will drive thrust rupture along weak zone")
        print("="*70 + "\n")
        
        hw_body = O.bodies[hanging_wall_id]
        hw_body.state.blockedDOFs = 'XYZ'
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
    1. Hanging wall displacement exceeds threshold (large deformation)
    2. Significant bond breakage (>30% of total bonds indicating fault formation)
    """
    global simulation_stopped, brokenBonds

    if phase2_active and not simulation_stopped:
        try:
            # Get hanging wall displacement
            hw_pos = O.bodies[hanging_wall_id].state.pos
            # UPDATED FORMULA: Initial center position accounting for wall thickness
            hw_disp_x = hw_pos[0] - 7.6  # Initial position was ~7.6m (7.5 + 0.1 half thickness)
            hw_disp_z = hw_pos[2] - 0.0  # Initial position was z=0.0

            # Count broken bonds
            broken_now = sum(1 for i in O.interactions if i.phys.cohesionBroken)
            brokenBonds = broken_now
            bond_damage_ratio = broken_now / total_bonds if total_bonds > 0 else 0

            # Termination check 1: Excessive hanging wall displacement (3m horizontal)
            if abs(hw_disp_x) > 3.0:
                print("\n" + "="*70)
                print("SIMULATION COMPLETE: Target Displacement Reached")
                print("="*70)
                print(f"Hanging wall displacement: X={hw_disp_x:.2f}m, Z={hw_disp_z:.2f}m")
                print(f"Broken bonds: {broken_now} / {total_bonds} ({bond_damage_ratio*100:.1f}%)")

                stopSimulation()

            # Termination check 2: Significant bond breakage
            elif bond_damage_ratio > 0.3 and O.iter > 30000:
                print("\n" + "="*70)
                print("SIMULATION COMPLETE: Significant Fault Damage")
                print("="*70)
                print(f"Bond damage: {bond_damage_ratio*100:.1f}% ({broken_now}/{total_bonds})")
                print(f"Hanging wall displacement: X={hw_disp_x:.2f}m, Z={hw_disp_z:.2f}m")

                stopSimulation()

            # Progress monitoring
            if O.iter % 2000 == 0:
                print(f"Phase 2 (Fault Loading): Iteration {O.iter:6d} | "
                      f"HW disp: X={hw_disp_x:.3f}m, Z={hw_disp_z:.3f}m | "
                      f"Broken bonds = {broken_now} ({bond_damage_ratio*100:.1f}%)")

        except Exception as e:
            pass  # Data not available yet


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
    """Record hanging wall position, bond status at regular intervals"""
    if phase0_complete:  # Only collect data after gravity settling
        try:
            # Get hanging wall position for tracking
            hw_pos = O.bodies[hanging_wall_id].state.pos if hanging_wall_id else (0, 0, 0)
            
            plot.addData(
                iteration=O.iter,
                # Hanging wall position
                hw_x=hw_pos[0],
                hw_z=hw_pos[2],
                # Bond tracking
                broken_bonds=brokenBonds,
                active_bonds=total_bonds - brokenBonds,
                damage_ratio=brokenBonds/total_bonds if total_bonds > 0 else 0
            )
        except:
            pass  # Data not ready yet

def monitorBonds():
    """Track bond breakage evolution"""
    global brokenBonds

    if phase0_complete:
        broken_count = sum(1 for i in O.interactions if i.phys.cohesionBroken)
        brokenBonds = broken_count

# Plot configuration
plot.plots = {
    'iteration': ('hw_x', 'hw_z'),  # Hanging wall position evolution
    'iteration ': ('broken_bonds', 'active_bonds')  # Bond damage
}

# ============================================================================
# SECTION 12: VISUALIZATION SETUP
# ============================================================================

# try:
#     from yade import qt
#     qt.Controller()
#     v = qt.View()

#     # Color particles by layer for visual identification
#     # Alternating colors for sandstone (warm) and shale (cool)
#     for b in O.bodies:
#         if isinstance(b.shape, Sphere):
#             z = b.state.pos[2]
            
#             # Determine layer based on z-position
#             layer_idx = None
#             for i in range(10):
#                 if LAYER_BOUNDARIES[i] <= z < LAYER_BOUNDARIES[i + 1]:
#                     layer_idx = i
#                     break
#             if layer_idx is None:
#                 layer_idx = 9
            
#             # Color by layer type
#             if layer_idx % 2 == 0:
#                 # Sandstone layers - warm colors (brown/orange gradient)
#                 intensity = 0.6 + (layer_idx / 10) * 0.3
#                 b.shape.color = (0.8 * intensity, 0.5 * intensity, 0.3 * intensity)
#             else:
#                 # Shale layers - cool colors (blue/gray gradient)
#                 intensity = 0.5 + (layer_idx / 10) * 0.3
#                 b.shape.color = (0.4 * intensity, 0.5 * intensity, 0.7 * intensity)

#     # Enable bond visualization
#     renderer = v.renderer
#     renderer.intrWire = True      # Show bonds as wires
#     renderer.intrRadius = 0.02    # Thin bond representation

#     print("\n--- 3D Visualization Active ---")
#     print("Bond wires enabled (will disappear when broken)")
#     print("Layer colors: Warm (sandstone) | Cool (shale)")

# except:
#     print("\n--- Running in batch mode (no GUI) ---")

# ============================================================================
# SECTION 13: SIMULATION START
# ============================================================================

O.saveTmp('initial')

print("\n" + "="*70)
print("STARTING SIMULATION - 10-LAYER STRATIGRAPHIC MODEL")
print("="*70)
print(f"Total particles: {total_particles} (6000 per layer × 10 layers)")
print(f"Domain: {domain}")
print(f"Burial depth: {BURIAL_DEPTH} m")
print(f"Layer structure: 5 Sandstone + 5 Shale (alternating)")
print(f"Particle radius: {PARTICLE_RADIUS} m (high resolution)")
print(f"\nPHASE WORKFLOW:")
print(f" Phase 0: Gravity deposition (target: unbalanced force < 0.01)")
print(f" Phase 1: Bond formation + stiffness restoration (100 MPa target)")
print(f" Phase 2: Kinematic hanging wall drive (thrust fault)")
print("="*70 + "\n")

print("Starting Phase 0: Gravity Deposition...")
print("(Particles will settle under gravity before bonding)\n")
O.run()
# Simulation will run until manually stopped or termination criteria met
# Use O.run() for batch mode or click Play in GUI for interactive mode

