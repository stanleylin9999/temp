import numpy as np

# ==============================================================================
# 1. 初始化 SimulationApp (GUI 模式)
# ==============================================================================
from isaacsim import SimulationApp

CONFIG = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)

# ==============================================================================
# 2. 導入 Isaac Sim 核心 API、USD 與機器人模組
# ==============================================================================
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, Vt


def create_cloth_mesh(
    stage,
    particle_system_path,
    prim_path="/World/ClothMesh",
    width=0.3,
    length=0.3,
    rows=12,
    cols=12,
):
    """建立自訂 3D 布料網格並套用 PhysX Particle Cloth 物理特性"""
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))

    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j / (cols - 1) - 0.5) * width
            y = (i / (rows - 1) - 0.5) * length
            points.append(Gf.Vec3f(x, y, 0.0))

    face_counts = [4] * ((rows - 1) * (cols - 1))
    face_indices = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            face_indices.extend([idx, idx + 1, idx + cols + 1, idx + cols])

    mesh.CreatePointsAttr().Set(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))

    # 布料懸空放置在水槽上方 (Z = 0.52m)
    xform = UsdGeom.Xformable(mesh)
    xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.52))

    mesh.CreateDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.95, 0.95, 0.95)])
    )
    prim = mesh.GetPrim()

    if hasattr(PhysxSchema, "PhysxParticleClothAPI"):
        PhysxSchema.PhysxParticleClothAPI.Apply(prim)
        rel = prim.CreateRelationship(
            "physxParticle:particleSystem", custom=False
        )
        rel.SetTargets([particle_system_path])
    elif hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
        PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(0.03)  # 乾布初始 0.03 kg

    return mesh, mass_api


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = world.stage

    # --------------------------------------------------------------------------
    # 關鍵修正：開啟 GPU Dynamics 並擴充 GPU PxGpuDynamicsMemoryConfig 容量
    # --------------------------------------------------------------------------
    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_broadphase_type("GPU")
    physics_scene_path = Sdf.Path(physics_context.prim_path)

    # 擴充 GPU 記憶體緩衝區配置 (徹底解決 PxGpuDynamicsMemoryConfig 記憶體爆滿)
    physx_scene_prim = stage.GetPrimAtPath(physics_scene_path)
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physx_scene_prim)

    physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10240)
    physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(10240)
    physx_scene_api.CreateGpuFoundLostPairsCapacityAttr().Set(10240)
    physx_scene_api.CreateGpuMaxRigidContactCountAttr().Set(1024000)
    physx_scene_api.CreateGpuMaxRigidPatchCountAttr().Set(163840)
    physx_scene_api.CreateGpuHeapCapacityAttr().Set(67108864)  # 64 MB GPU 堆疊容量

    print("[INFO] 1/5 水槽與 GPU PhysX 擴充記憶體配置完成...")

    # --------------------------------------------------------------------------
    # 3. 建構水槽 (0.8m x 0.8m x 0.5m)
    # --------------------------------------------------------------------------
    sink_color = np.array([0.75, 0.75, 0.8])
    wall_thick = 0.04
    w, d, h = 0.8, 0.8, 0.5
    sink_bottom_z_surface = wall_thick

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Sink/Bottom",
            name="sink_bottom",
            position=np.array([0.0, 0.0, wall_thick / 2]),
            scale=np.array([w, d, wall_thick]),
            color=sink_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Sink/Wall_PosX",
            name="sink_wall_pos_x",
            position=np.array([w / 2, 0.0, h / 2]),
            scale=np.array([wall_thick, d, h]),
            color=sink_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Sink/Wall_NegX",
            name="sink_wall_neg_x",
            position=np.array([-w / 2, 0.0, h / 2]),
            scale=np.array([wall_thick, d, h]),
            color=sink_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Sink/Wall_PosY",
            name="sink_wall_pos_y",
            position=np.array([0.0, d / 2, h / 2]),
            scale=np.array([w, wall_thick, h]),
            color=sink_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Sink/Wall_NegY",
            name="sink_wall_neg_y",
            position=np.array([0.0, -d / 2, h / 2]),
            scale=np.array([w, wall_thick, h]),
            color=sink_color,
        )
    )

    # --------------------------------------------------------------------------
    # 4. 建立 30,720 顆 PBD 流體粒子
    # --------------------------------------------------------------------------
    rest_offset = 0.008
    contact_offset = 0.010
    spacing_xy = 0.022
    spacing_z = 2 * rest_offset

    grid_x, grid_y, grid_z = 32, 32, 30
    total_particles = grid_x * grid_y * grid_z

    print(f"[INFO] 2/5 生成 PBD 流體粒子系統，總數: {total_particles:,} 顆...")

    particle_system_path = Sdf.Path("/World/FluidSystem")
    particle_system = PhysxSchema.PhysxParticleSystem.Define(
        stage, particle_system_path
    )
    particle_system.GetSimulationOwnerRel().SetTargets([physics_scene_path])
    particle_system.CreateParticleContactOffsetAttr().Set(contact_offset)
    particle_system.CreateRestOffsetAttr().Set(rest_offset)
    particle_system.CreateFluidRestOffsetAttr().Set(rest_offset)

    positions = []
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                px = (x - (grid_x - 1) / 2) * spacing_xy
                py = (y - (grid_y - 1) / 2) * spacing_xy
                pz = sink_bottom_z_surface + rest_offset + z * spacing_z
                positions.append(Gf.Vec3f(px, py, pz))

    points_path = Sdf.Path("/World/FluidSystem/Particles")
    points = UsdGeom.Points.Define(stage, points_path)
    points.CreatePointsAttr().Set(Vt.Vec3fArray(positions))
    points.CreateWidthsAttr().Set(
        Vt.FloatArray([spacing_xy * 0.8] * len(positions))
    )

    prim = points.GetPrim()
    PhysxSchema.PhysxParticleSetAPI.Apply(prim)
    rel = prim.CreateRelationship("physxParticle:particleSystem", custom=False)
    rel.SetTargets([particle_system_path])

    # --------------------------------------------------------------------------
    # 5. 生成布料與加載 SO-ARM101-USD.usd
    # --------------------------------------------------------------------------
    print("[INFO] 3/5 生成布料與配置 SO-ARM101 機械手臂...")
    cloth_mesh, cloth_mass_api = create_cloth_mesh(
        stage, particle_system_path, "/World/ClothMesh"
    )

    so_arm101_usd_path = "C:/Users/702A/Desktop/SO-ARM101-USD.usd"
    add_reference_to_stage(usd_path=so_arm101_usd_path, prim_path="/World/SO_ARM101")

    arm_robot = Robot(
        prim_path="/World/SO_ARM101",
        name="so_arm101",
        position=np.array([0.48, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    world.scene.add(arm_robot)

    # --------------------------------------------------------------------------
    # 6. 背景預先沉降 (Warm-up)
    # --------------------------------------------------------------------------
    print("[INFO] 4/5 執行背景預先沉降 (40 步)...")
    world.reset()

    for _ in range(40):
        world.step(render=False)

    print("[INFO] 5/5 模擬啟動！已擴充 GPU 緩衝，可穩定執行複雜流體與關節互動。")

    # 吸水動態參數
    dry_mass = 0.03
    max_wet_mass = 0.35
    current_mass = dry_mass
    water_surface_z = 0.45

    while simulation_app.is_running():
        world.step(render=True)

        cloth_points = cloth_mesh.GetPointsAttr().Get()
        if cloth_points and len(cloth_points) > 0:
            avg_z = np.mean([p[2] for p in cloth_points]) + 0.52
            if avg_z <= water_surface_z and current_mass < max_wet_mass:
                random_soak = np.random.uniform(0.0008, 0.0025)
                current_mass = min(max_wet_mass, current_mass + random_soak)
                cloth_mass_api.GetMassAttr().Set(current_mass)

                soak_ratio = (current_mass - dry_mass) / (
                    max_wet_mass - dry_mass
                )
                r = 0.95 * (1.0 - soak_ratio * 0.7)
                g = 0.95 * (1.0 - soak_ratio * 0.6)
                b = 0.95 * (1.0 - soak_ratio * 0.4)
                cloth_mesh.GetDisplayColorAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(r, g, b)])
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] 執行異常: {e}")
    finally:
        simulation_app.close()
