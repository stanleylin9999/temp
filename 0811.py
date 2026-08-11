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
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
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

    # 布料初始懸空放置在水槽上方 (Z = 0.73m)
    xform = UsdGeom.Xformable(mesh)
    xform.AddTranslateOp().Set(Gf.Vec3f(0.19, 0.15175, 0.73))

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


def attach_cloth_to_arm(stage, cloth_path, robot_prim_path):
    """自動尋找手臂末端剛體 Link，並建立 PhysX 物理固定連接 (Attachment)"""
    ee_link_path = None
    # 搜尋手臂下最後一個包含 RigidBodyAPI 的 Link 作為末端夾具
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if path_str.startswith(str(robot_prim_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                ee_link_path = prim.GetPath()

    if ee_link_path is not None:
        attachment_path = Sdf.Path("/World/ClothAttachment")
        
        # 修正：使用 PhysxPhysicsAttachment (Prim Schema，非 API Schema)
        if hasattr(PhysxSchema, "PhysxPhysicsAttachment"):
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(
                stage, attachment_path
            )
            attachment.CreateActor0Rel().SetTargets([ee_link_path])
            attachment.CreateActor1Rel().SetTargets([Sdf.Path(cloth_path)])

            # 套用 Auto Attachment API，讓 PhysX 自動尋找重疊區域綁定接點
            if hasattr(PhysxSchema, "PhysxAutoAttachmentAPI"):
                auto_attach = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
                auto_attach.CreateEnableAutoAttachmentAttr().Set(True)

            print(f"[INFO] 已將布料成功固定至手臂末端剛體: {ee_link_path}")
        else:
            print("[WARN] 當前版本未支援 PhysxPhysicsAttachment 類別。")
    else:
        print("[WARN] 未找到手臂末端剛體 Link，布料將進行自由落體。")


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
    # 5. 生成布料與加載 SO-ARM101 機械手臂，並建立物理連結
    # --------------------------------------------------------------------------
    print("[INFO] 3/5 生成布料與配置 SO-ARM101 機械手臂...")
    cloth_mesh, cloth_mass_api = create_cloth_mesh(
        stage, particle_system_path, "/World/ClothMesh"
    )

    arm_position = np.array([0.75, 0.25, 0.5])
    arm_orientation = euler_angles_to_quat(
        np.array([0.0, 0.0, -90.0]), degrees=True
    )

    so_arm101_usd_path = "C:/Users/702A/Desktop/SO-ARM101-USD.usd"
    add_reference_to_stage(
        usd_path=so_arm101_usd_path, prim_path="/World/SO_ARM101"
    )

    arm_prim = stage.GetPrimAtPath("/World/SO_ARM101")
    if arm_prim.IsValid() and not arm_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(arm_prim)

    arm_robot = Robot(
        prim_path="/World/SO_ARM101",
        name="so_arm101",
        position=arm_position,
        orientation=arm_orientation,
    )
    world.scene.add(arm_robot)

    # 建立布料與手臂末端的 PhysX 固定連結
    attach_cloth_to_arm(stage, "/World/ClothMesh", "/World/SO_ARM101")

    # --------------------------------------------------------------------------
    # 6. 背景預先沉降 (Warm-up) 與 手臂姿勢控制配置
    # --------------------------------------------------------------------------
    print("[INFO] 4/5 執行背景預先沉降 (40 步)...")
    world.reset()

    default_joint_positions = arm_robot.get_joint_positions()
    if default_joint_positions is None or len(default_joint_positions) == 0:
        num_dof = arm_robot.num_dof
        default_joint_positions = np.zeros(num_dof) if num_dof > 0 else np.array([])

    arm_controller = arm_robot.get_articulation_controller()

    if arm_robot.num_dof > 0:
        arm_controller.set_gains(
            kps=np.full(arm_robot.num_dof, 1e4),
            kds=np.full(arm_robot.num_dof, 1e2),
        )

    for _ in range(40):
        if len(default_joint_positions) > 0:
            arm_controller.apply_action(
                ArticulationAction(joint_positions=default_joint_positions)
            )
        world.step(render=False)

    print("[INFO] 5/5 模擬啟動！已進入 10 秒等待倒數階段...")

    # --------------------------------------------------------------------------
    # 7. 定義運動姿勢與時間軌跡
    # --------------------------------------------------------------------------
    dip_joint_positions = np.copy(default_joint_positions)
    if len(dip_joint_positions) >= 3:
        dip_joint_positions[1] += 0.45  # 肩/臂關節向下前伸
        dip_joint_positions[2] += 0.35  # 肘關節向下浸入水槽

    # 吸水動態參數
    dry_mass = 0.03
    max_wet_mass = 0.35
    current_mass = dry_mass
    water_surface_z = 0.45

    dt = world.get_physics_dt()
    sim_time = 0.0

    # 時間節點 (單位: 秒)
    WAIT_TIME = 10.0      # 先等待 10 秒
    DIP_DURATION = 4.0   # 下潛進入水槽歷時 4 秒
    SOAK_DURATION = 4.0  # 水槽內停留浸泡 4 秒
    LIFT_DURATION = 4.0  # 提起離水歷時 4 秒

    t_dip_start = WAIT_TIME
    t_soak_start = t_dip_start + DIP_DURATION
    t_lift_start = t_soak_start + SOAK_DURATION
    t_end = t_lift_start + LIFT_DURATION

    # 主模擬迴圈
    while simulation_app.is_running():
        sim_time += dt

        # --- 狀態機控制關節軌跡 ---
        if len(default_joint_positions) > 0:
            if sim_time < t_dip_start:
                # 階段 1: [0s ~ 10s] 等待階段
                target_q = default_joint_positions
                if int(sim_time * 10) % 20 == 0:
                    remaining = max(0.0, t_dip_start - sim_time)
                    print(f"[STATUS] 靜置等待中... 剩餘倒數: {remaining:.1f}s", end="\r")

            elif t_dip_start <= sim_time < t_soak_start:
                # 階段 2: [10s ~ 14s] 下潛將布料帶入水槽
                alpha = (sim_time - t_dip_start) / DIP_DURATION
                smooth_alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
                target_q = (1.0 - smooth_alpha) * default_joint_positions + smooth_alpha * dip_joint_positions
                print(f"[STATUS] 動作執行中: 下潛進入水槽 ({alpha*100:.0f}%)    ", end="\r")

            elif t_soak_start <= sim_time < t_lift_start:
                # 階段 3: [14s ~ 18s] 於水槽中浸泡吸水
                target_q = dip_joint_positions
                print(f"[STATUS] 動作執行中: 水槽內浸泡吸水...         ", end="\r")

            elif t_lift_start <= sim_time < t_end:
                # 階段 4: [18s ~ 22s] 將布料拉起帶離水槽
                alpha = (sim_time - t_lift_start) / LIFT_DURATION
                smooth_alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
                target_q = (1.0 - smooth_alpha) * dip_joint_positions + smooth_alpha * default_joint_positions
                print(f"[STATUS] 動作執行中: 拿拉起離開水槽 ({alpha*100:.0f}%)    ", end="\r")

            else:
                # 階段 5: [> 22s] 拿出來後懸空保持
                target_q = default_joint_positions

            arm_controller.apply_action(ArticulationAction(joint_positions=target_q))

        world.step(render=True)

        # --- 吸水物理與視覺顏色即時更新 ---
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
        print(f"\n[ERROR] 執行異常: {e}")
    finally:
        simulation_app.close()
