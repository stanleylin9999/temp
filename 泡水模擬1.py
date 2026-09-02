import asyncio
import math
import carb.settings
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# ==============================================================================
# 0. 全域配置與微觀物理參數 (Fei et al. 2018 論文基準組 + 493% 吸水率校準)
# ==============================================================================
CONFIG = {
    "usd_stage_path": "C:/isaacsim/dive/arm_and_plane.usd",
    "cloth_parent_path": "/World/RoboticCloth10x",
    "cloth_mesh_path": "/World/RoboticCloth10x/mesh",
    "water_root_path": "/World/WaterEnvironment",
    # 水槽幾何配置 (正對手臂正前方 Y=0.0)
    "tank_center": Gf.Vec3d(0.25, 0.0, 0.06),
    "tank_size": (0.35, 0.35, 0.12),
    "wall_thickness": 0.02,
    "water_surface_z": 0.110,  # 水面高度 (0.11 m)
    "water_bottom_z": 0.015,
    # 論文微觀多孔織物參數 (Table 1 & Fig. 10 基準組)
    "phi": 0.40,  # 織物固體積分率
    "cloth_thickness": 0.0012,  # 布料厚度 (m)
    "fiber_diameter_d": 100e-6,  # 纖維直徑 100 um
    "capillary_radius_rb": 61e-6,  # 微觀毛細孔隙半徑 61 um
    "contact_angle_deg": 40.8,  # 水-纖維接觸角
    "surface_tension_gamma": 0.0728,  # 水表面張力 (N/m)
    "water_viscosity_mu": 0.001,  # 水動力黏度 (Pa*s)
    "absorption_rate_kin": 3.0,  # 水面接觸充水常數 (1/s)
    # 巨觀吸水量與物性校準
    "dry_mass": 0.10,  # 乾燥初始質量 (kg)
    "water_absorption_ratio": 4.93,  # 飽和吸水率 493% (即吸水量為乾重的 4.93 倍)
    "dry_damping": 12.0,  # 初始線性阻尼
    "max_wet_damping": 65.0,  # 飽和吸水時最大線性阻尼
}


# ==============================================================================
# 1. 建立純視覺水槽與水體 (穩定無溢位配置: 2,097,152 接觸緩衝)
# ==============================================================================
def setup_clean_water_stage():
    settings = carb.settings.get_settings()
    settings.set("/physics/gpuMaxDeformableSurfaceContacts", 2097152)
    settings.set("/physics/maxDeformableSurfaceContacts", 2097152)
    settings.set("/physics/gpuCollisionStackSize", 134217728)

    usd_context = omni.usd.get_context()
    current_stage = usd_context.get_stage()
    current_url = (
        current_stage.GetRootLayer().identifier if current_stage else ""
    )
    target_path = CONFIG["usd_stage_path"].replace("\\", "/").lower()

    if target_path not in current_url.replace("\\", "/").lower():
        print(f"[INFO] Opening target stage: {CONFIG['usd_stage_path']} ...")
        usd_context.open_stage(CONFIG["usd_stage_path"])
        stage = usd_context.get_stage()
    else:
        stage = current_stage

    if not stage:
        raise RuntimeError(f"Failed to open stage: {CONFIG['usd_stage_path']}")

    # 配置 PhysicsScene GPU 動力學
    scene_path = "/World/PhysicsScene"
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_prim = scene.GetPrim()

    PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    scene_prim.CreateAttribute(
        "physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, True
    ).Set(True)
    scene_prim.CreateAttribute(
        "physxScene:broadphaseType", Sdf.ValueTypeNames.Token, True
    ).Set("GPU")
    scene_prim.CreateAttribute(
        "physxScene:gpuMaxDeformableSurfaceContacts",
        Sdf.ValueTypeNames.Int,
        True,
    ).Set(2097152)

    # 建立水環境根節點
    water_root = CONFIG["water_root_path"]
    if not stage.GetPrimAtPath(water_root).IsValid():
        UsdGeom.Xform.Define(stage, water_root)

    tank_root = f"{water_root}/Tank"
    if not stage.GetPrimAtPath(tank_root).IsValid():
        UsdGeom.Xform.Define(stage, tank_root)

    def create_visual_wall(name, pos, scale):
        cube = UsdGeom.Cube.Define(stage, f"{tank_root}/{name}")
        cube.CreateSizeAttr().Set(1.0)
        prim = cube.GetPrim()
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(pos)
        xform.SetScale(scale)
        cube.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.55)])
        )

    w, l, h = CONFIG["tank_size"]
    t = CONFIG["wall_thickness"]
    tc = CONFIG["tank_center"]
    create_visual_wall(
        "Bottom", tc + Gf.Vec3d(0, 0, -h / 2), Gf.Vec3f(w + 2 * t, l + 2 * t, t)
    )
    create_visual_wall(
        "Wall_Left",
        tc + Gf.Vec3d(-w / 2 - t / 2, 0, 0),
        Gf.Vec3f(t, l + 2 * t, h),
    )
    create_visual_wall(
        "Wall_Right",
        tc + Gf.Vec3d(w / 2 + t / 2, 0, 0),
        Gf.Vec3f(t, l + 2 * t, h),
    )
    create_visual_wall(
        "Wall_Front",
        tc + Gf.Vec3d(0, l / 2 + t / 2, 0),
        Gf.Vec3f(w, t, h),
    )
    create_visual_wall(
        "Wall_Back",
        tc + Gf.Vec3d(0, -l / 2 - t / 2, 0),
        Gf.Vec3f(w, t, h),
    )

    # 建立純視覺半透明水體 (無物理碰撞 API)
    water_mesh_path = f"{water_root}/WaterVolume"
    water_cube = UsdGeom.Cube.Define(stage, water_mesh_path)
    water_cube.CreateSizeAttr().Set(1.0)
    water_prim = water_cube.GetPrim()

    water_vol_h = CONFIG["water_surface_z"] - CONFIG["water_bottom_z"]
    water_vol_center = Gf.Vec3d(
        tc[0], tc[1], CONFIG["water_bottom_z"] + water_vol_h / 2.0
    )

    xform = UsdGeom.XformCommonAPI(water_prim)
    xform.SetTranslate(water_vol_center)
    xform.SetScale(Gf.Vec3f(w * 0.98, l * 0.98, water_vol_h))
    water_cube.CreateDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.12, 0.55, 0.95)])
    )
    water_cube.CreateDisplayOpacityAttr().Set(Vt.FloatArray([0.5]))

    print(
        f"[SUCCESS] Tank ready at X={tc[0]}, Y={tc[1]}. Water surface at Z = {CONFIG['water_surface_z']} m"
    )
    return stage


# ==============================================================================
# 2. 論文多孔吸水解算器 (已校準頂點對偶體積與 493% 吸水量)
# ==============================================================================
def resolve_cloth_mesh_prim(stage, config):
    prim = stage.GetPrimAtPath(config["cloth_mesh_path"])
    if prim.IsValid() and prim.IsA(UsdGeom.Mesh):
        return prim

    parent_prim = stage.GetPrimAtPath(config["cloth_parent_path"])
    if parent_prim.IsValid():
        if parent_prim.IsA(UsdGeom.Mesh):
            return parent_prim
        for child in parent_prim.GetChildren():
            if child.IsA(UsdGeom.Mesh):
                return child

    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh) and "cloth" in p.GetPath().pathString.lower():
            return p

    raise RuntimeError("Cannot resolve cloth UsdGeom.Mesh in current stage.")


class ExactMeshAbsorberValidation:

    def __init__(self, config, stage):
        self.cfg = config
        self.stage = stage
        self.cloth_parent = self.stage.GetPrimAtPath(
            self.cfg["cloth_parent_path"]
        )
        self.cloth_prim = resolve_cloth_mesh_prim(self.stage, self.cfg)
        self.cloth_mesh = UsdGeom.Mesh(self.cloth_prim)
        self.xformable = UsdGeom.Xformable(self.cloth_prim)

        # 讀取初始頂點與拓撲
        points = self.cloth_mesh.GetPointsAttr().Get()
        self.num_vertices = len(points) if points else 0
        face_indices = self.cloth_mesh.GetFaceVertexIndicesAttr().Get()
        face_counts = self.cloth_mesh.GetFaceVertexCountsAttr().Get()

        # 構建頂點鄰接圖與幾何對偶體積 (Dual Cell Volumes)[cite: 1]
        self.neighbors = [set() for _ in range(self.num_vertices)]
        self.vertex_volumes = np.zeros(self.num_vertices, dtype=np.float64)

        thickness = self.cfg["cloth_thickness"]
        idx = 0
        if face_counts and face_indices:
            for count in face_counts:
                if count >= 3:
                    v0 = face_indices[idx]
                    for k in range(1, count - 1):
                        v1 = face_indices[idx + k]
                        v2 = face_indices[idx + k + 1]

                        self.neighbors[v0].add(v1)
                        self.neighbors[v1].add(v0)
                        self.neighbors[v1].add(v2)
                        self.neighbors[v2].add(v1)
                        self.neighbors[v2].add(v0)
                        self.neighbors[v0].add(v2)

                        # 正確公尺尺度面積計算 (不縮小 1e-4)
                        p0 = np.array(points[v0])
                        p1 = np.array(points[v1])
                        p2 = np.array(points[v2])
                        tri_area = 0.5 * np.linalg.norm(
                            np.cross(p1 - p0, p2 - p0)
                        )
                        tri_volume = tri_area * thickness

                        # 論文重心對偶體積分量分配 (1/3 per vertex)[cite: 1]
                        self.vertex_volumes[v0] += tri_volume / 3.0
                        self.vertex_volumes[v1] += tri_volume / 3.0
                        self.vertex_volumes[v2] += tri_volume / 3.0
                idx += count

        self.neighbors_list = [list(n) for n in self.neighbors]

        # 計算各頂點在布料中的幾何體積權重
        total_mesh_volume = np.sum(self.vertex_volumes)
        if total_mesh_volume > 0:
            self.vertex_weights = self.vertex_volumes / total_mesh_volume
        else:
            self.vertex_weights = np.ones(self.num_vertices) / max(
                1, self.num_vertices
            )

        # 依據實驗要求的 493% 吸水率嚴格校準飽和含水量
        self.max_water_mass = (
            self.cfg["dry_mass"] * self.cfg["water_absorption_ratio"]
        )  # 0.1kg * 4.93 = 0.493kg

        # 計算論文微觀毛細吸力 p_alpha 與滲透率 k_alpha[cite: 1]
        phi = self.cfg["phi"]
        d = self.cfg["fiber_diameter_d"]
        rb = self.cfg["capillary_radius_rb"]
        theta = np.radians(self.cfg["contact_angle_deg"])
        gamma = self.cfg["surface_tension_gamma"]
        mu = self.cfg["water_viscosity_mu"]

        self.p_alpha = (2.0 * phi * gamma * np.cos(theta)) / (
            (1.0 - phi) * rb
        )
        self.k_alpha = (
            (-np.log(phi) - 1.476 + 2.0 * phi - 0.562 * (phi**2))
            / (16.0 * phi)
        ) * (d**2)
        self.c_alpha = mu / max(1e-15, self.k_alpha)

        # 頂點飽和度陣列 S_r in [0, 1][cite: 1]
        self.saturation = np.zeros(self.num_vertices, dtype=np.float64)

        print(
            f"[INIT] Mesh Parsed: {self.num_vertices} Vertices | Connected to {self.cloth_prim.GetPath()}"
        )
        print(
            f"[INIT] Microstructure: p_alpha={self.p_alpha:.1f} Pa | k_alpha={self.k_alpha:.2e} m^2"
        )
        print(
            f"[INIT] Calibration: Dry Mass = {self.cfg['dry_mass']:.3f} kg | Target Wet Increase = +493.0% ({self.max_water_mass:.3f} kg water)"
        )

    def reset(self):
        self.saturation.fill(0.0)
        self._apply_physics(0.0)

    def _apply_physics(self, total_water_mass):
        curr_mass = self.cfg["dry_mass"] + total_water_mass
        avg_sat = np.mean(self.saturation) if self.num_vertices > 0 else 0.0
        curr_damping = self.cfg["dry_damping"] + avg_sat * (
            self.cfg["max_wet_damping"] - self.cfg["dry_damping"]
        )

        if self.cloth_parent.IsValid():
            mass_attr = self.cloth_parent.GetAttribute("omniphysics:mass")
            damping_attr = self.cloth_parent.GetAttribute(
                "physxDeformableBody:linearDamping"
            )
            if mass_attr.IsValid():
                mass_attr.Set(float(curr_mass))
            if damping_attr.IsValid():
                damping_attr.Set(float(curr_damping))

        if self.cloth_prim.IsValid():
            color_attr = self.cloth_mesh.GetDisplayColorAttr()
            if not color_attr.IsValid():
                color_attr = self.cloth_mesh.CreateDisplayColorAttr()
            if color_attr.IsValid():
                shade = 0.85 * (1.0 - 0.45 * avg_sat)
                color_attr.Set(
                    Vt.Vec3fArray(
                        [Gf.Vec3f(shade, shade * 0.92, shade * 0.82)]
                    )
                )

    def step_simulation(self, dt):
        if not self.cloth_prim.IsValid() or self.num_vertices == 0:
            return 0.0, 0.0, False

        local_pts = self.cloth_mesh.GetPointsAttr().Get()
        world_tf = self.xformable.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        z_vals = np.array(
            [world_tf.Transform(pt)[2] for pt in local_pts], dtype=np.float64
        )

        water_z = self.cfg["water_surface_z"]
        submerged_mask = z_vals < water_z
        is_submerged = np.any(submerged_mask)

        # [機制 1] 界面多孔介質捕捉充水 (Liquid Capturing)[cite: 1]
        kin = self.cfg["absorption_rate_kin"]
        self.saturation[submerged_mask] += (
            kin * (1.0 - self.saturation[submerged_mask]) * dt
        )

        # [機制 2] 網格表面 Richards 芯吸擴散 (Wicking Diffusion)[cite: 1]
        phi = self.cfg["phi"]
        d_sr = ((1.0 - phi) * self.p_alpha * self.saturation) / self.c_alpha

        diff_flux = np.zeros(self.num_vertices, dtype=np.float64)
        for i in range(self.num_vertices):
            nbrs = self.neighbors_list[i]
            if nbrs:
                d_edge = 0.5 * (d_sr[i] + d_sr[nbrs])
                diff_flux[i] = np.sum(
                    d_edge * (self.saturation[nbrs] - self.saturation[i])
                )

        # 數值 CFL 限幅保護
        scaled_diff = np.clip(diff_flux * dt, -0.05, 0.05)
        self.saturation += scaled_diff
        self.saturation = np.clip(self.saturation, 0.0, 1.0)

        # [機制 3] 精確積分計算吸水質量 (符合 +493% 物理校準)[cite: 1]
        effective_saturation = np.sum(self.saturation * self.vertex_weights)
        total_absorbed_mass = self.max_water_mass * effective_saturation

        self._apply_physics(total_absorbed_mass)

        return float(np.min(z_vals)), total_absorbed_mass, is_submerged


# ==============================================================================
# 3. 機械臂「無擺動垂直深浸」軌跡控制 (Pure Vertical Dip & Lift)
# ==============================================================================
async def execute_arm_dipping_sequence():
    await asyncio.sleep(1.0)
    stage = omni.usd.get_context().get_stage()

    joints_root = stage.GetPrimAtPath("/so101_new_calib/joints")
    if not joints_root.IsValid():
        print("[ERROR] Cannot find /so101_new_calib/joints")
        return

    children = joints_root.GetChildren()
    joint_map = {child.GetName(): child for child in children}

    rot_prim = joint_map.get("Rotation") or joint_map.get("rotation")
    pitch_prim = (
        joint_map.get("Pitch")
        or joint_map.get("pitch")
        or joint_map.get("Shoulder")
        or joint_map.get("shoulder")
    )
    elbow_prim = joint_map.get("Elbow") or joint_map.get("elbow")
    wrist_prim = (
        joint_map.get("Wrist")
        or joint_map.get("wrist")
        or joint_map.get("Wrist_Pitch")
    )

    async def smooth_move_s_curve(prim, target_angle, duration=2.5, steps=50):
        if prim is None or not prim.IsValid():
            return
        attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not attr.IsValid():
            return
        start_angle = attr.Get() or 0.0
        step_dt = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            s = (1.0 - math.cos(t * math.pi)) * 0.5
            attr.Set(start_angle + (target_angle - start_angle) * s)
            await asyncio.sleep(step_dt)

    if rot_prim and rot_prim.IsValid():
        rot_attr = rot_prim.GetAttribute("drive:angular:physics:targetPosition")
        if rot_attr.IsValid():
            rot_attr.Set(0.0)

    # 1. 垂直下探浸水
    print("[ARM] 1. Moving straight DOWN into water...")
    await asyncio.gather(
        smooth_move_s_curve(pitch_prim, 52.0, duration=2.5),
        smooth_move_s_curve(elbow_prim, 32.0, duration=2.5),
        smooth_move_s_curve(wrist_prim, 10.0, duration=2.0),
    )

    # 2. 水中靜止浸泡 4 秒 (觀察網格吸水與毛細爬升)
    print(
        "[ARM] 2. Submerged in water: absorbing liquid & wicking towards 493% increase..."
    )
    await asyncio.sleep(4.0)

    # 3. 垂直抬起離水
    print("[ARM] 3. Lifting heavy wet cloth straight UP...")
    await asyncio.gather(
        smooth_move_s_curve(pitch_prim, 0.0, duration=2.8),
        smooth_move_s_curve(elbow_prim, 0.0, duration=2.8),
        smooth_move_s_curve(wrist_prim, 0.0, duration=2.0),
    )
    print("[ARM] Direct vertical dipping routine completed!")


# ==============================================================================
# 4. 主模擬迴圈掛載與吸水量數據輸出
# ==============================================================================
stage = setup_clean_water_stage()
absorber = ExactMeshAbsorberValidation(CONFIG, stage)
timeline = omni.timeline.get_timeline_interface()
sim_frame = 0


def on_render_physics_step(e):
    global sim_frame
    if not timeline.is_playing():
        if sim_frame > 0:
            absorber.reset()
            sim_frame = 0
        return

    sim_frame += 1
    dt = 1.0 / 60.0

    if sim_frame == 1:
        absorber.reset()
        asyncio.ensure_future(execute_arm_dipping_sequence())
        print(
            "[INFO] Simulation started. 493% calibrated mesh porous solver active."
        )

    cloth_z, water_mass, is_submerged = absorber.step_simulation(dt)

    if sim_frame % 30 == 0:
        water_z = CONFIG["water_surface_z"]
        status_tag = (
            ">>> [IN WATER - ABSORBING] <<<"
            if is_submerged
            else f"ABOVE WATER (Gap: {(cloth_z - water_z)*100:.1f} cm)"
        )
        total_m = CONFIG["dry_mass"] + water_mass
        avg_sat = np.mean(absorber.saturation) * 100.0
        weight_increase_pct = (water_mass / CONFIG["dry_mass"]) * 100.0
        print(
            f"[F{sim_frame:04d}] Lowest Z: {cloth_z:.3f}m | Water Z: {water_z:.3f}m | "
            f"{status_tag} | Water: {water_mass * 1000:6.1f}g (+{weight_increase_pct:5.1f}%) | "
            f"Sat: {avg_sat:4.1f}% | Mass: {total_m:.3f}kg"
        )


app = omni.kit.app.get_app()
if (
    "_exact_mesh_absorber_sub" in globals()
    and globals()["_exact_mesh_absorber_sub"] is not None
):
    globals()["_exact_mesh_absorber_sub"] = None

globals()["_exact_mesh_absorber_sub"] = (
    app.get_update_event_stream().create_subscription_to_pop(
        on_render_physics_step
    )
)
print(
    "\n[READY] 493% Water Absorption Calibration loaded successfully! Press PLAY."
)
