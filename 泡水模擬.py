import omni.usd
import omni.kit.app
import omni.timeline
import carb.settings
import asyncio
import math
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, Vt
import numpy as np

# ==============================================================================
# 0. 全域配置與物理常數
# ==============================================================================
CONFIG = {
    "usd_stage_path": "C:/isaacsim/dive/arm_and_plane.usd",
    "cloth_parent_path": "/World/RoboticCloth10x",
    "cloth_mesh_path": "/World/RoboticCloth10x/mesh",
    "water_root_path": "/World/WaterEnvironment",
    
    # 水槽中心正對手臂正前方 (Y=0.0)
    "tank_center": Gf.Vec3d(0.25, 0.0, 0.06),
    "tank_size": (0.35, 0.35, 0.12),
    "wall_thickness": 0.02,
    "water_surface_z": 0.110,          # 水面高度 (0.11m)
    "water_bottom_z": 0.015,
    
    # 達西吸水動力學參數
    "absorption_rate_k": 1.5,          # 吸水速率常數 (1/s)
    "s_max": 2.5,                      # 最大飽和度 s_max (kg/m^2)
    "cloth_area": 0.16,                # 布料表面積 (m^2)
    
    # 物理屬性變化區間
    "dry_mass": 0.1,                   # 乾燥初始質量 (kg)
    "dry_damping": 12.0,               # 提升初始阻尼以抑制晃動
    "max_wet_damping": 60.0,           # 吸飽水時最大線性阻尼
}

# ==============================================================================
# 1. 建立純視覺水槽與水體 (無碰撞，杜絕 Overflow)
# ==============================================================================
def setup_clean_water_stage():
    settings = carb.settings.get_settings()
    settings.set("/physics/gpuMaxDeformableSurfaceContacts", 2097152)
    settings.set("/physics/maxDeformableSurfaceContacts", 2097152)
    settings.set("/physics/gpuCollisionStackSize", 134217728)

    usd_context = omni.usd.get_context()
    print(f"[INFO] Opening stage: {CONFIG['usd_stage_path']} ...")
    usd_context.open_stage(CONFIG['usd_stage_path'])
    stage = usd_context.get_stage()
    
    if not stage:
        raise RuntimeError(f"Failed to open stage: {CONFIG['usd_stage_path']}")

    # 配置 PhysicsScene
    scene_path = "/World/PhysicsScene"
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_prim = scene.GetPrim()

    PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, True).Set(True)
    scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token, True).Set("GPU")
    scene_prim.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts", Sdf.ValueTypeNames.Int, True).Set(2097152)

    # 建立水環境根節點
    water_root = CONFIG["water_root_path"]
    if not stage.GetPrimAtPath(water_root).IsValid():
        UsdGeom.Xform.Define(stage, water_root)

    # 建立水槽視覺外殼
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
        cube.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.55)]))

    w, l, h = CONFIG["tank_size"]
    t = CONFIG["wall_thickness"]
    tc = CONFIG["tank_center"]
    create_visual_wall("Bottom",     tc + Gf.Vec3d(0, 0, -h/2), Gf.Vec3f(w + 2*t, l + 2*t, t))
    create_wall_left = create_visual_wall("Wall_Left",   tc + Gf.Vec3d(-w/2 - t/2, 0, 0), Gf.Vec3f(t, l + 2*t, h))
    create_wall_right = create_visual_wall("Wall_Right",  tc + Gf.Vec3d(w/2 + t/2, 0, 0),  Gf.Vec3f(t, l + 2*t, h))
    create_wall_front = create_visual_wall("Wall_Front",  tc + Gf.Vec3d(0, l/2 + t/2, 0),  Gf.Vec3f(w, t, h))
    create_wall_back = create_visual_wall("Wall_Back",   tc + Gf.Vec3d(0, -l/2 - t/2, 0), Gf.Vec3f(w, t, h))

    # 建立連續半透明水體
    water_mesh_path = f"{water_root}/WaterVolume"
    water_cube = UsdGeom.Cube.Define(stage, water_mesh_path)
    water_cube.CreateSizeAttr().Set(1.0)
    water_prim = water_cube.GetPrim()
    
    water_vol_h = CONFIG["water_surface_z"] - CONFIG["water_bottom_z"]
    water_vol_center = Gf.Vec3d(tc[0], tc[1], CONFIG["water_bottom_z"] + water_vol_h / 2.0)
    
    xform = UsdGeom.XformCommonAPI(water_prim)
    xform.SetTranslate(water_vol_center)
    xform.SetScale(Gf.Vec3f(w * 0.98, l * 0.98, water_vol_h))
    
    water_cube.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.12, 0.55, 0.95)]))
    water_cube.CreateDisplayOpacityAttr().Set(Vt.FloatArray([0.5]))
    
    print(f"[SUCCESS] Clean Tank ready at X={tc[0]}, Y={tc[1]}. Water surface at Z = {CONFIG['water_surface_z']} m")

# ==============================================================================
# 2. 論文吸水模型解算器
# ==============================================================================
class ContinuousFluidAbsorber:
    def __init__(self, config):
        self.cfg = config
        self.stage = omni.usd.get_context().get_stage()
        self.cloth_parent = self.stage.GetPrimAtPath(self.cfg["cloth_parent_path"])
        self.cloth_mesh = self.stage.GetPrimAtPath(self.cfg["cloth_mesh_path"])
        
        self.absorbed_mass = 0.0
        self.current_saturation = 0.0
        self.max_water_mass = self.cfg["s_max"] * self.cfg["cloth_area"]

    def reset(self):
        self.absorbed_mass = 0.0
        self.current_saturation = 0.0
        self._apply_physics(self.cfg["dry_mass"], self.cfg["dry_damping"], wet_ratio=0.0)
        print("[INFO] Cloth reset to dry state.")

    def _apply_physics(self, mass, damping, wet_ratio):
        if self.cloth_parent.IsValid():
            mass_attr = self.cloth_parent.GetAttribute("omniphysics:mass")
            damping_attr = self.cloth_parent.GetAttribute("physxDeformableBody:linearDamping")
            if mass_attr.IsValid():
                mass_attr.Set(float(mass))
            if damping_attr.IsValid():
                damping_attr.Set(float(damping))
                
        # 依含水量動態暗化
        if self.cloth_mesh.IsValid():
            color_attr = self.cloth_mesh.GetAttribute("primvars:displayColor")
            if color_attr.IsValid():
                shade = 0.85 * (1.0 - 0.45 * wet_ratio)
                color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(shade, shade * 0.92, shade * 0.82)]))

    def update_step(self, dt):
        if not self.cloth_mesh.IsValid():
            return 0.0, False

        bbox_min, bbox_max = omni.usd.get_context().compute_path_world_bounding_box(self.cfg["cloth_mesh_path"])
        cloth_z_min = bbox_min[2]
        cloth_z_max = bbox_max[2]
        water_surface = self.cfg["water_surface_z"]
        
        is_submerged = cloth_z_min < water_surface
        
        if is_submerged and self.absorbed_mass < self.max_water_mass:
            cloth_height = max(0.01, cloth_z_max - cloth_z_min)
            submerged_depth = max(0.0, min(water_surface - cloth_z_min, cloth_height))
            immersion_ratio = np.clip(submerged_depth / cloth_height, 0.0, 1.0)
            
            remaining_capacity = self.max_water_mass - self.absorbed_mass
            dm = self.cfg["absorption_rate_k"] * immersion_ratio * remaining_capacity * dt
            
            self.absorbed_mass += dm
            self.current_saturation = self.absorbed_mass / self.cfg["cloth_area"]
            
            wet_ratio = min(1.0, self.current_saturation / self.cfg["s_max"])
            new_mass = self.cfg["dry_mass"] + self.absorbed_mass
            new_damping = self.cfg["dry_damping"] + wet_ratio * (self.cfg["max_wet_damping"] - self.cfg["dry_damping"])
            
            self._apply_physics(new_mass, new_damping, wet_ratio)
            
        return cloth_z_min, is_submerged

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
    pitch_prim = joint_map.get("Pitch") or joint_map.get("pitch") or joint_map.get("Shoulder") or joint_map.get("shoulder")
    elbow_prim = joint_map.get("Elbow") or joint_map.get("elbow")
    wrist_prim = joint_map.get("Wrist") or joint_map.get("wrist") or joint_map.get("Wrist_Pitch")

    # S 形平滑加減速移動 (消除頓挫與布料擺盪)
    async def smooth_move_s_curve(prim, target_angle, duration=2.0, steps=50):
        if prim is None or not prim.IsValid():
            return
        attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not attr.IsValid():
            return
        start_angle = attr.Get() or 0.0
        step_dt = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            # Cosine Ease-in-Ease-out 插值
            s = (1.0 - math.cos(t * math.pi)) * 0.5
            attr.Set(start_angle + (target_angle - start_angle) * s)
            await asyncio.sleep(step_dt)

    # 確保底座固定朝正前方 (0.0 度)，完全不左右搖擺
    if rot_prim and rot_prim.IsValid():
        rot_attr = rot_prim.GetAttribute("drive:angular:physics:targetPosition")
        if rot_attr.IsValid():
            rot_attr.Set(0.0)

    # 1. 直接平穩下探浸水 (Pure Downward Dip)
    print("[ARM] 1. Moving straight DOWN into water...")
    await asyncio.gather(
        smooth_move_s_curve(pitch_prim, 52.0, duration=2.5),  # 肩膀前傾下探
        smooth_move_s_curve(elbow_prim, 32.0, duration=2.5),  # 手肘協同下沉
        smooth_move_s_curve(wrist_prim, 10.0, duration=2.0)
    )

    # 2. 靜止浸泡吸水 4 秒
    print("[ARM] 2. Submerged in water: absorbing liquid smoothly...")
    await asyncio.sleep(4.0)

    # 3. 直接平穩直直抬起 (Pure Upward Lift)
    print("[ARM] 3. Lifting heavy wet cloth straight UP...")
    await asyncio.gather(
        smooth_move_s_curve(pitch_prim, 0.0, duration=2.8),
        smooth_move_s_curve(elbow_prim, 0.0, duration=2.8),
        smooth_move_s_curve(wrist_prim, 0.0, duration=2.0)
    )
    print("[ARM] Direct vertical dipping routine completed!")

# ==============================================================================
# 4. 主模擬迴圈註冊
# ==============================================================================
setup_clean_water_stage()
absorber = ContinuousFluidAbsorber(CONFIG)
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
        print("[INFO] Simulation playing. Pure vertical dip-and-lift routine started.")

    cloth_z, is_submerged = absorber.update_step(dt)
    
    if sim_frame % 30 == 0:
        water_z = CONFIG["water_surface_z"]
        status_tag = ">>> [INSIDE WATER - ABSORBING] <<<" if is_submerged else f"ABOVE WATER (Gap: {(cloth_z - water_z)*100:.1f} cm)"
        total_m = CONFIG["dry_mass"] + absorber.absorbed_mass
        print(f"[F{sim_frame:04d}] Cloth Z: {cloth_z:.3f}m | Water: {water_z:.3f}m | {status_tag} | Mass: {total_m:.3f}kg")

app = omni.kit.app.get_app()
if '_clean_fluid_absorber_sub' in globals() and globals()['_clean_fluid_absorber_sub'] is not None:
    globals()['_clean_fluid_absorber_sub'] = None

globals()['_clean_fluid_absorber_sub'] = app.get_update_event_stream().create_subscription_to_pop(on_render_physics_step)
print("\n[READY] Straight-down dip-and-lift script loaded! Press PLAY to test.")
