# -----------------------------------------------------------------------------
# 1. 必須在導入任何其他 omni/isaac 模組「之前」初始化 SimulationApp
# -----------------------------------------------------------------------------
from isaacsim import SimulationApp

# 配置啟動參數（headless=False 代表顯示 GUI，改為 True 則為背景無介面執行）
CONFIG = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)

# -----------------------------------------------------------------------------
# 2. 啟動 SimulationApp 後才能導入 Isaac Sim / Omniverse 相關 API
# -----------------------------------------------------------------------------
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, GroundPlane

# 3. 建立物理世界 (World)
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 4. 在場景中加入物件
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="my_cube",
        position=np.array([0.0, 0.0, 2.0]), # XYZ 座標 (米)
        scale=np.array([0.5, 0.5, 0.5]),     # 尺寸
        color=np.array([0.1, 0.6, 0.9]),     # RGB 顏色 (0~1)
    )
)

# 5. 重置世界 (初始化 Physics Handles 與 USD 狀態)
world.reset()

# 6. 主模擬迴圈
print("[INFO] 開始物理模擬...")
for i in range(500):
    # 推進物理模擬一步 (頻率預設為 60Hz)
    world.step(render=True)
    
    # 每 50 步列印一次方塊位置
    if i % 50 == 0:
        position, orientation = cube.get_world_pose()
        print(f"Step {i:03d} | Cube Position Z: {position[2]:.3f} m")

# 7. 結束時關閉應用程式
simulation_app.close()
