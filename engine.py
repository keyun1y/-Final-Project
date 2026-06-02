import sqlite3
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, node_id: int, x: float, y: float, rx: int, ry: int, rmz: int):
        self.node_id = node_id
        self.x, self.y = x, y
        self.restraints = [bool(rx), bool(ry), bool(rmz)]
        self.dof_indices = []

    @property
    def is_fixed(self):
        return all(self.restraints)

class Material:
    def __init__(self, mat_id: int, name: str, E: float, I: float, A: float):
        self.mat_id, self.name, self.E, self.I, self.A = mat_id, name, E, I, A

class Element:
    def __init__(self, element_id: int, node_i: Node, node_j: Node, material: Material):
        self.element_id, self.node_i, self.node_j, self.material = element_id, node_i, node_j, material

    @property
    def length(self) -> float:
        return np.hypot(self.node_j.x - self.node_i.x, self.node_j.y - self.node_i.y)

    @property
    def angle(self) -> float:
        return np.arctan2(self.node_j.y - self.node_i.y, self.node_j.x - self.node_i.x)

    def get_transformation_matrix(self) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([
            [ c,  s,  0,  0,  0,  0], [-s,  c,  0,  0,  0,  0], [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0], [ 0,  0,  0, -s,  c,  0], [ 0,  0,  0,  0,  0,  1]
        ], dtype=float)

    def compute_internal_forces(self, U_global: np.ndarray, element_loads: list = None) -> np.ndarray:
        dofs = self.node_i.dof_indices + self.node_j.dof_indices
        u_global_el = np.array([U_global[i] for i in dofs])
        T, k_local = self.get_transformation_matrix(), self.get_local_stiffness()
        f_disp = k_local @ T @ u_global_el
        f_fef = np.zeros(len(dofs), dtype=float)
        if element_loads:
            for load in element_loads:
                if load.element.element_id == self.element_id:
                    f_fef = -load.get_equivalent_nodal_forces()
        return f_disp + f_fef

class FrameElement(Element):
    def get_local_stiffness(self) -> np.ndarray:
        E, I, A, L = self.material.E, self.material.I, self.material.A, self.length
        k = np.zeros((6, 6))
        k[0,0] = k[3,3] = A*E/L; k[0,3] = k[3,0] = -A*E/L
        k[1,1] = k[4,4] = 12*E*I/L**3; k[1,4] = k[4,1] = -12*E*I/L**3
        k[1,2] = k[2,1] = k[1,5] = k[5,1] = 6*E*I/L**2; k[4,2] = k[2,4] = k[4,5] = k[5,4] = -6*E*I/L**2
        k[2,2] = k[5,5] = 4*E*I/L; k[2,5] = k[5,2] = 2*E*I/L
        return k

class NodalLoad:
    def __init__(self, node, Fx=0.0, Fy=0.0, Mz=0.0):
        self.node, self.Fx, self.Fy, self.Mz = node, Fx, Fy, Mz

class ElementUniformLoad:
    def __init__(self, element, w: float):
        self.element, self.w = element, w 
        
    def get_equivalent_nodal_forces(self) -> np.ndarray:
        L, w = self.element.length, self.w
        return np.array([0.0, w*L/2.0, w*(L**2)/12.0, 0.0, w*L/2.0, -w*(L**2)/12.0], dtype=float)

class ModelDatabaseLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nodes, self.materials, self.elements = {}, {}, []
        
    def load_model(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT node_id, x_coord, y_coord, rx, ry, rmz FROM Nodes")
        for r in cursor.fetchall(): self.nodes[r[0]] = Node(*r)
        
        cursor.execute("SELECT id, name, E_value, I_value, A_value FROM Materials")
        for r in cursor.fetchall(): self.materials[r[0]] = Material(*r)
        
        cursor.execute("SELECT element_id, node_i, node_j, material_id FROM Elements")
        for r in cursor.fetchall():
            self.elements.append(FrameElement(r[0], self.nodes[r[1]], self.nodes[r[2]], self.materials[r[3]]))
        conn.close()
        return self.nodes, self.elements

    def load_loads(self):
        nodal_loads, element_loads = [], []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT target_type, target_id, fx, fy, mz FROM Loads")
        for r in cursor.fetchall():
            t_type, t_id, fx, fy, mz = r
            if t_type == 'NODE':
                nodal_loads.append(NodalLoad(self.nodes[t_id], fx, fy, mz))
            elif t_type == 'ELEMENT':
                el = next((e for e in self.elements if e.element_id == t_id), None)
                if el: element_loads.append(ElementUniformLoad(el, fy))
        conn.close()
        return nodal_loads, element_loads

class LinearStaticSolver:
    def __init__(self, nodes: dict, elements: list, dof_per_node: int = 3):
        self.nodes, self.elements, self.dof_per_node = nodes, elements, dof_per_node
        self.total_dofs = len(nodes) * dof_per_node
        self.K_global = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self.F_global = np.zeros(self.total_dofs, dtype=float)

    def solve(self, nodal_loads=[], element_loads=[]):
        curr = 0
        for nid in sorted(self.nodes.keys()):
            self.nodes[nid].dof_indices = [curr + i for i in range(self.dof_per_node)]
            curr += self.dof_per_node
            
        for el in self.elements:
            K_e = el.get_transformation_matrix().T @ el.get_local_stiffness() @ el.get_transformation_matrix()
            dofs = el.node_i.dof_indices + el.node_j.dof_indices
            for i in range(len(dofs)):
                for j in range(len(dofs)):
                    self.K_global[dofs[i], dofs[j]] += K_e[i, j]
                    
        for load in nodal_loads:
            d = load.node.dof_indices
            self.F_global[d[0]] += load.Fx; self.F_global[d[1]] += load.Fy; self.F_global[d[2]] += load.Mz
        for load in element_loads:
            enf_global = load.element.get_transformation_matrix().T @ load.get_equivalent_nodal_forces()
            d = load.element.node_i.dof_indices + load.element.node_j.dof_indices
            for i in range(len(d)): self.F_global[d[i]] += enf_global[i]

        for node in self.nodes.values():
            for local_dof, is_locked in enumerate(node.restraints):
                if is_locked:
                    global_dof = node.dof_indices[local_dof]
                    self.K_global[global_dof, :] = 0.0
                    self.K_global[:, global_dof] = 0.0
                    self.K_global[global_dof, global_dof] = 1.0
                    self.F_global[global_dof] = 0.0
                    
        self.U_global = np.linalg.solve(self.K_global, self.F_global)
        return self.U_global

class ResultVisualizer:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements

    def plot_model(self, nodal_loads=None, element_loads=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        for el in self.elements:
            ax.plot([el.node_i.x, el.node_j.x], [el.node_i.y, el.node_j.y], 'k-', linewidth=3)
            
        for node in self.nodes.values():
            rx, ry, rmz = node.restraints
            
            # 1. 固定端 (Fixed) [1, 1, 1]：紅色大正方形
            if rx == 1 and ry == 1 and rmz == 1:
                ax.plot(node.x, node.y, marker='s', color='red', markersize=10, zorder=5)
                
            # 2. 鉸支承 (Pinned) [1, 1, 0]：綠色正三角形
            elif rx == 1 and ry == 1 and rmz == 0:
                ax.plot(node.x, node.y, marker='^', color='forestgreen', markersize=12, zorder=5)
                
            # 3. 滾支承 (Roller) [0, 1, 0] 或 [1, 0, 0]：橘色圓圈 (代表滾輪)
            elif (rx == 0 and ry == 1 and rmz == 0) or (rx == 1 and ry == 0 and rmz == 0):
                ax.plot(node.x, node.y, marker='o', color='darkorange', markerfacecolor='white', markeredgewidth=2, markersize=10, zorder=5)
                
            # 4. 自由端 (Free) [0, 0, 0]：藍色小點 (只標示幾何位置，不干擾視覺)
            else:
                ax.plot(node.x, node.y, marker='.', color='royalblue', markersize=6, zorder=5)
            
        if nodal_loads: self._draw_nodal_loads(ax, nodal_loads)
        if element_loads: self._draw_element_loads(ax, element_loads)
            
        ax.set_aspect('equal')
        ax.set_title("V2 Structural Model & Loads", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.margins(0.3)

    def _draw_nodal_loads(self, ax, nodal_loads):
        for l in nodal_loads:
            if l.Fy != 0: 
                ax.annotate(f"{abs(l.Fy/1000):.1f}kN", xy=(l.node.x, l.node.y), xytext=(0, 30 if l.Fy<0 else -30), textcoords="offset points", ha='center', va='bottom' if l.Fy<0 else 'top', color='darkorange', fontweight='bold', arrowprops=dict(facecolor='darkorange', shrink=0.1, width=2, headwidth=8))
            if l.Fx != 0: 
                ax.annotate(f"{abs(l.Fx/1000):.1f}kN", xy=(l.node.x, l.node.y), xytext=(-40 if l.Fx>0 else 40, 0), textcoords="offset points", ha='right' if l.Fx>0 else 'left', va='center', color='darkorange', fontweight='bold', arrowprops=dict(facecolor='darkorange', shrink=0.1, width=2, headwidth=8))

    def _draw_element_loads(self, ax, element_loads):
        for l in element_loads:
            if l.w == 0: continue
            L, theta = l.element.length, l.element.angle
            h = L * 0.15
            visual_h = h if l.w < 0 else -h
            x_loc = np.linspace(0, L, 5)
            y_tail = np.full_like(x_loc, visual_h)
            y_head = np.zeros_like(x_loc)
            c, s = np.cos(theta), np.sin(theta)
            top_X = l.element.node_i.x + x_loc*c - y_tail*s
            top_Y = l.element.node_i.y + x_loc*s + y_tail*c
            for i in range(5):
                Xt = l.element.node_i.x + x_loc[i]*c - y_tail[i]*s
                Yt = l.element.node_i.y + x_loc[i]*s + y_tail[i]*c
                Xh = l.element.node_i.x + x_loc[i]*c - y_head[i]*s
                Yh = l.element.node_i.y + x_loc[i]*s + y_head[i]*c
                ax.arrow(Xt, Yt, (Xh-Xt)*0.9, (Yh-Yt)*0.9, color='purple', width=L*0.005, head_width=L*0.03, length_includes_head=True, alpha=0.7)
            ax.plot(top_X, top_Y, color='purple', linewidth=2, alpha=0.7)
            ax.text((top_X[0]+top_X[-1])/2 - s*h*0.3, (top_Y[0]+top_Y[-1])/2 + c*h*0.3, f"{abs(l.w/1000):.1f} kN/m", color='purple', fontweight='bold', ha='center', va='center')

    def plot_internal_force_diagram(self, U_global: np.ndarray, element_loads: list = None, force_type: str = 'moment', scale: float = 0.01):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for el in self.elements:
            # 1. 畫出結構原始骨架
            ax.plot([el.node_i.x, el.node_j.x], [el.node_i.y, el.node_j.y], 'k-', alpha=0.3)
            
            # 2. 計算節點端點內力
            f = el.compute_internal_forces(U_global, element_loads)
            
            # 計算該桿件上的總均佈載重 (w)
            w_total = 0.0
            if element_loads:
                for load in element_loads:
                    if load.element.element_id == el.element_id:
                        w_total += load.w
            
            # 3. 建立連續點以繪製平滑曲線 (將桿件切成 21 個點)
            L = el.length
            x_vals = np.linspace(0, L, 21)
            
            if force_type.lower() == 'moment': 
                c_color, t = 'royalblue', 'Bending Moment Diagram (M-D)'
                # 🌟 彎矩真實物理方程式： M(x) = M_i - V_i*x - 0.5*w*x^2
                y_vals = f[2] - f[1] * x_vals - 0.5 * w_total * (x_vals ** 2)
            else: 
                c_color, t = 'mediumseagreen', 'Shear Force Diagram (V-D)'
                # 🌟 剪力真實物理方程式： V(x) = V_i + w*x
                y_vals = f[1] + w_total * x_vals
            
            # 取得兩端點數值供文字標註
            v_i, v_j = y_vals[0], y_vals[-1]
            
            # 4. 座標轉換黑魔法 (首尾補 0，形成封閉多邊形以便填色)
            l_x = np.concatenate(([0], x_vals, [L]))
            l_y = np.concatenate(([0], y_vals * scale, [0]))
            
            cos, sin = np.cos(el.angle), np.sin(el.angle)
            gX = el.node_i.x + l_x * cos - l_y * sin
            gY = el.node_i.y + l_x * sin + l_y * cos
            
            # 繪製曲線外框並填色 (gX[1:-1] 代表只畫拋物線本體，不畫連回軸線的直線)
            ax.plot(gX[1:-1], gY[1:-1], color=c_color, linewidth=2)
            ax.fill(gX, gY, color=c_color, alpha=0.4)
            
            # 5. 標註數值
            unit = "kN·m" if force_type.lower() == 'moment' else "kN"
            
            # 標註 i 端
            ax.text(gX[1], gY[1], f"{v_i/1000:.1f} {unit}", 
                    color='darkred', fontsize=9, fontweight='bold', ha='center', 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            # 標註 j 端
            ax.text(gX[-2], gY[-2], f"{v_j/1000:.1f} {unit}", 
                    color='darkred', fontsize=9, fontweight='bold', ha='center', 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                    
            # 🌟 (加碼功能) 如果是拋物線，自動在圖上標註出「曲線極大值」
            if force_type.lower() == 'moment' and w_total != 0:
                max_idx = np.argmax(np.abs(y_vals))
                # 如果最大彎矩發生在桿件中間 (不是在兩端)，就用黃色底框特別標示出來
                if max_idx != 0 and max_idx != 20: 
                    ax.text(gX[max_idx+1], gY[max_idx+1], f"{y_vals[max_idx]/1000:.1f} {unit}", 
                            color='black', fontsize=9, fontweight='bold', ha='center', 
                            bbox=dict(facecolor='gold', alpha=0.8, edgecolor='none'))

        ax.set_aspect('equal')
        ax.set_title(t, fontsize=12, fontweight='bold')

# ==========================================\n
# 5. 資料庫匯出函式 (新增至 engine.py 最後面)\n
# ==========================================\n
def export_extreme_values_to_db(db_path: str, case_name: str, elements: list, U_global: np.ndarray, element_loads: list):
    """計算全結構的最大剪力與彎矩，並存回資料庫 (純新增模式)"""
    max_V = 0.0
    max_M = 0.0
    
    # 1. 尋找全結構的極端值
    for el in elements:
        f = el.compute_internal_forces(U_global, element_loads)
        el_max_V = max(abs(f[1]), abs(f[4]))
        el_max_M = max(abs(f[2]), abs(f[5]))
        
        if el_max_V > max_V: max_V = el_max_V
        if el_max_M > max_M: max_M = el_max_M
        
    # 2. 寫入 SQLite 資料庫 (保留所有紀錄，不覆蓋)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Analysis_Results (case_name, max_moment, max_shear)
        VALUES (?, ?, ?)
    ''', (case_name, max_M, max_V))
    
    conn.commit()
    conn.close()
    
    print("\n💾 分析結果已成功新增至資料庫！")
    print(f"   ▶ 案例名稱：{case_name}")
    print(f"   ▶ 全系統最大剪力：{max_V/1000:.2f} kN")
    print(f"   ▶ 全系統最大彎矩：{max_M/1000:.2f} kN-m")