import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import base64

# 確保從你的 engine.py 匯入所有核心功能
from engine import ModelDatabaseLoader, LinearStaticSolver, ResultVisualizer, export_extreme_values_to_db

st.set_page_config(page_title="二維結構 OOP 分析系統", layout="wide")
st.title("🏗️ 二維結構分析系統 ")

DB_PATH = "structure_v2.db"

# ==========================================
# 🚨 新增：自動修復與初始化資料庫功能
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 檢查資料庫裡面有沒有 Nodes 這個表格
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Nodes'")
    if not cursor.fetchone():
        # 如果找不到，代表是全新或被刪除的空資料庫，自動寫入預設的「ㄇ字型剛構架」資料
        pd.DataFrame([
            {"node_id": 1, "x_coord": 0, "y_coord": 0, "rx": 1, "ry": 1, "rmz": 1},
            {"node_id": 2, "x_coord": 0, "y_coord": 4, "rx": 0, "ry": 0, "rmz": 0},
            {"node_id": 3, "x_coord": 5, "y_coord": 4, "rx": 0, "ry": 0, "rmz": 0},
            {"node_id": 4, "x_coord": 5, "y_coord": 0, "rx": 1, "ry": 1, "rmz": 1}
        ]).to_sql("Nodes", conn, index=False)
        
        pd.DataFrame([
            {"id": 1, "name": "Steel", "E_value": 2e11, "I_value": 0.0005, "A_value": 0.02}
        ]).to_sql("Materials", conn, index=False)
        
        pd.DataFrame([
            {"element_id": 1, "node_i": 1, "node_j": 2, "material_id": 1},
            {"element_id": 2, "node_i": 2, "node_j": 3, "material_id": 1},
            {"element_id": 3, "node_i": 3, "node_j": 4, "material_id": 1}
        ]).to_sql("Elements", conn, index=False)
        
        pd.DataFrame([
            {"load_id": 1, "target_type": "NODE", "target_id": 2, "fx": 50000, "fy": 0, "mz": 0}
        ]).to_sql("Loads", conn, index=False)
    conn.close()

# 網頁啟動時，強制執行一次檢查！
init_db()
# ==========================================



# --- 輔助函式：讀取與寫入資料庫 ---
def load_data(table_name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def save_data(table_name, df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

# --- UI 佈局：左側輸入參數，右側顯示結果 ---
col_input, col_result = st.columns([1, 1.2])

with col_input:
    st.header("📝 模型參數輸入")
    
    # 🌟 盲點二修正：讓使用者可以直接在網頁上自訂「案例名稱」！
    # 這樣每一筆歷史紀錄才會有獨一無二的名字
    case_name = st.text_input("🏷️ 請輸入本次分析案例名稱", value="ㄇ字型剛構架_案例01")
    st.markdown("---")

    # 1. 節點輸入 (Nodes)
    st.subheader("1. 節點定義 (Nodes)")
    df_nodes = load_data("Nodes")
    edited_nodes = st.data_editor(df_nodes, num_rows="dynamic", key="editor_nodes")

    # 2. 桿件輸入 (Elements)
    st.subheader("2. 桿件定義 (Elements)")
    df_elements = load_data("Elements")
    edited_elements = st.data_editor(df_elements, num_rows="dynamic", key="editor_elements")

    # 3. 載重輸入 (Loads)
    st.subheader("3. 外力載重 (Loads)")
    df_loads = load_data("Loads")
    edited_loads = st.data_editor(df_loads, num_rows="dynamic", key="editor_loads")

    # 執行按鈕
    run_button = st.button("🚀 儲存並執行分析", type="primary", use_container_width=True)

# --- 執行分析與視覺化 ---
with col_result:
    st.header("📊 分析結果與視覺化")
    
    if run_button:
        with st.spinner('力學引擎運算中，請稍候...'):
            try:
                # 步驟 A: 將使用者修改的模型覆寫回資料庫
                save_data("Nodes", edited_nodes)
                save_data("Elements", edited_elements)
                save_data("Loads", edited_loads)

                # 步驟 B: 呼叫 V2 引擎讀取模型
                loader = ModelDatabaseLoader(DB_PATH)
                nodes, elements = loader.load_model()
                nodal_loads, element_loads = loader.load_loads()

                # 步驟 C: 矩陣位移法計算
                solver = LinearStaticSolver(nodes, elements)
                U_global = solver.solve(nodal_loads=nodal_loads, element_loads=element_loads)

                if U_global is not None:
                    st.success("✅ 矩陣位移法求解成功！")
                    
                    # 🌟 步驟 D：將極值分析結果「寫入/新增」到 Analysis_Results 資料表
                    # 傳入剛剛在網頁文字框輸入的 case_name
                    export_extreme_values_to_db(
                        db_path=DB_PATH,
                        case_name=case_name,
                        elements=elements,
                        U_global=U_global,
                        element_loads=element_loads
                    )
                    
                    st.markdown("---")
                    st.subheader("⚠️ 結構安全自動檢核")
                    
                    # 從資料庫中讀取剛剛存進去的最新一筆極值
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT max_moment, max_shear FROM Analysis_Results ORDER BY id DESC LIMIT 1")
                    latest_result = cursor.fetchone()
                    conn.close()
                    
                    max_M_kN = latest_result[0] / 1000
                    max_V_kN = latest_result[1] / 1000
                    
                    # 假設我們的鋼材容許彎矩是 150 kN-m，容許剪力是 100 kN
                    allowable_M = 150.0 
                    allowable_V = 100.0
                    
                    if max_M_kN > allowable_M or max_V_kN > allowable_V:
                        st.error(f"❌ 警告：桿件內力超過材料容許值！\n\n系統最大彎矩：{max_M_kN:.1f} kN-m (上限 {allowable_M})\n系統最大剪力：{max_V_kN:.1f} kN (上限 {allowable_V})\n\n建議：請加大桿件斷面積 (A) 與慣性矩 (I)。")
                    else:
                        st.success(f"✅ 安全通過檢核：全區桿件內力皆在容許範圍內。\n\n系統最大彎矩：{max_M_kN:.1f} kN-m (上限 {allowable_M})\n系統最大剪力：{max_V_kN:.1f} kN (上限 {allowable_V})")
                    st.markdown("---")

                    # 步驟 E: 渲染 Matplotlib 圖表到網頁
                    vis = ResultVisualizer(nodes, elements)
                    
                    st.subheader("結構與載重模型")
                    vis.plot_model(nodal_loads, element_loads)
                    st.pyplot(plt.gcf())
                    plt.clf()

                    st.subheader("剪力圖 (V-D)")
                    vis.plot_internal_force_diagram(U_global, element_loads, force_type='shear', scale=0.0001)
                    st.pyplot(plt.gcf())
                    plt.clf()

                    st.subheader("彎矩圖 (M-D)")
                    vis.plot_internal_force_diagram(U_global, element_loads, force_type='moment', scale=0.00005)
                    
                    # 🌟 步驟 2：利用 Base64 技術把圖片轉成文字
                    fig = plt.gcf()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight', transparent=False, facecolor='white')
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode()
                    data_url = f"data:image/png;base64,{img_b64}" # 這就是我們要存進資料庫的圖片文字
                    
                    # 顯示在網頁上並清除畫布
                    st.pyplot(fig)
                    plt.clf()

                    # 🌟 步驟 3：呼叫引擎，把數字跟「圖片文字(data_url)」一起存進資料庫
                    export_extreme_values_to_db(
                        db_path=DB_PATH,
                        case_name=case_name,
                        elements=elements,
                        U_global=U_global,
                        element_loads=element_loads,
                        image_url=data_url  # 傳入圖片！
                    )

            except Exception as e:
                st.error(f"❌ 運算發生錯誤：{e}")
    else:
        st.write("👈 請在左側設定參數後，點擊「儲存並執行分析」。")

# =======================================================
# 🌟 終極加碼：網頁最下方的「即時歷史紀錄看板」
# =======================================================
st.markdown("---")
st.header("📜 歷史分析紀錄看板")
try:
    # 直接用 pandas 去撈 Analysis_Results 資料表
    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query("SELECT * FROM Analysis_Results ORDER BY id DESC", conn)
    conn.close()
    
    if not df_history.empty:
         # 為了方便口試或報告看，自動把 N 和 N-m 換算成工程常用的 kN 和 kN-m
         df_history['最大彎矩 (kN-m)'] = (df_history['max_moment'] / 1000).round(2)
         df_history['最大剪力 (kN)'] = (df_history['max_shear'] / 1000).round(2)
         
         # 重新調整欄位順序並精簡呈現
         df_display = df_history[['id', 'case_name', '最大彎矩 (kN-m)', '最大剪力 (kN)']]
         df_display = df_history[['id', 'case_name', '最大彎矩 (kN-m)', '最大剪力 (kN)', 'image_url']]
         
         # 使用 ImageColumn 讓網頁把文字解析回圖片
         st.dataframe(
             df_display,
             column_config={
                 "id": "編號",
                 "case_name": "案例名稱",
                 "image_url": st.column_config.ImageColumn(
                     "彎矩圖預覽 (M-D)", help="歷次分析的彎矩圖形"
                 )
             },
             use_container_width=True,
             hide_index=True
         )
    else:
         st.info("💡 目前資料表裡還沒有歷史紀錄喔！請輸入模型並按下分析按鈕。")
except Exception as e:
    st.error(f"無法讀取歷史紀錄資料表，請確認 Analysis_Results 是否存在。錯誤原因：{e}")