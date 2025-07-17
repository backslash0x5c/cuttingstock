import streamlit as st
import pandas as pd
import time
import pulp
from datetime import datetime
import io

# ベースパターンの設定
BASE_PATTERNS = {
    'D10': [4000, 4500, 5500, 6000],
    'D13': [3000, 4000, 4500, 5500, 6000, 7500],
    'D16': [3000, 4000, 4500, 5500, 6000, 7000],
    'D19': [3500, 4000, 4500, 5500, 6000],
    'D22': [4000, 4500, 5500, 6000]
}

def expand_required_cuts(required_dict):
    """必要な切り出し長さの辞書を展開してリストに変換"""
    expanded = []
    for length, count in required_dict.items():
        expanded.extend([int(length)] * count)
    return expanded

def find_all_unique_combinations(available_rods, required_cuts):
    """与えられた数列から和がavailable_rods以下となる全ての異なる組み合わせを求める"""
    unique_combinations = set()
    
    def backtrack(index, current_subset, current_sum):
        if current_sum <= available_rods and current_sum > 0:
            sorted_subset = tuple(sorted(current_subset))
            unique_combinations.add((sorted_subset, current_sum))
        
        if current_sum > available_rods or index >= len(required_cuts):
            return
        
        backtrack(index + 1, current_subset, current_sum)
        
        current_subset.append(required_cuts[index])
        backtrack(index + 1, current_subset, current_sum + required_cuts[index])
        current_subset.pop()
    
    backtrack(0, [], 0)
    
    return [(list(combo), sum_val) for combo, sum_val in unique_combinations]

def generate_all_combinations(available_rods, required_cuts, is_surplus=False):
    """指定された棒の長さから切り出し可能な全ての組み合わせを生成"""
    if is_surplus:
        required_cuts.extend(available_rods)

    combinations = find_all_unique_combinations(max(available_rods), required_cuts)
    combinations.sort(key=lambda x: x[1])

    available_rods.sort()
    i = 0

    all_combinations = []
    for combo, total_cut_length in combinations:
        if is_surplus and set(combo).issubset(set(available_rods)):
            continue
        while available_rods[i] < total_cut_length:
            i += 1
        loss = (available_rods[i] - total_cut_length) / available_rods[i]
        all_combinations.append({
            'rod_length': available_rods[i],
            'cuts': tuple(combo),
            'loss': loss,
        })
    
    all_combinations.sort(key=lambda x: x['loss'], reverse=False)
    return all_combinations

def optimal_cutting_plan(c, a, q):
    """最適な切り出しプランを計算"""
    n = len(a)
    m = len(q)

    prob = pulp.LpProblem("CuttingStockProblem", pulp.LpMinimize)
    x = [pulp.LpVariable(f"y{j+1}", lowBound=0, cat='Integer') for j in range(n)]

    objective = pulp.lpSum(c[j] * x[j] for j in range(n))
    prob += objective, "Total_Loss"

    for i in range(m):
        production_constraint = pulp.lpSum(a[j][i] * x[j] for j in range(n))
        prob += production_constraint == q[i], f"Demand_constraint_{i+1}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status == pulp.LpStatusOptimal:
        optimal_solution = [var.varValue for var in x]
        optimal_value = pulp.value(prob.objective)
        return optimal_solution, optimal_value
    else:
        return None, None

def parse_csv_data(csv_data):
    """CSVデータをパースして必要な形式に変換"""
    try:
        df = pd.read_csv(io.StringIO(csv_data), index_col=1)
        df = df.dropna(subset=[df.columns[0]])
        
        # 不要な列を削除
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # NaNを-1に置き換えてから整数型に変換
        df = df.fillna(-1).astype(int)
        
        tasks = df.to_dict(orient='dict')
        
        # 値が0より大きいもののみを残す
        tasks = {
            outer_key: {inner_key: value for inner_key, value in inner_dict.items() if value > 0}
            for outer_key, inner_dict in tasks.items()
        }
        
        return tasks
    except Exception as e:
        st.error(f"CSVファイルの解析に失敗しました: {e}")
        return None

def display_cutting_patterns(cutting_patterns):
    """切断パターンを見やすい箇条書き形式で表示"""
    if not cutting_patterns:
        st.write("切断パターンがありません")
        return
    
    st.write("**切断パターン詳細:**")
    for i, pattern in enumerate(cutting_patterns):
        # 切断パターンを整理して表示
        cuts_str = " + ".join([str(cut) for cut in pattern['cuts']])
        st.write(f"• **{i+1}本目:** {pattern['rod_length']}mm → [{cuts_str}] (端材: {pattern['waste']}mm)")

def main():
    st.set_page_config(page_title="鉄筋切断最適化アプリ", layout="wide")
    
    st.title("🔧 鉄筋切断最適化アプリ")
    st.write("鉄筋の切断パターンを最適化し、材料の無駄を最小化します。")
    
    # サイドバーで実行履歴を表示
    with st.sidebar:
        st.header("📊 実行履歴")
        
        # セッション状態の初期化
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        if st.session_state.history:
            for i, record in enumerate(reversed(st.session_state.history)):
                with st.expander(f"{record['timestamp']} - {record['diameter']}"):
                    st.write(f"**径:** {record['diameter']}")
                    # st.write(f"**残材オプション:** {'有効' if record['surplus'] else '無効'}")
                    st.write(f"**歩留り率:** {record['yield_rate']:.2f}%")
                    st.write(f"**総材料長:** {record['total_length']} mm")
                    st.write(f"**端材:** {record['loss']} mm")
                    st.write(f"**処理時間:** {record['time']:.4f} s")
                    
                    # 必要な切り出し長さを表示
                    # st.write("**必要な切り出し:**")
                    # for length, count in record['required_cuts'].items():
                    #     st.write(f"• {length}mm × {count}本")
                    
                    # 切断パターンを見やすく表示
                    display_cutting_patterns(record['cutting_patterns'])
        else:
            st.write("実行履歴がありません")
    
    # メインコンテンツ
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("⚙️ 設定")
        
        # 径の選択
        diameter = st.selectbox(
            "鉄筋の径を選択してください:",
            options=list(BASE_PATTERNS.keys())
        )
        
        # 利用可能な棒の長さを表示
        st.write(f"**{diameter}の利用可能な棒の長さ:** {BASE_PATTERNS[diameter]}")
        
        # Surplusオプション
        # is_surplus = st.checkbox("残材の有効", value=False,
        #                         help="大きい生材から小さい生材の切り出しが有効になります")
        
        # 入力方法の選択
        input_method = st.radio(
            "入力方法を選択してください:",
            ("CSVファイルアップロード", "手入力")
        )
        
        required_cuts = {}
        
        if input_method == "手入力":
            st.subheader("必要な切り出し長さと本数")
            
            # 動的入力フィールド
            if 'input_rows' not in st.session_state:
                st.session_state.input_rows = 3
            
            for i in range(st.session_state.input_rows):
                col_length, col_count = st.columns([1, 1])
                
                with col_length:
                    length = st.number_input(
                        f"長さ {i+1} (mm)",
                        min_value=0,
                        value=0,
                        key=f"length_{i}"
                    )
                
                with col_count:
                    count = st.number_input(
                        f"本数 {i+1}",
                        min_value=0,
                        value=0,
                        key=f"count_{i}"
                    )
                
                if length > 0 and count > 0:
                    required_cuts[length] = count
            
            col_add, col_remove = st.columns([1, 1])
            with col_add:
                if st.button("行を追加"):
                    st.session_state.input_rows += 1
                    st.rerun()
            
            with col_remove:
                if st.button("行を削除") and st.session_state.input_rows > 1:
                    st.session_state.input_rows -= 1
                    st.rerun()
        
        else:  # CSVファイルアップロード
            st.subheader("CSVファイルアップロード")
            
            uploaded_file = st.file_uploader(
                "CSVファイルを選択してください",
                type=['csv'],
                help="csv形式のファイルをアップロードしてください"
            )
            
            if uploaded_file is not None:
                csv_data = uploaded_file.read().decode('utf-8')
                tasks_data = parse_csv_data(csv_data)
                
                if tasks_data and diameter in tasks_data:
                    required_cuts = tasks_data[diameter]
                    st.success(f"CSVファイルを正常に読み込みました。{diameter}のデータが見つかりました。")
                    st.write("**読み込まれたデータ:**")
                    st.write(required_cuts)
                else:
                    st.error(f"CSVファイルに{diameter}のデータが見つかりませんでした。")
    
    with col2:
        st.header("🎯 最適化結果")
        
        if st.button("最適化を実行", type="primary") and required_cuts:
            with st.spinner("最適化を実行中..."):
                start_time = time.perf_counter()
                
                # 利用可能な棒の長さを取得
                available_rods = BASE_PATTERNS[diameter].copy()
                
                # 必要な切り出し長さを展開
                l = [int(s) for s in required_cuts.keys()]
                q = [int(s) for s in required_cuts.values()]
                expand_required_cuts_list = expand_required_cuts(required_cuts)
                
                # 全組み合わせを計算
                all_combinations = generate_all_combinations(available_rods, expand_required_cuts_list)
                
                if not all_combinations:
                    st.error("有効な組み合わせが見つかりませんでした。")
                else:
                    # 最適化問題用に変数を定義
                    a = []
                    c = []
                    for combo in all_combinations:
                        a.append([combo['cuts'].count(i) for i in l])
                        c.append(combo['loss'])
                    
                    # 最適な切り出しプランを計算
                    optimal_solution, optimal_value = optimal_cutting_plan(c, a, q)
                    
                    if optimal_solution is not None:
                        end_time = time.perf_counter()
                        processing_time = end_time - start_time
                        
                        # 結果の計算
                        total_rod_length = 0
                        used_length = 0
                        used_list = []
                        cutting_patterns = []
                        
                        for i in range(len(all_combinations)):
                            j = int(optimal_solution[i])
                            for _ in range(j):
                                pattern = {
                                    'rod_length': all_combinations[i]['rod_length'],
                                    'cuts': all_combinations[i]['cuts'],
                                    'waste': all_combinations[i]['rod_length'] - sum(all_combinations[i]['cuts'])
                                }
                                cutting_patterns.append(pattern)
                                total_rod_length += all_combinations[i]['rod_length']
                                used_length += sum(all_combinations[i]['cuts'])
                                used_list.extend(all_combinations[i]['cuts'])
                        
                        # 検証
                        used_count = [used_list.count(i) for i in l]
                        
                        if used_count == q:
                            loss = total_rod_length - used_length
                            yield_rate = used_length * 100 / total_rod_length
                            
                            # 結果を表示
                            st.success("最適化が完了しました！")
                            
                            # サマリー表示
                            col_summary1, col_summary2 = st.columns([1, 1])
                            
                            with col_summary1:
                                st.metric("歩留り率", f"{yield_rate:.2f}%")
                                st.metric("総材料長", f"{total_rod_length} mm")
                            
                            with col_summary2:
                                st.metric("端材", f"{loss} mm")
                                st.metric("処理時間", f"{processing_time:.4f} s")
                            
                            # 切断パターンの表示
                            st.subheader("切断パターン")
                            
                            df_results = pd.DataFrame([
                                {
                                    'No.': i + 1,
                                    '棒の長さ': pattern['rod_length'],
                                    '切断': str(pattern['cuts']),
                                    '端材': pattern['waste']
                                }
                                for i, pattern in enumerate(cutting_patterns)
                            ])
                            
                            st.dataframe(df_results, use_container_width=True)
                            
                            # 履歴に追加
                            history_record = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'diameter': diameter,
                                # 'surplus': is_surplus,
                                'yield_rate': yield_rate,
                                'total_length': total_rod_length,
                                'loss': loss,
                                'time': processing_time,
                                'required_cuts': required_cuts,
                                'cutting_patterns': cutting_patterns
                            }
                            st.session_state.history.append(history_record)
                            
                            # CSVダウンロード
                            csv_data = df_results.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="結果をCSVでダウンロード",
                                data=csv_data,
                                file_name=f"cutting_result_{diameter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
                        else:
                            st.error("最適化に失敗しました。要求本数と切り出し個数が一致しません。")
                    else:
                        st.error("最適解が見つかりませんでした。")
        
        # elif not required_cuts:
        #     st.info("切断指示を入力してください。")
    
    # フッター
    st.markdown("---")
    st.markdown("💡 **使用方法:**")
    st.markdown("1. 鉄筋の径を選択")
    st.markdown("2. CSVファイルまたは手入力より切断指示を与える")
    st.markdown("3. 最適化を実行")

if __name__ == "__main__":
    main()