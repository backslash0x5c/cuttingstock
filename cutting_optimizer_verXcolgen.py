import pulp

def cut_bars(available_bars, required_cuts):
    """
    棒材から必要な長さを切り出すプログラム
    
    Args:
        available_bars: 利用可能な棒の長さリスト [4000, 4500, 5500, 6000]
        required_cuts: 必要な切り出し長さの辞書 {'長さ': 個数}
    
    Returns:
        patterns: 使用パターンのリスト
        total_bars_used: 使用した棒の総数
        loss_rates: ロス率のリスト
    """
    
    # 最大長の棒を使用
    max_bar_length = max(available_bars)
    # print(f"使用する棒の長さ: {max_bar_length}mm")
    
    # 必要な切り出し長さを展開（個数分リストに追加）
    cuts_needed = []
    cut_lengths = list(required_cuts.keys())
    
    for length_str, count in required_cuts.items():
        length = int(length_str)
        cuts_needed.extend([length] * count)
    
    # print(f"必要な切り出し: {cuts_needed}")
    # print(f"切り出し長さの種類: {cut_lengths}")
    
    # 使用パターンを記録するリスト
    patterns = []
    loss_rates = []  # ロス率を記録するリスト
    remaining_cuts = cuts_needed.copy()
    bar_count = 0
    
    while remaining_cuts:
        bar_count += 1
        current_bar_length = max_bar_length
        current_pattern = [0] * len(cut_lengths)  # 各長さの使用回数
        cuts_in_this_bar = []
        
        # print(f"\n--- 棒 {bar_count} ---")
        # print(f"残り長さ: {current_bar_length}mm")
        
        # 長い順にソートして効率的に切り出し
        remaining_cuts.sort(reverse=True)
        
        # 現在の棒から切り出し可能な長さを探す
        i = 0
        while i < len(remaining_cuts) and current_bar_length > 0:
            cut_length = remaining_cuts[i]
            
            if cut_length <= current_bar_length:
                # 切り出し実行
                current_bar_length -= cut_length
                cuts_in_this_bar.append(cut_length)
                
                # パターンに記録
                length_index = cut_lengths.index(str(cut_length))
                current_pattern[length_index] += 1
                
                # 残りリストから削除
                remaining_cuts.pop(i)
                
                # print(f"  {cut_length}mm を切り出し (残り: {current_bar_length}mm)")
            else:
                i += 1
        
        # パターンをタプルとして記録
        pattern_tuple = tuple(current_pattern)
        patterns.append(pattern_tuple)
        
        # ロス率を計算してリストcに追加
        used_length = max_bar_length - current_bar_length
        loss_rate = current_bar_length * 100 / max_bar_length
        loss_rates.append(loss_rate)
        
        # print(f"  切り出した長さ: {cuts_in_this_bar}")
        # print(f"  使用パターン: {pattern_tuple}")
        # print(f"  使用長さ: {used_length}mm")
        # print(f"  余り: {current_bar_length}mm")
        # print(f"  ロス率: {loss_rate:.4f} ({loss_rate*100:.2f}%)")
    
    return patterns, bar_count, loss_rates

def solve_optimization_problem(patterns, loss_rates, required_cuts):
    """
    最適化問題を解く
    
    変数: y = [y1, y2, ..., ym] (各パターンの使用回数)
    目的関数: max Σ(qj * yj) ただし qj は各長さの必要個数
    制約条件: Σ(aij * yi) <= cj for all j
    
    Args:
        patterns: 使用パターンのリスト [(1,0,0,0,0,0,0), (0,1,0,0,1,0,0), ...]
        loss_rates: ロス率のリスト [0.2508, 0.1083, ...]
        required_cuts: 必要な切り出し長さの辞書
    
    Returns:
        optimal_solution: 最適解
        optimal_value: 最適値
    """
    
    # cベクトル（ロス率）
    c = loss_rates
    # print(f"c (ロス率): {c}")

    # qベクトル（必要な切り出し長さのバリュー）
    q = list(required_cuts.values())  # [1, 2, 3, 1, 1, 4, 2]
    # print(f"q (必要個数): {q}")
    
    # パターン数
    n = len(patterns)
    # 切り出し長さの種類数
    m = len(q)
    
    # print(f"パターン数 (m): {n}")
    # print(f"切り出し長さの種類数 (n): {m}")
    
    # 最適化問題の定義
    prob = pulp.LpProblem("CuttingStockProblem", pulp.LpMaximize)
    
    # 変数の定義: y = [y1, y2, ..., ym]
    y = [pulp.LpVariable(f"y{i+1}", lowBound=0) for i in range(m)]
    
    # 目的関数: max Σ(qi * yi)
    objective = 0
    for i in range(m):
        objective += q[i] * y[i]
    
    prob += objective, "Total_Value"
    # print(prob.objective)
    
    # 制約条件: 各長さjに対して、生産量 <= 必要量
    for j in range(n):
        production_constraint = 0
        for i in range(m):
            production_constraint += patterns[j][i] * y[i]
        prob += production_constraint <= c[j], f"Demand_constraint_{j+1}"
        # print(prob.constraints[f"Demand_constraint_{j+1}"])
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 結果の表示
    # print(f"\n=== 最適化結果 ===")
    # print(f"ステータス: {pulp.LpStatus[prob.status]}")
    
    if prob.status == pulp.LpStatusOptimal:
        optimal_solution = [int(var.varValue) for var in y]
        optimal_value = pulp.value(prob.objective)
        
        # # 各パターンの使用回数と生産量の詳細
        # print(f"\n--- 詳細結果 ---")
        # total_bars_used = sum(optimal_solution)
        # total_loss = 0
        
        # for i, (pattern, loss_rate, usage) in enumerate(zip(patterns, loss_rates, optimal_solution)):
        #     print(f"パターン {i+1}: {pattern} (ロス率: {loss_rate:.4f})")
        #     total_loss += loss_rate * usage
        
        # print(f"\n使用棒数: {total_bars_used}本")
        
        # 生産量の確認
        # print(f"\n--- 生産量確認 ---")
        # cut_lengths = list(required_cuts.keys())
        # for i in range(m):
        #     produced = sum(patterns[j][i] * optimal_solution[i] for j in range(n))
        #     required = q[i]
        #     print(f"{cut_lengths[i]}mm: 生産{produced}本 / 必要{required}本")
        
        return optimal_solution, optimal_value
    else:
        print("最適解が見つかりませんでした")
        return None, None

def solve_second_optimization(optimal_y, required_cuts, available_bars):
    """
    第二段階最適化問題を解く
    
    変数: z = [z1, z2, ..., zm] (各切り出し長さの使用量)
    目的関数: min Σ(yi * zi)  (yiは第一段階の最適解)
    制約条件: Σ(li * zi) <= Lk for all k (liは切り出し長さ、Lkは利用可能な棒の長さ)
    
    Args:
        optimal_y: 第一段階の最適解 [y1, y2, ..., ym]
        required_cuts: 必要な切り出し長さの辞書 {'長さ': 個数}
        available_bars: 利用可能な棒の長さリスト [4000, 4500, 5500, 6000]
    
    Returns:
        optimal_z: 第二段階の最適解
        optimal_value: 第二段階の最適値
    """
    
    # 切り出し長さのリスト (li)
    cut_lengths = list(required_cuts.keys())
    cut_lengths_int = [int(length) for length in cut_lengths]
    
    # print(f"第一段階の最適解 y: {optimal_y}")
    # print(f"切り出し長さ li: {cut_lengths_int}")
    print(f"利用可能な棒の長さ Lk: {available_bars}")
    
    # 変数の数
    m = len(cut_lengths)
    # 利用可能な棒の数
    # k = len(available_bars)
    
    # 最適化問題の定義（最小化問題）
    prob = pulp.LpProblem("SecondStageOptimization", pulp.LpMaximize)
    
    # 変数の定義: z = [z1, z2, ..., zm]
    z = [pulp.LpVariable(f"z{i+1}", lowBound=0, cat='Integer') for i in range(m)]
    
    # 目的関数: min Σ(yi * zi)
    objective = 0
    for i in range(m):
        objective += optimal_y[i] * z[i]
    
    prob += objective, "Total_Cost"
    # print(f"目的関数: max {objective}")
    
    # 制約条件: Σ(li * zi) <= Lk for all k
    capacity_constraint = 0
    for i in range(m):
        capacity_constraint += cut_lengths_int[i] * z[i]
    prob += capacity_constraint <= available_bars, f"Capacity_constraint_{available_bars}"
    # print(f"制約 : {capacity_constraint} <= {available_bars}")
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 結果の表示
    # print(f"\n=== 第二段階最適化結果 ===")
    # print(f"ステータス: {pulp.LpStatus[prob.status]}")
    
    if prob.status == pulp.LpStatusOptimal:
        optimal_z = [var.varValue for var in z]
        optimal_value = pulp.value(prob.objective)
        
        # print(f"最適解: z = {optimal_z}")
        # print(f"最適値: {optimal_value}")
        
        return optimal_z, optimal_value
    else:
        print("最適解が見つかりませんでした")
        return None, None

# メイン実行部分
if __name__ == "__main__":
    # 入力データ
    available_bars = [4000, 4500, 5500, 6000]  # mm
    required_cuts = {
        '4495': 1,
        '3585': 2, 
        '2675': 3,
        '2600': 1,
        '1765': 1,
        '1080': 4,
        '855': 2
    }
    
    print("=== 棒材切り出しプログラム ===")
    print(f"利用可能な棒: {available_bars}mm")
    print(f"必要な切り出し長さ: {required_cuts}")
    
    # 第一段階: 切り出し実行
    patterns, total_bars, loss_rates = cut_bars(available_bars, required_cuts)
    
    # print(f"\n=== 第一段階結果 ===")
    # print(f"使用した棒の総数: {total_bars}本")
    
    cut_lengths = [int(s) for s in list(required_cuts.keys())]
    # print(f"必要なカット長さ種類: {cut_lengths}")
    
    second_optimal_value = 2
    while (second_optimal_value > 1 and len(patterns) < 10):
        for i, (pattern, loss_rate) in enumerate(zip(patterns, loss_rates), 1):
            print(f"  棒 {i}: {pattern} (ロス率: {loss_rate:.4f})")
        
        optimal_y, optimal_value = solve_optimization_problem(patterns, loss_rates, required_cuts)

        print(f"最適解 y: {optimal_y}")
        print(f"最適値: {optimal_value}")
        
        # 第二段階最適化問題を解く
        L_i = 0
        temp_value = -float('inf')
        for i in range(len(available_bars)):
            optimal_z, second_optimal_value = solve_second_optimization(optimal_y, required_cuts, available_bars[i])
            if second_optimal_value > temp_value:
                temp_value = second_optimal_value
                L_i = i
            else:
                break
        
        patterns.append(tuple(optimal_z))
        for i in range(len(optimal_z)):
            used_length = cut_lengths[i] * optimal_z[i]
        loss_rates.append(used_length * 100 / available_bars[L_i])