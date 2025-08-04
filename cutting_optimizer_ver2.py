from read_xlsx import get_patterns_from_xlsx
import time

# include available_rods size in the cut combo
isSurplus = False

def dfs(index, current_combination, current_sum, remaining_counts, sorted_numbers, max_sum, all_combinations):
    """
    深さ優先探索の再帰関数
    
    Args:
        index: 現在処理している整数のインデックス
        current_combination: 現在の組み合わせ
        current_sum: 現在の合計値
        remaining_counts: 各整数の残り使用可能個数
        sorted_numbers: ソート済みの整数リスト
        max_sum: 最大合計値
        all_combinations: 結果を格納するset
    """
    # 現在の組み合わせが有効な場合（合計値が0より大きく、max_sum以下）
    if 0 < current_sum <= max_sum:
        # 組み合わせをソートしてタプルに変換（重複判定のため）
        sorted_combo = tuple(sorted(current_combination))
        all_combinations.add((sorted_combo, current_sum))
    
    # 合計値がmax_sumを超えた場合、または全ての整数を処理した場合は終了
    if current_sum > max_sum or index >= len(sorted_numbers):
        return
    
    current_number = sorted_numbers[index]
    max_count = remaining_counts[current_number]
    
    # 現在の整数を0個から最大個数まで使用する全ての場合を試す
    for count in range(max_count + 1):
        new_sum = current_sum + current_number * count
        
        # 枝刈り: 新しい合計値がmax_sumを超える場合はスキップ
        if new_sum > max_sum:
            break
        
        # 現在の整数をcount個使用した場合の組み合わせを作成
        new_combination = current_combination + [current_number] * count
        new_remaining = remaining_counts.copy()
        new_remaining[current_number] -= count
        
        # 次の整数に進む
        dfs(index + 1, new_combination, new_sum, new_remaining, sorted_numbers, max_sum, all_combinations)

def find_combinations_dfs(numbers_dict, max_sum=7500):
    """
    深さ優先探索を使って、指定された最大合計値以下となる整数の組み合わせを全て求める
    
    Args:
        numbers_dict: {整数値: 最大個数} の辞書
        max_sum: 最大合計値（デフォルト: 7500）
    
    Returns:
        組み合わせのリスト [(組み合わせ, 合計値), ...]
    """
    # 整数を降順でソートして効率的な枝刈りを可能にする
    sorted_numbers = sorted(numbers_dict.keys(), reverse=True)
    all_combinations = set()  # 重複を避けるためにsetを使用
    
    # 初期値で探索開始
    dfs(0, [], 0, numbers_dict.copy(), sorted_numbers, max_sum, all_combinations)
    
    # setからリストに変換して返す
    return [(list(combo), sum_val) for combo, sum_val in all_combinations]

def generate_all_combinations(available_rods, required_cuts):
    """
    指定された棒の長さから切り出し可能な全ての組み合わせを生成
    """
    # 大きいAvailable_rodから小さいAvailable_rodのカットを許可
    if isSurplus:
        required_cuts.extend(available_rods)

    # この長さの棒から切り出し可能な組み合わせを求める
    combinations = find_combinations_dfs(required_cuts, max(available_rods))
    # 結果を和でソート
    combinations.sort(key=lambda x: x[1])
    # print(combinations[:20])
    # print(f"{len(combinations)} combinations")

    # 小さいavailable_rodsからtotal_cut_lengthを切り出す
    available_rods.sort()
    i = 0

    # カットパターンと歩留り率の辞書を追加
    all_combinations = []
    for combo, total_cut_length in combinations:
        if isSurplus and set(combo).issubset(set(available_rods)):
            continue
        while available_rods[i] < total_cut_length:
            i += 1
        loss = (available_rods[i] - total_cut_length) / available_rods[i]
        all_combinations.append({
            'rod_length': available_rods[i],
            'cuts': tuple(combo),
            'loss': loss,
        })
    
    # ロス率の低い順にソート
    all_combinations.sort(key=lambda x: x['loss'], reverse=False)

    # 結果表示（最初の20通り）
    # for i, combo in enumerate(all_combinations[:20]):
    #     print(f"{i+1:3d}. {combo['rod_length']} = {combo['cuts']} [{combo['loss']}]")
    # print(f"{len(all_combinations)} combinations")

    return all_combinations

def optimal_cutting_plan(c, a, q):
    """
    最適な切り出しプランを計算
    c: ロス率のリスト
    a: 各組み合わせのカット数のリスト
    q: 必要なカット数のリスト
    """
    import pulp

    n = len(a)  # パターン数
    m = len(q)  # 切り出し長さの種類数

    # 問題の定義
    prob = pulp.LpProblem("CuttingStockProblem", pulp.LpMinimize)

    # 変数の定義
    x = [pulp.LpVariable(f"y{j+1}", lowBound=0, cat='Integer') for j in range(n)]

    # 目的関数: 総ロス率を最小化
    objective = pulp.lpSum(c[j] * x[j] for j in range(n))
    prob += objective, "Total_Loss"

    # 制約条件: 各長さiに対して、生産量 == 必要量
    for i in range(m):
        production_constraint = pulp.lpSum(a[j][i] * x[j] for j in range(n))
        prob += production_constraint == q[i], f"Demand_constraint_{i+1}"

    # 問題を解く、timeLimitを設定
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=180, msg=False))
    # 結果の表示    
    if prob.status == pulp.LpStatusOptimal:
        optimal_solution = [var.varValue for var in x]
        optimal_value = pulp.value(prob.objective)
        return optimal_solution, optimal_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None

def main():
    # 設定径: D10, D13, D16, D19, D22
    diameter = 'D10'
    
    # Excelファイルから集約済みデータを取得
    pattern = get_patterns_from_xlsx("required_cuts.xlsx", diameter)
    available_rods = pattern['base_patterns']
    required_cuts = pattern['tasks']

    print(diameter)
    print(f"Available rods:\n{available_rods}")
    print(f"Required cuts:\n{required_cuts}")
    print()
    
    l = [int(s) for s in required_cuts.keys()]      # Required lengths
    q = [int(s) for s in required_cuts.values()]    # Required counts
    
    # 全組み合わせを計算
    start = time.perf_counter()
    all_combinations = generate_all_combinations(available_rods, required_cuts)
    end = time.perf_counter()
    time1 = end - start
    print(f"{len(all_combinations)} [combinations]")
    print(f"gen_patterns: {time1:.4f} [s]")

    # 最適化問題用に変数を定義
    a = []
    c = []
    for combo in all_combinations:
        a.append([combo['cuts'].count(i) for i in l])
        c.append(combo['loss'])
   
    # 最適な切り出しプランを計算
    start = time.perf_counter()
    optimal_solution, optimal_value = optimal_cutting_plan(c, a, q)
    end = time.perf_counter()
    time2 = end - start
    print(f"opt_solve: {time2:.4f} [s]")
    # print(f"optimal_solution:\n{optimal_solution}\n")
    # print(f"optimal_value: {optimal_value}\n")
    print()

    if optimal_solution is None:
        print("最適化に失敗しました。")
        return

    # 最適な切り出し結果
    k = 1
    total_rod_length = 0
    used_length = 0
    used_list = []
    for i in range(len(all_combinations)):
        j = optimal_solution[i]
        while(j > 0):
            print(f"{k:3d}. {all_combinations[i]['rod_length']} = {all_combinations[i]['cuts']} [{all_combinations[i]['rod_length']-sum(all_combinations[i]['cuts'])}]")
            total_rod_length += all_combinations[i]['rod_length']
            used_length += sum(all_combinations[i]['cuts'])
            used_list.extend(all_combinations[i]['cuts'])
            k += 1
            j -= 1
    print()

    # 要求本数と解の切り出し個数が同じかチェック
    used_count = [used_list.count(i) for i in l]
    
    if (used_count == q):
        # カットパターン探索時間
        print(f"Time:   {(time1+time2):.4f} [s]")
        # 端材
        print(f"Loss:   {total_rod_length - used_length} [mm]")
        # 使用材の合計
        print(f"Total:  {total_rod_length} [mm]")
        # 歩留り率
        print(f"Rate: {used_length * 100 / total_rod_length:.2f} [%]")
        
        # 切断指示の詳細表示
        print(f"\n=== 切断指示詳細 ===")
        total_pieces = sum(required_cuts.values())
        cutting_types = len(required_cuts)
        print(f"切断種類数: {cutting_types}")
        print(f"総切断本数: {total_pieces}")
        for length, count in sorted(required_cuts.items(), reverse=True):
            print(f"  {length}mm × {count}本")
    else:
        print("Used_cuts not equal Required_cuts")
        print(f"Required_cuts:  {q}")
        print(f"Used_cuts:      {used_count}")

if __name__ == "__main__":
    main()