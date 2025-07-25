from read_csv import getPatterns
import time

# include available_rods size in the cut combo
isSurplus = False

def expand_required_cuts(required_dict):
    """
    必要な切り出し長さの辞書を展開してリストに変換
    """
    expanded = []
    for length, count in required_dict.items():
        expanded.extend([int(length)] * count)
    return expanded

def find_all_unique_combinations(available_rods, required_cuts):
    """
    与えられた数列から和がavailable_rods以下となる全ての異なる組み合わせを求める
    同じ和でも異なる組み合わせは保持し、完全に同じ組み合わせのみを重複排除
    
    Args:
        available_rods: 原材料（この値以下の和を持つ組み合わせを探す）
        required_cuts: 数値のリスト
    
    Returns:
        異なる組み合わせのリスト [(組み合わせ, 和), ...]
    """
    # 重複を避けるためにsetを使用（tupleに変換してハッシュ化）
    unique_combinations = set()
    
    def backtrack(index, current_subset, current_sum):
        # 現在の和がavailable_rods以下の場合
        if current_sum <= available_rods and current_sum > 0:
            # 組み合わせをソートしてタプルに変換（重複判定のため）
            sorted_subset = tuple(sorted(current_subset))
            unique_combinations.add((sorted_subset, current_sum))
        
        # 現在の和がavailable_rodsを超えた場合、これ以上探索しない
        if current_sum > available_rods or index >= len(required_cuts):
            return
        
        # 現在の要素を含まない場合
        backtrack(index + 1, current_subset, current_sum)
        
        # 現在の要素を含む場合
        current_subset.append(required_cuts[index])
        backtrack(index + 1, current_subset, current_sum + required_cuts[index])
        current_subset.pop()  # バックトラック
    
    backtrack(0, [], 0)
    
    # setからリストに変換して返す
    return [(list(combo), sum_val) for combo, sum_val in unique_combinations]

def generate_all_combinations(available_rods, required_cuts):
    """
    指定された棒の長さから切り出し可能な全ての組み合わせを生成
    """
    # 大きいAvailable_rodから小さいAvailable_rodのカットを許可
    if isSurplus:
        required_cuts.extend(available_rods)

    # この長さの棒から切り出し可能な組み合わせを求める
    combinations = find_all_unique_combinations(max(available_rods), required_cuts)
    # 結果を和でソート
    combinations.sort(key=lambda x: x[1])
    # print(combinations)
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

    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 結果の表示    
    if prob.status == pulp.LpStatusOptimal:
        optimal_solution = [var.varValue for var in x]
        optimal_value = pulp.value(prob.objective)
        return optimal_solution, optimal_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None

if __name__ == "__main__":
    # 設定径: D10, D13, D19
    diameter = 'D19'
    pattern = getPatterns(diameter)
    available_rods = pattern['base_patterns']
    required_cuts = pattern['tasks']

    print(diameter)
    print(f"Available rods: {available_rods}")
    print(f"Required cuts: {required_cuts}")
    
    l = [int(s) for s in required_cuts.keys()]      # Required lengths
    q = [int(s) for s in required_cuts.values()]    # Required counts
    
    # 必要な切り出し長さを展開
    expand_required_cuts_list = expand_required_cuts(required_cuts)
    
    # 全組み合わせを計算
    start = time.perf_counter()
    all_combinations = generate_all_combinations(available_rods, expand_required_cuts_list)
    end = time.perf_counter()
    print()

    # 最適化問題用に変数を定義
    a = []
    c = []
    for combo in all_combinations:
        a.append([combo['cuts'].count(i) for i in l])
        c.append(combo['loss'])
   
    # 最適な切り出しプランを計算
    optimal_solution, optimal_value = optimal_cutting_plan(c, a, q)
    # print(f"optimal_solution:\n{optimal_solution}\n")
    # print(f"optimal_value: {optimal_value}\n")

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
        print(f"Time:   {end - start:.4f} [s]")
        # 端材
        print(f"Loss:   {total_rod_length - used_length} [mm]")
        # 使用材の合計
        print(f"Total:  {total_rod_length} [mm]")
        # 歩留り率
        print(f"Yield_rate: {used_length * 100 / total_rod_length:.2f} [%]")
    else:
        print("Used_cuts not equal Required_cuts")
        print(f"Required_cuts:  {q}")
        print(f"Used_cuts:      {used_count}")