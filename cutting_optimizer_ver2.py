from read_csv import getPatterns
import itertools
import time

def expand_required_cuts(required_dict):
    """
    必要な切り出し長さの辞書を展開してリストに変換
    """
    expanded = []
    for length, count in required_dict.items():
        expanded.extend([int(length)] * count)
    return expanded

def generate_all_combinations(available_rods, required_cuts):
    """
    指定された棒の長さから切り出し可能な全ての組み合わせを生成
    """
    all_combinations = []
    # 大きいrodから小さいrodの切出しを考慮
    # required_cuts.extend(available_rods)
    
    # 1本から最大可能本数まで全ての組み合わせを試す
    for rod_length in available_rods:
        combinations = set()
        max_pieces = rod_length // min(required_cuts)

        for num_pieces in range(1, max_pieces + 1):
            for combo in itertools.combinations(required_cuts, num_pieces):
                # if sum(combo) <= rod_length and not set(combo).issubset(set(available_rods)):
                if sum(combo) <= rod_length:
                    combinations.add(combo)

        # カットパターンと歩留り率の辞書を追加
        for combo in combinations:
            loss = (rod_length - sum(combo)) #/ rod_length
            all_combinations.append({
                'rod_length': rod_length,
                'cuts': tuple(combo),
                'loss': loss,
            })
    
    # ロス率の低い順にソート
    all_combinations.sort(key=lambda x: x['loss'], reverse=False)

    # 結果表示（最初の20通り）
    # for i, combo in enumerate(all_combinations[:20]):
    #     print(f"{i+1:3d}. {combo['rod_length']} = {combo['cuts']}: {combo['loss']}")
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

def dual_optimal_cutting_plan(c, a, q):
    """
    max: Σ q_i * y_i
    s.t. Σ a_ji * y_i <= c_j (j=1~n)

    c: ロス率のリスト
    a: 各組み合わせのカット数のリスト
    q: 必要なカット数のリスト
    """
    import pulp

    n = len(a)  # パターン数
    m = len(q)  # 切り出し長さの種類数

    # 問題の定義
    prob = pulp.LpProblem("CuttingStockProblem", pulp.LpMaximize)

    # 変数の定義
    y = [pulp.LpVariable(f"y{i+1}", cat='Integer') for i in range(m)]

    # 目的関数:
    objective = pulp.lpSum(q[i] * y[i] for i in range(m))
    prob += objective

    # 制約条件:
    for j in range(n):
        production_constraint = pulp.lpSum(a[j][i] * y[i] for i in range(m))
        prob += production_constraint <= c[j]

    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 結果の表示    
    if prob.status == pulp.LpStatusOptimal:
        optimal_solution = [var.varValue for var in y]
        optimal_value = pulp.value(prob.objective)
        return optimal_solution, optimal_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None
    
def knapsack(y, l, available_rods):
    """
    max: Σ y_i * a_i
    s.t. Σ l_i * a_i <= available_rods[k] (k=1~len(ava_rods))

    y: 暫定の双対問題の最適解
    l: Required lengths
    """
    import pulp

    m = len(y)  # 切り出し長さの種類数

    # 問題の定義
    prob = pulp.LpProblem("CuttingStockProblem", pulp.LpMaximize)

    # 変数の定義
    a = [pulp.LpVariable(f"a{i+1}", cat='Integer') for i in range(m)]

    # 目的関数:
    objective = pulp.lpSum(y[i] * a[i] for i in range(m))
    prob += objective

    # 制約条件:
    """
    s.t. <=4000と<=4500などの場合、<=4500は制約の意味がない
    """
    for k in range(len(available_rods)):
        production_constraint = pulp.lpSum(l[i] * a[i] for i in range(m))
        prob += production_constraint <= available_rods[k]

    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 結果の表示    
    if prob.status == pulp.LpStatusOptimal:
        optimal_solution = [var.varValue for var in y]
        optimal_value = pulp.value(prob.objective)
        return optimal_solution, optimal_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None

if __name__ == "__main__":
    # 設定径: D10, D13, D19
    diameter = 'D10'
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

    while True:
        # 双対問題
        dual_solution, dual_value = dual_optimal_cutting_plan(c, a, q)
        # print(optimal_solution2)
        # print(optimal_value2)

        # knapsack problem
        knap_solution, knap_value = knapsack(dual_solution, l, available_rods)

        # 主問題に遷移用
        print(knap_value)
        knap_value=0

        if knap_value > 1:
            a.append(knap_solution)
            # ロス率どう求める？
            # どのava_rodsから切り出したのかがわからない
            # all_combにも情報を反映させないといけない
        else:
            break

    # 主問題
    main_solution, main_value = optimal_cutting_plan(c, a, q)

    # 最適な切り出し結果
    k = 1
    total_rod_length = 0
    used_length = 0
    used_list = []
    for i in range(len(all_combinations)):
        j = main_solution[i]
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
        print(f"Time: {end - start:.4f} [s]")
        # 端材
        print(f"Loss length: {total_rod_length - used_length} [mm]")
        # 使用材の合計
        print(f"Total rod length: {total_rod_length} [mm]")
        # 歩留り率
        print(f"Yield rate: {used_length * 100 / total_rod_length:.2f} [%]")
    else:
        print("Used_cuts not equal Required_cuts")
        print(q)
        print(used_count)