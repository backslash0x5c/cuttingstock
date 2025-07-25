"""
PROTOTYPE of find_all_unique_combinations in cutting_optimize_ver1.py
"""

import time

# バックトラッキング法  
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

# 実行例とベンチマーク
if __name__ == "__main__":
    # 与えられた数列
    numbers = [4495, 4495, 
               3585, 3585, 3585, 3585, 3585, 3585, 3585, 3585, 3585, 3585,
               2675, 2675, 2675, 2675, 2675, 2675, 2675, 2675, 2675, 2675,
               2220, 2220,
               1765, 1765, 1765, 1765, 
               1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 
               855, 855, 855, 855, 855, 855, 855, 855, 855, 855, ]
    
    # 目標値を設定
    target_sum = 6000
      
    start_time = time.time()
    combinations = find_all_unique_combinations(target_sum, numbers)
    end_time = time.time()
    
    print(f"{len(combinations)} combinations")
    print(f"Time: {end_time - start_time:.3f} [s]")
    
    # 結果を和でソート
    combinations.sort(key=lambda x: x[1])
    print(combinations)
    
    # print("\n最初の10個の組み合わせ:")
    # for i, (combo, sum_val) in enumerate(combinations[:10]):
    #     print(f"{i+1:3d}: {combo} → 和: {sum_val}")
    
    # if len(combinations) > 10:
    #     print(f"\n... (他に{len(combinations)-10}個)")
    
    # 同じ和を持つ異なる組み合わせの例を表示
    # print("=== 同じ和を持つ異なる組み合わせの例 ===")
    
    # # 和でグループ化
    # from collections import defaultdict
    # sum_groups = defaultdict(list)
    
    # for combo, sum_val in combinations:
    #     sum_groups[sum_val].append(combo)
    
    # # 複数の組み合わせを持つ和を見つける
    # multiple_combo_sums = [(sum_val, combos) for sum_val, combos in sum_groups.items() if len(combos) > 1]
    # multiple_combo_sums.sort(key=lambda x: x[0])  # 和でソート
    
    # print(f"複数の組み合わせを持つ和の数: {len(multiple_combo_sums)}")
    
    # # 最初の5個の例を表示
    # for i, (sum_val, combos) in enumerate(multiple_combo_sums[:5]):
    #     print(f"\n和 {sum_val} の組み合わせ ({len(combos)}通り):")
    #     for j, combo in enumerate(combos):
    #         print(f"  {j+1}: {combo}")
    
    # if len(multiple_combo_sums) > 5:
    #     print(f"\n... (他に{len(multiple_combo_sums)-5}個の和)")
