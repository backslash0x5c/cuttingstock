def find_unique_sum_combinations(nums, target):
    """
    与えられた数列から和がtarget以下となる異なる和の組み合わせを1つずつ求める
    
    Args:
        nums: 数値のリスト
        target: 目標値（この値以下の和を持つ組み合わせを探す）
    
    Returns:
        各和に対して1つの組み合わせを持つ辞書 {和: 組み合わせ}
    """
    # 各和に対して1つの組み合わせを保存
    sum_to_combination = {}
    
    def backtrack(index, current_subset, current_sum):
        # 現在の和がtarget以下で、まだ記録されていない場合
        if current_sum <= target and current_sum not in sum_to_combination:
            if current_sum > 0:  # 空集合は除く
                sum_to_combination[current_sum] = current_subset.copy()
        
        # 現在の和がtargetを超えた場合、これ以上探索しない
        if current_sum > target or index >= len(nums):
            return
        
        # 現在の要素を含まない場合
        backtrack(index + 1, current_subset, current_sum)
        
        # 現在の要素を含む場合
        current_subset.append(nums[index])
        backtrack(index + 1, current_subset, current_sum + nums[index])
        current_subset.pop()  # バックトラック
    
    backtrack(0, [], 0)
    return sum_to_combination


def find_unique_sums_dp(nums, target):
    """
    動的プログラミングを使って異なる和の組み合わせを効率的に求める
    
    Args:
        nums: 数値のリスト
        target: 目標値
    
    Returns:
        各和に対して1つの組み合わせを持つ辞書 {和: 組み合わせ}
    """
    # dp[i] = 和iが作れるかどうか
    dp = [False] * (target + 1)
    dp[0] = True
    
    # 各和に対して1つの組み合わせを保存
    combinations = [[] for _ in range(target + 1)]
    
    for num in nums:
        # 後ろから処理することで、同じ数を複数回使うことを避ける
        for i in range(target, num - 1, -1):
            if dp[i - num] and not dp[i]:  # まだ和iが作れていない場合のみ
                dp[i] = True
                if i == num:
                    combinations[i] = [num]
                else:
                    combinations[i] = combinations[i - num] + [num]
    
    # 結果を辞書形式で返す
    result = {}
    for i in range(1, target + 1):
        if dp[i]:
            result[i] = combinations[i]
    
    return result

# 実行例
if __name__ == "__main__":
    # 与えられた数列
    numbers = [4495, 3585, 3585, 2675, 2675, 2675, 2600, 1765, 1080, 1080, 1080, 1080, 855, 855]
    
    # 目標値を設定
    target_sum = 10000
    
    print(f"数列: {numbers}")
    print(f"目標値: {target_sum}以下")
    print()
    
    # 方法1: バックトラッキング法で異なる和の組み合わせを求める
    print("=== 方法1: バックトラッキング法（異なる和のみ） ===")
    unique_combinations = find_unique_sum_combinations(numbers, target_sum)
    print(unique_combinations)
    exit()
    print(f"異なる和の種類: {len(unique_combinations)}")
    print("\n各和とその組み合わせ例:")
    
    # 和でソートして表示
    sorted_sums = sorted(unique_combinations.items())
    for i, (sum_val, combination) in enumerate(sorted_sums[:15]):  # 最初の15個表示
        print(f"{sum_val:5d}: {combination}")
    
    if len(sorted_sums) > 15:
        print(f"\n... (他に{len(sorted_sums)-15}個)")
    
    print("\n" + "="*50)
    
    # 方法2: 動的プログラミング法
    print("=== 方法2: 動的プログラミング法（異なる和のみ） ===")
    dp_combinations = find_unique_sums_dp(numbers, target_sum)
    
    print(f"異なる和の種類: {len(dp_combinations)}")
    print("\n各和とその組み合わせ例:")
    
    sorted_dp_sums = sorted(dp_combinations.items())
    for i, (sum_val, combination) in enumerate(sorted_dp_sums[:15]):
        print(f"{sum_val:5d}: {combination}")
    
    if len(sorted_dp_sums) > 15:
        print(f"\n... (他に{len(sorted_dp_sums)-15}個)")
    
    print("\n" + "="*50)
