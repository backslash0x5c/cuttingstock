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

def display_results(combinations, max_display=20):
    """
    結果を見やすく表示する
    
    Args:
        combinations: 組み合わせのリスト
        max_display: 表示する最大件数
    """
    print(f"見つかった組み合わせ数: {len(combinations)}")
    print(f"最初の{min(max_display, len(combinations))}件を表示:\n")
    
    # 合計値でソート
    combinations.sort(key=lambda x: x[1], reverse=True)
    
    for i, (combo, total) in enumerate(combinations[:max_display]):
        print(f"{i+1:3d}. 合計={total:5d}: {combo}")
    
    if len(combinations) > max_display:
        print(f"\n... 他 {len(combinations) - max_display} 件")

def analyze_combinations(combinations):
    """
    組み合わせの統計情報を分析する
    
    Args:
        combinations: 組み合わせのリスト
    """
    if not combinations:
        print("組み合わせが見つかりませんでした。")
        return
    
    totals = [total for _, total in combinations]
    
    print(f"\n=== 統計情報 ===")
    print(f"組み合わせ総数: {len(combinations)}")
    print(f"最大合計値: {max(totals)}")
    print(f"最小合計値: {min(totals)}")
    print(f"平均合計値: {sum(totals) / len(totals):.1f}")
    
    # 合計値の分布
    print(f"\n=== 合計値の分布 ===")
    ranges = [(0, 1000), (1001, 2000), (2001, 3000), (3001, 4000), 
              (4001, 5000), (5001, 6000), (6001, 7000), (7001, 7500)]
    
    for start, end in ranges:
        count = sum(1 for total in totals if start <= total <= end)
        if count > 0:
            print(f"{start:4d}-{end:4d}: {count:4d}件")

def main():
    """
    メイン処理
    """
    # 例のデータ
    numbers_dict = {
        4495: 2, 2675: 8, 2625: 4, 2220: 1, 1765: 6, 
        1310: 1, 1235: 2, 1195: 1, 1085: 1, 855: 16, 
        805: 2, 400: 8, 325: 2, 310: 2
    }

    numbers_dict2 = {
        4495: 2, 3585: 10, 2675: 10, 2220: 2, 1765: 4, 
        1080: 16, 855: 10
    }
    
    # print("=== 整数組み合わせ探索 ===")
    # print(f"入力データ: {numbers_dict}")
    # print(f"最大合計値: 7500\n")
    
    # 組み合わせを探索
    import time
    start_time = time.time()
    
    combinations = find_combinations_dfs(numbers_dict, max_sum=7500)
    # combinations = find_combinations_dfs(numbers_dict2, max_sum=6000)
    
    end_time = time.time()
    processing_time = end_time - start_time

    combinations.sort(key=lambda x: x[1])
    print(combinations[:100])
    print(len(combinations))
    print(processing_time)
    
    # 結果表示
    # display_results(combinations)
    # analyze_combinations(combinations)
    
    # print(f"\n処理時間: {processing_time:.4f} 秒")

if __name__ == "__main__":
    main()