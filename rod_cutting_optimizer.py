import itertools
from typing import List, Tuple, Dict, Set
from functools import lru_cache

def solve_rod_cutting():
    # 利用可能な棒の長さ
    available_rods = [4000, 4500, 5500, 6000]
    
    # 必要な棒の長さと本数
    required_cuts = [
        (4495, 1),  # 4495mm * 1本
        (3585, 2),  # 3585mm * 2本
        (2675, 3),  # 2675mm * 3本
        (2600, 1),  # 2600mm * 1本
        (1765, 1),  # 1765mm * 1本
        (1080, 4),  # 1080mm * 4本
        (855, 2)    # 855mm * 2本
    ]
    
    # 必要な棒を展開（長さのリスト）
    needed_lengths = []
    for length, count in required_cuts:
        needed_lengths.extend([length] * count)
    
    print("必要な棒の長さ:", needed_lengths)
    print("必要な棒の総数:", len(needed_lengths))
    print("必要な総長:", sum(needed_lengths), "mm")
    print()
    
    # 改良された動的計画法による最適解の探索
    best_solution = find_optimal_cutting_plan(available_rods, needed_lengths)
    
    return best_solution

def find_optimal_cutting_plan(available_rods: List[int], needed_pieces: List[int]) -> Dict:
    """
    改良された動的計画法で最適切断計画を見つける
    """
    # 必要な部品をタプルに変換（ハッシュ可能にするため）
    needed_tuple = tuple(sorted(needed_pieces, reverse=True))
    
    # メモ化用のキャッシュ
    memo = {}
    
    def dp_solve(remaining_needs: Tuple[int, ...], used_rods: Tuple[int, ...]) -> Tuple[int, List]:
        """
        動的計画法の再帰関数
        remaining_needs: まだ必要な部品の長さ（ソート済み）
        used_rods: これまでに使用した棒の長さ
        戻り値: (端材の合計, 切断計画)
        """
        # ベースケース：必要な部品がすべて満たされた
        if not remaining_needs:
            return (sum(used_rods) - sum(needed_pieces), [])
        
        # メモ化チェック
        state_key = (remaining_needs, used_rods)
        if state_key in memo:
            return memo[state_key]
        
        best_waste = float('inf')
        best_plan = None
        
        # 各タイプの棒を試す
        for rod_length in available_rods:
            # この棒で切り出せる最適な組み合わせを見つける
            best_cuts = find_best_cuts_for_rod(rod_length, remaining_needs)
            
            if best_cuts['cuts']:  # 何か切り出せる場合
                # 残りの必要部品を更新
                new_remaining = list(remaining_needs)
                for cut in best_cuts['cuts']:
                    if cut in new_remaining:
                        new_remaining.remove(cut)
                
                new_remaining_tuple = tuple(sorted(new_remaining, reverse=True))
                new_used_rods = tuple(sorted(used_rods + (rod_length,)))
                
                # 再帰的に残りを解く
                sub_waste, sub_plan = dp_solve(new_remaining_tuple, new_used_rods)
                total_waste = sub_waste
                
                if total_waste < best_waste:
                    best_waste = total_waste
                    best_plan = [best_cuts] + sub_plan
        
        memo[state_key] = (best_waste, best_plan if best_plan else [])
        return memo[state_key]
    
    # 初期状態から解を求める
    waste, cutting_plan = dp_solve(needed_tuple, ())
    
    if cutting_plan:
        return {
            'success': True,
            'cutting_plan': cutting_plan,
            'waste': waste,
            'total_length': sum([plan['rod_length'] for plan in cutting_plan]),
            'used_length': sum(needed_pieces)
        }
    else:
        return None

@lru_cache(maxsize=10000)
def find_best_cuts_for_rod(rod_length: int, needed_pieces_tuple: Tuple[int, ...]) -> Dict:
    """
    動的計画法を使用して、1本の棒から切り出せる最適な組み合わせを見つける
    LRUキャッシュでメモ化を行う
    """
    if not needed_pieces_tuple:
        return {
            'rod_length': rod_length,
            'cuts': [],
            'used_length': 0,
            'waste': rod_length
        }
    
    # ナップサック問題として解く - この部分は削除（使用しないため）
    # 実際にはknapsack_with_duplicatesを使用
    
    # 重複を考慮した改良版
    used_length, selected_pieces = knapsack_with_duplicates(list(needed_pieces_tuple), rod_length)
    waste = rod_length - used_length
    
    return {
        'rod_length': rod_length,
        'cuts': selected_pieces,
        'used_length': used_length,
        'waste': waste
    }

def knapsack_with_duplicates(items: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    重複ありナップサック問題を解く（同じ長さの部品が複数ある場合に対応）
    """
    # アイテムを種類別にカウント
    item_counts = {}
    for item in items:
        item_counts[item] = item_counts.get(item, 0) + 1
    
    unique_items = list(item_counts.keys())
    n = len(unique_items)
    
    # dp[i][w] = (最大価値, 選択したアイテムのリスト)
    dp = [[None for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # 初期化
    for w in range(capacity + 1):
        dp[0][w] = (0, [])
    
    # DP計算
    for i in range(1, n + 1):
        item = unique_items[i - 1]
        max_count = item_counts[item]
        
        for w in range(capacity + 1):
            # このアイテムを使わない場合
            dp[i][w] = dp[i - 1][w]
            
            # このアイテムを1個以上使う場合
            for count in range(1, max_count + 1):
                if item * count <= w:
                    prev_value, prev_items = dp[i - 1][w - item * count]
                    new_value = prev_value + item * count
                    
                    if new_value > dp[i][w][0]:
                        dp[i][w] = (new_value, prev_items + [item] * count)
    
    return dp[n][capacity]

# 改良されたメイン実行関数
def print_solution(result):
    """結果を整理して表示"""
    if not result or not result.get('success'):
        print("解が見つかりませんでした。")
        return
    
    print("=== 最適解が見つかりました ===")
    print()
    
    # 使用する棒の統計
    rod_usage = {}
    for plan in result['cutting_plan']:
        rod_length = plan['rod_length']
        rod_usage[rod_length] = rod_usage.get(rod_length, 0) + 1
    
    print("使用する棒:")
    total_rods = 0
    for rod_length in sorted(rod_usage.keys()):
        count = rod_usage[rod_length]
        print(f"  {rod_length}mm: {count}本")
        total_rods += count
    print(f"総使用本数: {total_rods}本")
    print()
    
    # 切断計画を表示
    print("切断計画:")
    for i, plan in enumerate(result['cutting_plan']):
        rod_length = plan['rod_length']
        cuts = plan['cuts']
        waste = plan['waste']
        
        print(f"  棒{i+1} ({rod_length}mm):")
        if cuts:
            cuts_str = " + ".join([f"{cut}mm" for cut in sorted(cuts, reverse=True)])
            print(f"    切断: {cuts_str}")
            print(f"    使用: {sum(cuts)}mm")
        else:
            print(f"    切断: なし")
            print(f"    使用: 0mm")
        print(f"    端材: {waste}mm")
        print()
    
    print("=== 結果サマリー ===")
    print(f"総材料長: {result['total_length']}mm")
    print(f"使用長: {result['used_length']}mm")
    print(f"端材合計: {result['waste']}mm")
    print(f"材料効率: {result['used_length']/result['total_length']*100:.1f}%")

# メイン実行
if __name__ == "__main__":
    print("=== 棒材最適切断問題の解決（改良版） ===")
    print()
    
    result = solve_rod_cutting()
    print_solution(result)
