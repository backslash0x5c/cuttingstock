from read_csv import getPatterns
from itertools import combinations_with_replacement
from collections import defaultdict, Counter
import copy

def find_cutting_combinations(available_rods, required_cuts):
    """
    利用可能な棒から必要な切り出し長さの最適な組み合わせを見つける
    
    Args:
        available_rods: 利用可能な棒の長さのリスト
        required_cuts: 必要な切り出し長さの辞書（キー：長さ、値：数量）
    
    Returns:
        最適化された切り出し組み合わせのリスト
    """
    # 必要な切り出し長さのリストを作成（数量は無視）
    cut_lengths = list(required_cuts.keys())
    cut_lengths = [int(length) for length in cut_lengths]
    
    # 各棒に対する最適な組み合わせを格納
    all_combinations = []
    
    # 各棒の長さに対して可能な組み合わせを探索
    for rod_length in available_rods:
        print(f"\n棒の長さ {rod_length}mm での組み合わせ:")
        print("-" * 50)
        
        rod_combinations = []
        
        # 1つから最大可能数までの組み合わせを試す
        max_cuts = rod_length // min(cut_lengths) if cut_lengths else 0
        
        for num_cuts in range(1, max_cuts + 1):
            # 重複を許可した組み合わせを生成
            for combo in combinations_with_replacement(cut_lengths, num_cuts):
                total_length = sum(combo)
                
                # 棒の長さ以下の場合のみ有効
                if total_length <= rod_length:
                    waste = rod_length - total_length
                    rod_combinations.append({
                        'rod_length': rod_length,
                        'cuts': combo,
                        'total_used': total_length,
                        'waste': waste,
                        'efficiency': (total_length / rod_length) * 100
                    })
        
        # 効率順（余りの少ない順）でソート
        rod_combinations.sort(key=lambda x: x['waste'])
        
        # 上位の組み合わせを表示（余りが少ない順に最大10個）
        for i, combo in enumerate(rod_combinations[:10]):
            cuts_str = ' + '.join(map(str, combo['cuts']))
            print(f"{i+1:2d}. {cuts_str} = {combo['total_used']}mm "
                  f"(余り: {combo['waste']}mm, 効率: {combo['efficiency']:.1f}%)")
        
        all_combinations.extend(rod_combinations)
    
    return all_combinations

def select_cutting_plan(all_combinations, required_cuts):
    """
    必要な切り出し個数を満たす最適な切り出しプランを選択
    
    Args:
        all_combinations: すべての切り出し組み合わせ
        required_cuts: 必要な切り出し長さの辞書（キー：長さ、値：数量）
    
    Returns:
        選択された切り出しプランのリスト
    """
    print("\n必要個数を満たす切り出しプランの選択:")
    print("=" * 60)
    
    # 必要な切り出しを整数に変換
    required = {int(length): quantity for length, quantity in required_cuts.items()}
    remaining = copy.deepcopy(required)
    
    print("必要な切り出し:")
    for length, quantity in required.items():
        print(f"  {length}mm * {quantity}個")
    print()
    
    # 選択されたプラン
    selected_plan = []
    total_waste = 0
    
    # 効率順（余りの少ない順）でソート
    sorted_combinations = sorted(all_combinations, key=lambda x: x['waste'])
    
    plan_num = 1
    
    # 残りの需要がある限りループ
    while any(qty > 0 for qty in remaining.values()):
        best_combo = None
        best_score = -1
        
        # 余りの少ない組み合わせから選択
        for combo in sorted_combinations:                
            # この組み合わせで満たせる需要をカウント
            combo_cuts = Counter(combo['cuts'])
            score = 0
            
            for cut_length, cut_count in combo_cuts.items():
                if cut_length in remaining:
                    # 実際に使える数（残りの需要以下）
                    usable = min(cut_count, remaining[cut_length])
                    score += usable
                else:
                    # 不要な切り出しがある場合はペナルティ
                    score -= cut_count * 0.1
            
            # より多くの需要を満たし、効率が良い組み合わせを選択
            # 0.001->0.01にすると99%以上の効率になる？,その代わり本数が増える
            combined_score = score - (combo['waste'] * 0.015)

            if combined_score > best_score:
                best_score = combined_score
                best_combo = combo
        
        if best_combo is None:
            print("警告: 残りの需要を満たす組み合わせが見つかりません")
            break
        
        # 選択された組み合わせから実際に使用する切り出しを決定
        combo_cuts = Counter(best_combo['cuts'])
        used_cuts = {}
        
        for cut_length, available_count in combo_cuts.items():
            if cut_length in remaining and remaining[cut_length] > 0:
                used_count = min(available_count, remaining[cut_length])
                used_cuts[cut_length] = used_count
                remaining[cut_length] -= used_count
        
        # プランに追加
        plan_info = {
            'plan_number': plan_num,
            'combination': best_combo,
            'used_cuts': used_cuts,
            'total_used_cuts': sum(used_cuts.values())
        }
        selected_plan.append(plan_info)
        total_waste += best_combo['waste']
        
        # 結果表示
        cuts_str = ' + '.join([f"{length}mm * {count}" for length, count in used_cuts.items()])
        combo_str = ' + '.join(map(str, best_combo['cuts']))
        print(f"プラン{plan_num}: 棒{best_combo['rod_length']}mm = {cuts_str}")
        print(f"         組み合わせ: {combo_str}")
        print(f"         使用: {best_combo['total_used']}mm, 余り: {best_combo['waste']}mm")
        
        plan_num += 1
        
        # 残りの需要を表示
        remaining_items = [f"{length}mm * {qty}" for length, qty in remaining.items() if qty > 0]
        if remaining_items:
            print(f"         残り需要: {', '.join(remaining_items)}")
        print()
    
    # 結果サマリー
    print("切り出しプラン完了!")
    print("-" * 30)
    print(f"使用した棒の数: {len(selected_plan)}本")
    print(f"総余り: {total_waste}mm")
    
    if len(selected_plan) > 0:
        avg_efficiency = sum(plan['combination']['efficiency'] for plan in selected_plan) / len(selected_plan)
        print(f"平均効率: {avg_efficiency:.1f}%")

        # 棒ごとの使用状況を表示
        rod_usage = {}
        for plan in selected_plan:
            rod_length = plan['combination']['rod_length']
            if rod_length not in rod_usage:
                rod_usage[rod_length] = 0
            rod_usage[rod_length] += 1
        
        print("棒の使用内訳:")
        for rod_length, count in sorted(rod_usage.items()):
            print(f"  {rod_length}mm * {count}本")
    
    # 最終確認
    # print("\n最終確認 - 切り出し個数:")
    # actual_cuts = Counter()
    # for plan in selected_plan:
    #     for length, count in plan['used_cuts'].items():
    #         actual_cuts[length] += count
    
    # all_satisfied = True
    # for length, required_qty in required.items():
    #     actual_qty = actual_cuts.get(length, 0)
    #     status = "✓" if actual_qty >= required_qty else "✗"
    #     print(f"  {length}mm: 必要{required_qty}個 → 実際{actual_qty}個 {status}")
    #     if actual_qty < required_qty:
    #         all_satisfied = False
    
    # if all_satisfied:
    #     print("\n✓ すべての必要個数が満たされました！")
    #     print(f"✓ {len(selected_plan)}本で完了")
    # else:
    #     print("\n✗ 一部の必要個数が満たされていません")
    
    return selected_plan

# メイン実行部分
if __name__ == "__main__":
    # 設定径: D10, D13, D19
    diameter = 'D19'
    pattern = getPatterns(diameter)
    available_rods = pattern['base_patterns']
    required_cuts = pattern['tasks']

    print("棒材切り出し最適化プログラム:", diameter)
    print("=" * 50)
    
    print(f"利用可能な棒: {available_rods}mm")
    print(f"必要な切り出し長さ: {required_cuts}")
    
    # 組み合わせを探索
    all_combinations = find_cutting_combinations(available_rods, required_cuts)
    
    # 必要個数を満たす切り出しプランを選択
    cutting_plan = select_cutting_plan(all_combinations, required_cuts)
    
    print(f"\n総組み合わせ数: {len(all_combinations)}通り")
