"""
ver3.pyのペナルティパラメータのグリッドサーチ用プログラム
"""

from read_csv import getPatterns
from collections import Counter
from tqdm import tqdm
import itertools, copy, csv

desplay = -1  # デバッグ:正の値で詳細表示

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
        if desplay > 0:
            print(f"\n棒の長さ {rod_length}mm での組み合わせ:")
            print("-" * 50)
        
        rod_combinations = []
        
        # 1つから最大可能数までの組み合わせを試す
        max_cuts = rod_length // min(cut_lengths) if cut_lengths else 0
        
        for num_cuts in range(1, max_cuts + 1):
            # 重複を許可した組み合わせを生成
            for combo in itertools.combinations_with_replacement(cut_lengths, num_cuts):
                total_length = sum(combo)
                
                # 棒の長さ以下の場合のみ有効
                if total_length <= rod_length:
                    waste = rod_length - total_length
                    rod_combinations.append({
                        'rod_length': rod_length,
                        'cuts': combo,
                        'total_used': total_length,
                        'waste': waste,
                        'efficiency': total_length * 100 / rod_length
                    })
        
        # 歩留り順（ロスの少ない順）でソート
        rod_combinations.sort(key=lambda x: x['waste'])
        
        # 上位の組み合わせを表示（ロスが少ない順に最大10個）
        if desplay > 0:
            for i, combo in enumerate(rod_combinations[:10]):
                cuts_str = ' + '.join(map(str, combo['cuts']))
                print(f"{i+1:2d}. {cuts_str} = {combo['total_used']}mm "
                    f"(ロス: {combo['waste']}mm, 歩留り: {combo['efficiency']:.2f}%)")
        
        all_combinations.extend(rod_combinations)
    
    return all_combinations

def select_cutting_plan(all_combinations, required_cuts, over_penalty, waste_penalty):
    """
    必要な切り出し個数を満たす最適な切り出しプランを選択
    
    Args:
        all_combinations: すべての切り出し組み合わせ
        required_cuts: 必要な切り出し長さの辞書（キー：長さ、値：数量）
    
    Returns:
        選択された切り出しプランのリスト
    """
    
    # 必要な切り出しを整数に変換
    required = {int(length): quantity for length, quantity in required_cuts.items()}
    remaining = copy.deepcopy(required)
    
    if desplay > 0:
        print("必要な切り出し:")
        for length, quantity in required.items():
            print(f"  {length}mm * {quantity}個")
        print()
    
    # 選択されたプラン
    plan_num = 0
    selected_plan = []
    total_waste = 0
    total_stored_cuts = []
    
    # 歩留り順（ロスの少ない順）でソート
    sorted_combinations = sorted(all_combinations, key=lambda x: x['waste'])
    
    error_status = 1
    # 残りの需要がある限りループ
    while any(qty > 0 for qty in remaining.values()):
        best_combo = None
        best_score = -float('inf')
        
        # ロスの少ない組み合わせから選択
        for combo in sorted_combinations:
            # この組み合わせで満たせる需要をカウント
            combo_cuts = Counter(combo['cuts'])
            score = 0
            
            for cut_length, cut_count in combo_cuts.items():
                if cut_length in remaining:
                    # 実際に使える数（残りの需要以下）
                    score += min(cut_count, remaining[cut_length])
                    # 余分な切り出しはペナルティ
                    score -= (cut_count - remaining[cut_length]) * over_penalty if cut_count > remaining[cut_length] else 0
                else:
                    # 不要な切り出しがある場合はペナルティ
                    score -= cut_count * over_penalty
            
            # より多くの需要を満たし、歩留りが良い組み合わせを選択
            # 0.001->0.01にすると99%以上の歩留りになる？,その代わり本数が増える
            score -= (combo['waste'] * waste_penalty)

            if score > best_score:
                best_score = score
                best_combo = combo
        
        if best_combo is None:
            print("警告: 残りの需要を満たす組み合わせが見つかりません")
            error_status = -1
            break
        
        # 選択された組み合わせから実際に使用する切り出しを決定
        combo_cuts = Counter(best_combo['cuts'])
        used_cuts = {}
        
        for cut_length, available_count in combo_cuts.items():
            if cut_length in remaining and remaining[cut_length] > 0:
                used_count = min(available_count, remaining[cut_length])
                used_cuts[cut_length] = used_count
                remaining[cut_length] -= used_count
        
        if not used_cuts:
            # print("警告: 残りの需要を満たす切り出しがありません")
            # print("Please change some penalties.")
            error_status = -1
            break

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
        combo_str = ' + '.join(map(str, best_combo['cuts']))    # ある切出しの全カット
        cuts_str_list = [f"{key}" for key, count in used_cuts.items() for _ in range(count)]
        cuts_str = ' + '.join(cuts_str_list) # ある切出しの需要を満たすカット
        cuts_int_tuple = tuple(map(int, cuts_str_list))

        stored_cuts = [x for x in best_combo['cuts'] if x not in cuts_int_tuple] # 残材本数 = 全切出し - 使用する切出し
        total_stored_cuts.append(stored_cuts)

        plan_num += 1
        # print(f"{plan_num}本目: {best_combo['rod_length']}mm = {combo_str} + {best_combo['waste']}")

        if desplay > 0:
            print(f"    需要を満たすカット: {cuts_str}")
            print(f"    使用: {best_combo['total_used']}mm, ロス: {best_combo['waste']}mm, 残材: {len(stored_cuts)}本")
            
            # 残りの需要を表示
            remaining_items = [f"{length}mm * {qty}" for length, qty in remaining.items() if qty > 0]
            if remaining_items:
                print(f"    残り需要: {', '.join(remaining_items)}")
            print()
    
    # 結果サマリー
    if error_status > 0:
        # print("-" * 30)
        # print(f"使用した棒の数: {len(selected_plan)}本")
        # print(f"ロスの合計: {total_waste}mm")
        
        if len(selected_plan) > 0:
            avg_efficiency = sum(plan['combination']['total_used'] for plan in selected_plan) * 100 / sum(plan['combination']['rod_length'] for plan in selected_plan)
            # print(f"歩留り: {avg_efficiency:.2f}%")

            # # 棒ごとの使用状況を表示
            # rod_usage = {}
            # for plan in selected_plan:
            #     rod_length = plan['combination']['rod_length']
            #     if rod_length not in rod_usage:
            #         rod_usage[rod_length] = 0
            #     rod_usage[rod_length] += 1
            
            # print("棒の使用内訳:")
            # for rod_length, count in sorted(rod_usage.items()):
            #     print(f"  {rod_length}mm * {count}本")
            # print()

            # # 発生した残材の本数と内容を表示
            total_stored_cuts = tuple(itertools.chain.from_iterable(total_stored_cuts))
            # print(f"発生した残材の本数: {len(total_stored_cuts)}本")
            # print(f"発生した残材: {total_stored_cuts}")

            return avg_efficiency, len(total_stored_cuts)
    else:
        return None, None

# メイン実行部分
if __name__ == "__main__":
    # 設定径: D10, D13, D19
    diameter = 'D10'
    pattern = getPatterns(diameter)
    available_rods = pattern['base_patterns']
    required_cuts = pattern['tasks']

    print("棒材切り出し最適化プログラム:", diameter)
    print("=" * 50)
    
    # print(f"利用可能な棒: {available_rods}mm")
    # print(f"必要な切り出し長さ: {required_cuts}")
    
    # 組み合わせを探索
    all_combinations = find_cutting_combinations(available_rods, required_cuts)

    if desplay > 0:
        print(f"\n探索された組み合わせ数: {len(all_combinations)}通り")

    eff = []
    stored =[]
    n = 20
    for i in tqdm(range(n)):
        eff2 = []
        stored2 = []
        for j in range(n):
            # scoreのペナルティ設定
            over_penalty = 0.01 * (i+1) # 余分な切り出しに対するペナルティ 0.001-0.01
            waste_penalty = 0.001 * (j+1)  # ロス長に対するペナルティ 0.001-0.01
            # あるケースでの切り出しプランのロス率
            efficiency, stored_num = select_cutting_plan(all_combinations, required_cuts, over_penalty, waste_penalty)
            eff2.append(round(efficiency,2) if efficiency != None else efficiency)
            stored2.append(stored_num)
        eff.append(eff2)
        stored.append(stored2)
    
    csv_path = f"table_{diameter}.csv"

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(eff)
        writer.writerows(stored)