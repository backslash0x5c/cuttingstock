from read_xlsx import extract_cutting_data_from_sheet
import openpyxl
import pandas as pd

target_diameter = 'D22'

# CSVファイルを読み込み
df = pd.read_csv(f'result/csv/result_{target_diameter}.csv', dtype=str)

# pattern列の最大項目数を計算
max_items = 0
for pattern in df['pattern']:
    if pd.notna(pattern):
        # カンマで区切られた項目数を数える
        items = [item.strip().replace('"', '') for item in str(pattern).split(',')]
        max_items = max(max_items, len(items))

# print(f"最大項目数: {max_items}")

# pattern列を展開するための新しい列を作成
for i in range(1, max_items + 1):
    df[f'item_{i}'] = ''

# 各行のpattern列を展開
for index, row in df.iterrows():
    pattern = row['pattern']
    if pd.notna(pattern):
        # カンマで区切って各項目を抽出
        items = [item.strip().replace('"', '') for item in str(pattern).split(',')]
        
        # 各項目を対応する列に設定
        for i, item in enumerate(items):
            df.at[index, f'item_{i + 1}'] = item

# 元のpattern列を削除
df = df.drop('pattern', axis=1)

# item列から全てのユニークな値を取得
unique_values = set()
for i in range(1, max_items + 1):
    col_name = f'item_{i}'
    for value in df[col_name]:
        if value and value.strip():
            unique_values.add(value.strip())

# ユニークな値を降順でソートしてリストに変換
unique_values = sorted(list(unique_values), key=lambda x: int(x) if x.isdigit() else float('inf'), reverse=True)

# カウント表を作成
count_data = []
for index, row in df.iterrows():
    id_value = row['id']
    times_value = int(row['times']) if pd.notna(row['times']) else 0
    
    # 各ユニーク値に対するカウントを初期化
    count_row = {'id': id_value}
    for value in unique_values:
        count_row[value] = 0
    
    # このidで使用されている値をカウント
    for i in range(1, max_items + 1):
        col_name = f'item_{i}'
        item_value = row[col_name]
        if item_value and item_value.strip():
            item_value = item_value.strip()
            if item_value in count_row:
                count_row[item_value] += times_value
    
    count_data.append(count_row)

# カウント結果をDataFrameに変換
count_df = pd.DataFrame(count_data)

# カウント表に合計行を追加
cutting_total_row = {'id': '合計'}
for value in unique_values:
    cutting_total_row[str(value)] = count_df[str(value)].sum()
count_df = pd.concat([count_df, pd.DataFrame([cutting_total_row])], ignore_index=True)
# print(cutting_total_row)
# print()

# 切断集計表からのデータ処理を追加
file_path = '切断集計表.xlsx'

# エクセルファイルを開く
workbook = openpyxl.load_workbook(file_path, data_only=True)

# 各シートから切断データを収集
sheet_cutting_data_all = {}
for sheet_name in workbook.sheetnames:
    worksheet = workbook[sheet_name]
    
    # シートから切断データを抽出
    sheet_cutting_data = extract_cutting_data_from_sheet(worksheet, target_diameter)
    
    if sheet_cutting_data:
        # print(f"'{sheet_name}' : {sheet_cutting_data}")
        sheet_cutting_data_all[sheet_name] = sheet_cutting_data

workbook.close()

# 切断集計表データから全てのユニークな値を取得
cutting_unique_values = set()
for sheet_data in sheet_cutting_data_all.values():
    cutting_unique_values.update(sheet_data.keys())

# ユニークな値を降順でソート
cutting_unique_values = sorted(list(cutting_unique_values), key=lambda x: int(x) if str(x).isdigit() else float('inf'), reverse=True)

# 切断集計表のカウント表を作成
cutting_count_data = []
for sheet_name, sheet_data in sheet_cutting_data_all.items():
    # 各ユニーク値に対するカウントを初期化
    count_row = {'シート名': sheet_name}
    for value in cutting_unique_values:
        count_row[str(value)] = sheet_data.get(value, 0)
    
    cutting_count_data.append(count_row)

# 切断集計表のカウント結果をDataFrameに変換
cutting_count_df = pd.DataFrame(cutting_count_data)

# 切断集計表カウント表に合計行を追加
project_total_row = {'シート名': '合計'}
for value in cutting_unique_values:
    project_total_row[str(value)] = cutting_count_df[str(value)].sum()
cutting_count_df = pd.concat([cutting_count_df, pd.DataFrame([project_total_row])], ignore_index=True)

# 差分の計算用
sum_sheet2 = list(cutting_total_row.values())[1:]
sum_sheet3 = list(project_total_row.values())[1:]
diff = [i-j for i, j in zip(sum_sheet2, sum_sheet3)]
diff_df = pd.DataFrame([diff], columns=cutting_unique_values)

# ExcelWriterを使用して複数シートで出力
output_filename = f'result/xlsx/result_sheet_{target_diameter}.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # 最適切断パターンを最初のシートに保存
    df.to_excel(writer, sheet_name='最適切断パターン', index=False)
    
    # 切断種類のカウント結果を2番目のシートに保存
    count_df.to_excel(writer, sheet_name='出力結果集計表', index=False)
    
    # 切断集計表のカウント結果を3番目のシートに保存
    cutting_count_df.to_excel(writer, sheet_name='切断指示集計表', index=False)

    # 差分を新規シートに保存
    diff_df.to_excel(writer, sheet_name='出力結果と切断指示の差分', index=False)
