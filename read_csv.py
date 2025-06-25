import pandas as pd

def read_base_pattern(file_path):
    """
    base_pattern.csvを読み込み、径別にベースパターンのリストを作成
    """
    # CSVファイルを読み込み（最初の列をインデックスとして使用）
    df = pd.read_csv(file_path, index_col=0)
    
    base_patterns = {}
    # 各径に対してベースパターンを作成
    for diameter in df.columns:
        patterns = []
        for tekkinkei, row in df.iterrows():
            if row[diameter] == '◯':
                # 文字列から数値に変換（カンマを除去）
                tekkinkei_num = int(str(tekkinkei).replace(',', ''))
                patterns.append(tekkinkei_num)
        
        base_patterns[diameter] = sorted(patterns)
    
    return base_patterns

def read_task_list(file_path):
    """
    task.csvを読み込み、径別にタスクの辞書を作成
    """
    # CSVファイルを読み込み
    df = pd.read_csv(file_path, index_col=1)
    df = df.dropna(subset=[df.columns[0]])  # 最初の列が空でない行のみ
    df = df.drop('Unnamed: 0', axis=1)  # 最初の列を削除

    # dfのNaNを一時的に-1に置き換えてから整数型に変換
    df = df.fillna(-1).astype(int)
    # print(df)

    tasks = df.to_dict(orient='dict')   # 辞書形式に変換
    # 各径のタスクをフィルタリングして、値が0より大きい(-1でないもの)もののみを残す
    tasks = {
        outer_key: {inner_key: value for inner_key, value in inner_dict.items() if value > 0}
        for outer_key, inner_dict in tasks.items()
    }
    # print(tasks)
    return tasks

def getPatterns(diameter):
    """
    メイン処理
    """
    try:
        # ファイルを読み込み
        base_patterns = read_base_pattern('base_pattern.csv')
        tasks = read_task_list('task.csv')
        
        # # データをPythonオブジェクトとして返す
        # return {
        #     'base_patterns': base_patterns,
        #     'tasks': tasks
        # }

        # 径を指定してパターンとタスクを返す
        return {
            'base_patterns': base_patterns[diameter],
            'tasks': tasks[diameter]
        }
        
    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

# 実行例
if __name__ == "__main__":
    pattern = getPatterns('D19')
    available_rods = pattern['base_patterns']
    required_cuts = pattern['tasks']
